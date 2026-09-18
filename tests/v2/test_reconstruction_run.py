"""从原始 Parquet 到完整 epoch、两阶段衔接与精确续训。"""

import argparse
from dataclasses import asdict
from pathlib import Path
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from latent_working_memory.v2.memory_codec import CodecConfig
from latent_working_memory.v2.pretrain.config import SelectionConfig, TrainingConfig
from latent_working_memory.v2.pretrain.train import run_training
from latent_working_memory.v2.pretrain.prepare_data import DataPreparationConfig, prepare_dataset
from latent_working_memory.v2.pretrain.checkpoint import prune_checkpoints


def make_experiment(tmp_path, tiny_base, train_samples=12):
    parquet = tmp_path / "fineweb.parquet"
    records = [
        {
            "id": str(i),
            "url": f"https://example.org/document/{i}",
            "text": f"article{i} " + "red blue sky water " * (8 + i % 4),
        }
        for i in range(100)
    ]
    pq.write_table(pa.Table.from_pylist(records), parquet)
    model = CodecConfig(
        str(tiny_base),
        encoder_layers=2,
        alignment_layers=1,
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.1,
        gradient_checkpointing=True,
        attention_implementation="eager",
    )
    preparation = DataPreparationConfig(
        str(parquet),
        capacity=2,
        continuation_tokens=2,
        continuation_reserve_tokens=3,
        max_documents=100,
        split_fractions=(0.6, 0.2, 0.2),
        warmup={"train": train_samples, "dev": 8, "test": 8},
        multiround={"train": train_samples, "dev": 8, "test": 8},
    )
    prepare_dataset(preparation, tmp_path / "dataset")
    selection = SelectionConfig(str(tmp_path / "dataset"))
    training = TrainingConfig(
        objective="ae_lm",
        lm_ratio=0.5,
        warmup_epochs=1,
        multiround_epochs=1,
        global_batch_size=2,
        micro_batch_size=2,
        generation_samples=1,
        save_every=2,
        eval_every=2,
    )
    for name, config in (("model", model), ("selection", selection)):
        (tmp_path / f"{name}.json").write_text(json.dumps(asdict(config)))
    experiment = tmp_path / "experiment.json"
    experiment.write_text(
        json.dumps(
            {"model": "model.json", "selection": "selection.json", "training": asdict(training)}
        )
    )
    return experiment


def arguments(experiment, output, stop=None, resume=None):
    return argparse.Namespace(
        experiment=experiment,
        output_dir=output,
        device="cpu",
        resume=resume,
        stop_after_steps=stop,
        swanlab_mode="disabled",
        swanlab_project="latent-working-memory-v2",
        swanlab_group=None,
        swanlab_tag=[],
    )


def test_checkpoint_retention_preserves_stage_endpoints_and_other_artifacts(tmp_path):
    for step in (1000, 2000, 3000, 4000, 5000):
        (tmp_path / f"step-{step:06d}.pt").write_text("checkpoint")
    (tmp_path / "notes.json").write_text("{}")
    prune_checkpoints(tmp_path, 2, {2000})
    assert {p.name for p in tmp_path.iterdir()} == {
        "step-002000.pt",
        "step-004000.pt",
        "step-005000.pt",
        "notes.json",
    }


def test_raw_data_epochs_and_resume_match_uninterrupted(tiny_base, tmp_path):
    torch.set_num_threads(1)
    experiment = make_experiment(tmp_path, tiny_base)
    uninterrupted = run_training(arguments(experiment, tmp_path / "full"))
    # Reload the same prepared data, restore dropout RNG and continue across stages.
    interrupted = run_training(arguments(experiment, tmp_path / "resumed", stop=2))
    assert not interrupted["complete"]
    checkpoint_path = tmp_path / "resumed" / "checkpoints" / "step-000002.pt"
    resumed = run_training(arguments(experiment, tmp_path / "resumed", resume=checkpoint_path))
    assert resumed["complete"] and resumed["completed_steps"] == uninterrupted["completed_steps"]
    expected = torch.load(uninterrupted["checkpoint"], weights_only=False, map_location="cpu")
    actual = torch.load(resumed["checkpoint"], weights_only=False, map_location="cpu")
    torch.testing.assert_close(actual["codec"], expected["codec"], rtol=0, atol=0)
    torch.testing.assert_close(actual["optimizer"], expected["optimizer"], rtol=0, atol=0)
    assert actual["cursor"] == expected["cursor"]
    architecture = actual["run"]["resolved_architecture"]
    assert architecture["shared_backbone"] is False and architecture["reader_adapter"] is None
    assert architecture["encoder_layers"] == architecture["decoder_layers"] == 2
    log = [
        json.loads(line) for line in (tmp_path / "resumed" / "train.jsonl").read_text().splitlines()
    ]
    full_log = [
        json.loads(line) for line in (tmp_path / "full" / "train.jsonl").read_text().splitlines()
    ]
    for expected_record, actual_record in zip(full_log, log, strict=True):
        for key in ("loss", "ae", "lm", "ae_samples", "lm_samples", "ae_tokens", "lm_tokens"):
            assert actual_record[key] == expected_record[key]
    assert sum(r["ae_samples"] for r in log) > 0
    assert sum(r["lm_samples"] for r in log) > 0
    stats = json.loads((tmp_path / "resumed" / "data-summary.json").read_text())
    assert sum(r["samples"] for r in log) == sum(stats[s]["train"]["trajectories"] for s in stats)
    assert all(r["samples"] in (1, 2) for r in log)
    assert "datasets" not in actual and "token_ids" not in actual
    assert not list((tmp_path / "resumed").rglob("*.parquet"))
    evaluation = json.loads((tmp_path / "resumed" / "test.json").read_text())
    assert evaluation["metrics"]["trajectories"] == stats["multiround"]["test"]["trajectories"]
    assert "generation" in evaluation["samples"][0]
    assert set(evaluation["samples"][0]["generation"]) == {
        "prediction",
        "reference",
        "final_round_exact_match",
        "hit_limit",
    }
    assert "final_ae" not in evaluation["metrics"]
    assert "final_lm" not in evaluation["metrics"]
    last_round = evaluation["samples"][0]["depth"]
    for objective in ("ae", "lm"):
        assert f"round/{last_round}/{objective}_nll" in evaluation["metrics"]


def _distributed_run(rank, rendezvous, experiment, output, stop=None, resume=None):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    run_training(arguments(experiment, output, stop, resume))
    dist.destroy_process_group()


def test_two_rank_fixed_batch_stage_transfer_and_resume(tiny_base, tmp_path):
    experiment = make_experiment(tmp_path, tiny_base, train_samples=40)
    raw = json.loads(experiment.read_text())
    raw["training"].update(global_batch_size=32, micro_batch_size=8, multiround_epochs=2)
    experiment.write_text(json.dumps(raw))
    output = tmp_path / "distributed"
    resumed_output = tmp_path / "resumed-distributed"
    for label, destination, stop, resume in (
        ("full", output, None, None),
        ("stop", resumed_output, 2, None),
        ("resume", resumed_output, None, resumed_output / "checkpoints/step-000002.pt"),
    ):
        mp.spawn(
            _distributed_run,
            args=(str(tmp_path / label), experiment, destination, stop, resume),
            nprocs=2,
            join=True,
        )
    result = json.loads((output / "training-result.json").read_text())
    resumed = json.loads((resumed_output / "training-result.json").read_text())
    assert result["complete"] and resumed["complete"]
    log = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    resumed_log = [
        json.loads(line) for line in (resumed_output / "train.jsonl").read_text().splitlines()
    ]
    assert len(log) == len(resumed_log) == result["total_steps"]
    assert log[0]["stage"] == "warmup" and log[-1]["stage"] == "multiround"
    assert {r["global_epoch"] for r in log} == {1, 2, 3}
    assert all(
        r["max_microbatch_size"] == 8 and r["microbatches"] == 4 for r in log if r["samples"] == 32
    )
    for a, b in zip(log, resumed_log, strict=True):
        for key in ("step", "stage", "samples", "microbatches", "loss", "ae", "lm"):
            assert a[key] == b[key]
    full_checkpoint = torch.load(result["checkpoint"], weights_only=False)
    resumed_checkpoint = torch.load(resumed["checkpoint"], weights_only=False)
    for key in ("codec", "optimizer"):
        torch.testing.assert_close(full_checkpoint[key], resumed_checkpoint[key], rtol=0, atol=0)
    assert full_checkpoint["cursor"] == resumed_checkpoint["cursor"]


@pytest.mark.parametrize("objective", ["ae", "ae_lm"])
def test_static_baseline_trains_only_single_writes_and_evaluates_shared_trajectories(
    tiny_base, tmp_path, objective
):
    torch.set_num_threads(1)
    experiment = make_experiment(tmp_path, tiny_base)
    raw = json.loads(experiment.read_text())
    raw["training"].update(
        objective=objective,
        lm_ratio=0.5 if objective == "ae_lm" else 0,
        warmup_epochs=2,
        multiround_epochs=0,
    )
    experiment.write_text(json.dumps(raw))
    output = tmp_path / "static"
    result = run_training(arguments(experiment, output))
    assert result["complete"] and result["completed_steps"] == result["total_steps"]
    log = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert all(r["stage"] == "warmup" for r in log)
    summary = json.loads((output / "data-summary.json").read_text())
    assert summary["multiround"]["train"]["trajectories"] == 0
    count = summary["warmup"]["train"]["trajectories"]
    assert summary["warmup"]["train"]["rounds"] == {"1": count}
    assert sum(r["samples"] for r in log) == 2 * count
    assert [r["samples"] for r in log if r["epoch"] == 1] == [
        r["samples"] for r in log if r["epoch"] == 2
    ]
    for file in [*sorted((output / "dev").glob("*.json")), output / "test.json"]:
        evaluation = json.loads(file.read_text())
        assert all(3 <= row["depth"] <= 5 for row in evaluation["samples"])
        assert "trajectory_ae" in evaluation["metrics"]
        assert "trajectory_lm" in evaluation["metrics"]
        assert not any("ppl" in key for key in evaluation["metrics"])
    assert "generation/final_round_exact_match" in evaluation["metrics"]


@pytest.mark.parametrize(
    "name",
    [
        "ae-warmup",
        "ae-lm-warmup",
        "ae_dynamic",
        "ae-lm_dynamic",
        "ae-static",
        "ae-lm-static",
    ],
)
def test_all_six_formal_training_policies_complete(tiny_base, tmp_path, name):
    torch.set_num_threads(1)
    experiment = make_experiment(tmp_path, tiny_base, train_samples=40)
    source = (
        Path(__file__).resolve().parents[2]
        / "configs/v2/pretrain"
        / f"qwen3-4b_pooling_{name}"
        / "experiment.json"
    )
    raw = json.loads(experiment.read_text())
    raw["training"] = json.loads(source.read_text())["training"]
    experiment.write_text(json.dumps(raw))
    output = tmp_path / "full-policy"
    result = run_training(arguments(experiment, output))
    assert result["complete"] and result["completed_epochs"] == 3
    assert (output / "test.json").exists()
    rows = [json.loads(x) for x in (output / "train.jsonl").read_text().splitlines()]
    cfg = raw["training"]
    assert all(r["max_microbatch_size"] <= cfg["micro_batch_size"] for r in rows)
    assert any(r["samples"] == 32 for r in rows)
    assert all(
        r["max_microbatch_size"] == cfg["micro_batch_size"] for r in rows if r["samples"] == 32
    )
    assert {r["global_epoch"] for r in rows} == {1, 2, 3}
    stats = json.loads((output / "data-summary.json").read_text())
    for stage, epochs in [
        ("warmup", cfg["warmup_epochs"]),
        ("multiround", cfg["multiround_epochs"]),
    ]:
        if epochs:
            assert (
                sum(r["samples"] for r in rows if r["stage"] == stage)
                == epochs * stats[stage]["train"]["trajectories"]
            )


@pytest.mark.parametrize(
    "steps,offset,expected",
    [
        (999, 0, {250, 500, 750, 999}),
        (1000, 999, {1249, 1499, 1749, 1999}),
        (1000, 1000, {1250, 1500, 1750, 2000}),
        (999, 1998, {2248, 2498, 2748, 2997}),
        (2, 0, {1, 2}),
    ],
)
def test_epoch_evaluation_points_are_even_and_deduplicated(steps, offset, expected):
    config = TrainingConfig(eval_every=None, evals_per_epoch=4)
    assert config.evaluation_steps(steps, offset) == expected


def test_legacy_evaluation_and_checkpoint_config_are_preserved():
    config = TrainingConfig()
    legacy = asdict(config)
    legacy.pop("evals_per_epoch")
    assert config.to_dict() == legacy
    assert TrainingConfig(**legacy).to_dict() == legacy
    assert config.evaluation_steps(999, 0) == {999}
    assert config.evaluation_steps(1000, 999) == {1000, 1999}


@pytest.mark.parametrize(
    "settings",
    [
        {"eval_every": None, "evals_per_epoch": None},
        {"eval_every": 1000, "evals_per_epoch": 4},
        {"eval_every": None, "evals_per_epoch": 0},
        {"eval_every": None, "evals_per_epoch": True},
        {"eval_every": None, "evals_per_epoch": 1.5},
    ],
)
def test_evaluation_configuration_has_one_positive_policy(settings):
    with pytest.raises(ValueError):
        TrainingConfig(**settings)


def test_epoch_evaluation_resume_keeps_same_points_and_checkpoint_policy(tiny_base, tmp_path):
    torch.set_num_threads(1)
    experiment = make_experiment(tmp_path, tiny_base)
    raw = json.loads(experiment.read_text())
    raw["training"].update(eval_every=None, evals_per_epoch=4, save_every=1000)
    experiment.write_text(json.dumps(raw))
    full_dir, resumed_dir = tmp_path / "full-epoch-eval", tmp_path / "resumed-epoch-eval"
    full = run_training(arguments(experiment, full_dir))
    run_training(arguments(experiment, resumed_dir, stop=2))
    resumed = run_training(
        arguments(experiment, resumed_dir, resume=resumed_dir / "checkpoints/step-000002.pt")
    )
    plan = json.loads((full_dir / "epoch-plan.json").read_text())
    expected = sorted(step for epoch in plan["epochs"] for step in epoch["evaluation_steps"])
    for directory in [full_dir, resumed_dir]:
        observed = sorted(int(p.stem.split("-")[1]) for p in (directory / "dev").glob("*.json"))
        assert observed == expected
    for filename in (full_dir / "dev").glob("*.json"):
        assert json.loads(filename.read_text()) == json.loads(
            (resumed_dir / "dev" / filename.name).read_text()
        )
    full_state = torch.load(full["checkpoint"], weights_only=False)
    resumed_state = torch.load(resumed["checkpoint"], weights_only=False)
    torch.testing.assert_close(full_state["codec"], resumed_state["codec"], rtol=0, atol=0)
    torch.testing.assert_close(full_state["optimizer"], resumed_state["optimizer"], rtol=0, atol=0)
    assert full_state["cursor"] == resumed_state["cursor"]
    # More dev points do not cause extra scheduled checkpoints.
    assert len(list((full_dir / "checkpoints").glob("*.pt"))) == 2
