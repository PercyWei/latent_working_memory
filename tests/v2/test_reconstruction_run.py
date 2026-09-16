"""从原始 Parquet 到完整 epoch、两阶段衔接与精确续训。"""

import argparse
from dataclasses import asdict
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
from latent_working_memory.v2.pretrain.checkpoint import prune_checkpoints


def make_experiment(tmp_path, tiny_base):
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
    selection = SelectionConfig(
        str(parquet),
        capacity=2,
        continuation_tokens=2,
        max_documents=100,
        split_fractions=(0.6, 0.2, 0.2),
        warmup={"train": 5, "dev": 1, "test": 1},
        multiround={"train": 5, "dev": 1, "test": 1},
    )
    training = TrainingConfig(
        objective="ae_lm",
        warmup_epochs=1,
        multiround_epochs=1,
        global_batch_size=2,
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
    # Stop inside warm-up; rebuild fixed data, restore dropout RNG and continue across stages.
    interrupted = run_training(arguments(experiment, tmp_path / "resumed", stop=2))
    assert not interrupted["complete"]
    checkpoint_path = tmp_path / "resumed" / "checkpoints" / "step-000002.pt"
    resumed = run_training(arguments(experiment, tmp_path / "resumed", resume=checkpoint_path))
    assert resumed["complete"] and resumed["completed_steps"] == 6
    expected = torch.load(uninterrupted["checkpoint"], weights_only=False, map_location="cpu")
    actual = torch.load(resumed["checkpoint"], weights_only=False, map_location="cpu")
    torch.testing.assert_close(actual["codec"], expected["codec"], rtol=0, atol=0)
    torch.testing.assert_close(actual["optimizer"], expected["optimizer"], rtol=0, atol=0)
    assert actual["cursor"] == expected["cursor"]
    log = [
        json.loads(line) for line in (tmp_path / "resumed" / "train.jsonl").read_text().splitlines()
    ]
    assert [r["samples"] for r in log] == [2, 2, 1, 2, 2, 1]
    assert sum(r["samples"] for r in log) == 10
    assert "datasets" not in actual and "token_ids" not in actual
    assert not list((tmp_path / "resumed").rglob("*.parquet"))
    evaluation = json.loads((tmp_path / "resumed" / "test.json").read_text())
    assert evaluation["metrics"]["trajectories"] == 1
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


def _distributed_run(rank, rendezvous, experiment, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    run_training(arguments(experiment, output))
    dist.destroy_process_group()


def test_two_rank_stage_transfer_and_full_epoch_tail(tiny_base, tmp_path):
    experiment = make_experiment(tmp_path, tiny_base)
    output = tmp_path / "distributed"
    mp.spawn(
        _distributed_run,
        args=(str(tmp_path / "rendezvous"), experiment, output),
        nprocs=2,
        join=True,
    )
    result = json.loads((output / "training-result.json").read_text())
    assert result["complete"] and result["completed_steps"] == 6
    log = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [r["samples"] for r in log] == [2, 2, 1, 2, 2, 1]
    assert [r["stage"] for r in log] == ["warmup"] * 3 + ["multiround"] * 3


@pytest.mark.parametrize("objective", ["ae", "ae_lm"])
def test_static_baseline_trains_only_single_writes_and_evaluates_shared_trajectories(
    tiny_base, tmp_path, objective
):
    torch.set_num_threads(1)
    experiment = make_experiment(tmp_path, tiny_base)
    raw = json.loads(experiment.read_text())
    raw["training"].update(objective=objective, warmup_epochs=2, multiround_epochs=0)
    experiment.write_text(json.dumps(raw))
    output = tmp_path / "static"
    result = run_training(arguments(experiment, output))
    assert result["complete"] and result["completed_steps"] == 6
    log = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert all(r["stage"] == "warmup" for r in log)
    assert [r["samples"] for r in log] == [2, 2, 1, 2, 2, 1]
    summary = json.loads((output / "data-summary.json").read_text())
    assert summary["multiround"]["train"]["trajectories"] == 0
    assert summary["warmup"]["train"]["rounds"] == {"1": 5}
    for file in [*sorted((output / "dev").glob("*.json")), output / "test.json"]:
        evaluation = json.loads(file.read_text())
        assert all(3 <= row["depth"] <= 5 for row in evaluation["samples"])
        assert "trajectory_ae" in evaluation["metrics"]
        assert "trajectory_lm" in evaluation["metrics"]
        assert not any("ppl" in key for key in evaluation["metrics"])
    assert "generation/final_round_exact_match" in evaluation["metrics"]
