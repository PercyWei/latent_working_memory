import json
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from latent_working_memory.data_preparation.pretrain.pipeline import prepare_fineweb
from latent_working_memory.v1.checkpoint import load_model_checkpoint
from latent_working_memory.v1.pretrain.curriculum import (
    distribution_at,
    epoch_distribution,
    maximal_quotas,
    validate_curriculum,
)
from latent_working_memory.v1.pretrain.data_selection import select_experiment
from latent_working_memory.v1.pretrain.sampling import EpochSampler
from latent_working_memory.v1.pretrain.training import run_pretraining
from latent_working_memory.v1.pretrain import training as training_module
from pretrain_reference import LegacyPretrainTrainer


def test_maximal_integer_quotas_and_independent_schedules():
    training = {
        "source_schedule": [
            {"epoch": 1, "weights": {"semantic": 1, "random": 1}},
            {"epoch": 3, "weights": {"semantic": 0, "random": 1}},
        ],
        "task_schedule": [
            {"epoch": 1, "weights": {"ae": 1, "continuation": 0}},
            {"epoch": 2, "weights": {"ae": 1, "continuation": 1}},
        ],
        "length_schedule": [
            {"epoch": 1, "weights": {"32": 3, "64": 1}},
            {"epoch": 3, "weights": {"32": 1, "64": 1}},
        ],
    }
    validate_curriculum(training, ("semantic", "random"), (32, 64))
    assert distribution_at(training["length_schedule"], 2, True) == {
        "32": Fraction(5, 8),
        "64": Fraction(3, 8),
    }
    for epoch in (1, 2, 3, 4):
        probabilities = epoch_distribution(training, epoch)
        assert sum(probabilities.values()) == 1
        available = {k: 70 + i * 13 for i, k in enumerate(probabilities)}
        quotas, bottlenecks, unit = maximal_quotas(available, probabilities, 8)
        n = sum(quotas.values())
        feasible = [
            total
            for total in range(8, sum(available.values()) + 1, 8)
            if all(
                (total * p).denominator == 1 and total * p <= available[k]
                for k, p in probabilities.items()
            )
        ]
        assert n == max(feasible) and n % unit == 0 and bottlenecks
        assert all(quotas[k] == n * p for k, p in probabilities.items())
        assert {k[1] for k in probabilities} == ({"ae"} if epoch == 1 else {"ae", "continuation"})
        if epoch >= 3:
            assert {k[0] for k in probabilities} == {"random"}
    with pytest.raises(ValueError, match="no complete epoch"):
        maximal_quotas({}, epoch_distribution(training, 1), 8)
    training["task_schedule"][1]["weights"]["ae"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        validate_curriculum(training, ("semantic", "random"), (32, 64))


@pytest.mark.parametrize("natural", [False, True])
@pytest.mark.parametrize("mode", ["sample", "mean"])
def test_epoch_nonreplacement_resume_and_fixed_evaluation(
    parquet_source,
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    epoch_selection,
    mode,
    natural,
):
    root = tmp_path / "data"
    config = replace(tiny_config, compression_mode=mode)
    prepare_fineweb(
        parquet_source(preparation_records), tokenizer, config, root, preparation_recipe
    )
    spec = json.loads(
        epoch_selection(root, config, {v: v for v in ("semantic", "random")}).read_text()
    )
    if natural:
        spec["training"] = {
            "input_tokens": {"min": 1, "max": config.max_input_tokens},
            "source_schedule": None,
            "task_schedule": None,
            "length_schedule": None,
        }
    indices, report = select_experiment(spec, config, tokenizer)
    index = indices["train", "train"]
    # Training retains the full eligible pool; epoch quotas are applied only by the sampler.
    assert len(index.ids) == sum(report["task_source_counts"]["train/train"].values())
    sampler = EpochSampler(index, tokenizer, config, spec["training"], 5, 4, 3)
    first = sampler.sample_batch()
    state = sampler.state_dict()
    expected = [sampler.sample_batch() for _ in range(sampler.total_steps - 1)]
    restored = EpochSampler(index, tokenizer, config, spec["training"], 99, 4, 3)
    restored.load_state_dict(state)
    assert [restored.sample_batch() for _ in expected] == expected
    with pytest.raises(StopIteration):
        restored.sample_batch()
    batches = [first, *expected]
    offset = 0
    orders = []
    for epoch in range(1, 4):
        nsteps = sampler.epoch_report(epoch)["steps"]
        chosen = []
        counts = Counter()
        for batch in batches[offset : offset + nsteps]:
            ids = list(dict.fromkeys(e.episode.episode_id for e in batch))
            assert len(ids) == 4
            assert sum(e.loss_weight for e in batch) == pytest.approx(4)
            for sample_id in ids:
                example = next(e for e in batch if e.episode.episode_id == sample_id)
                chosen.append(sample_id)
                counts[
                    (
                        example.episode.sources[0].provenance["boundary_variant"],
                        example.episode.reads[0].task,
                        64,
                    )
                ] += 1
            if mode == "sample":
                assert len(batch) == 4
            else:
                assert all(
                    sum(e.loss_weight for e in batch if e.episode.episode_id == i)
                    == pytest.approx(1)
                    for i in ids
                )
                assert len({(e.episode.episode_id, e.capacity) for e in batch}) == len(batch)
        assert len(chosen) == len(set(chosen))
        observed = {(None, None, None): sum(counts.values())} if natural else counts
        assert observed == sampler.plans[epoch - 1][1]
        orders.append(chosen)
        offset += nsteps
    assert orders[0] != orders[1]
    alternate = EpochSampler(
        index,
        tokenizer,
        replace(config, compression_mode=("mean" if mode == "sample" else "sample")),
        spec["training"],
        5,
        4,
        3,
    )
    for batch in batches:
        other = alternate.sample_batch()
        assert list(dict.fromkeys(e.episode.episode_id for e in other)) == list(
            dict.fromkeys(e.episode.episode_id for e in batch)
        )
    # Evaluation balancing affects only dev/test, with strict per-source task quotas.
    spec["evaluation"]["balance_task_lengths"] = True
    balanced, balanced_report = select_experiment(spec, config, tokenizer)
    assert balanced["train", "train"].ids == indices["train", "train"].ids
    for source in spec["sources"]:
        for split in ("dev", "test"):
            assert balanced[source, split].tasks.count("ae") == balanced[source, split].tasks.count(
                "continuation"
            )
            assert balanced_report["samples_per_cell"][f"{source}/{split}"] > 0
    spec["evaluation"]["balance_task_lengths"] = False
    # Training schedules must not change dev/test panel membership.
    spec["training"]["source_schedule"] = [
        {"epoch": 1, "weights": {"semantic": 0, "random": 1}}
    ]
    spec["training"]["input_tokens"] = {"min": 16, "max": 32}
    spec["training"]["task_schedule"] = [{"epoch": 1, "tasks": ["ae"], "weights": None}]
    repeated, _ = select_experiment(spec, config, tokenizer)
    for source in spec["sources"]:
        for split in ("dev", "test"):
            assert repeated[source, split].ids == indices[source, split].ids
    # Fixed evaluation quotas remain strict.
    spec["evaluation"]["samples_per_source"]["dev"] = 100000
    with pytest.raises(ValueError, match="insufficient data"):
        select_experiment(spec, config, tokenizer)


@pytest.mark.parametrize("natural", [False, True])
def test_warmup_complete_training_keeps_optimizer_and_resolves_final_checkpoint(
    parquet_source,
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    epoch_selection,
    natural,
    monkeypatch,
):
    model_dir = tmp_path / "model"
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    config = replace(
        tiny_config,
        model_name_or_path=str(model_dir),
        eval_generation_examples=0,
        split_fractions=(0.6, 0.2, 0.2),
    )
    recipe = replace(preparation_recipe, samples_per_task=(16, 16, 16), candidates_per_document=16)
    root = tmp_path / "data"
    prepare_fineweb(parquet_source(preparation_records), tokenizer, config, root, recipe)
    path = epoch_selection(root, config)
    spec = json.loads(path.read_text())
    spec["training"]["task_schedule"] = [
        {"epoch": 1, "weights": {"ae": 1, "continuation": 0}},
        {"epoch": 2, "weights": {"ae": 1, "continuation": 1}},
    ]
    if natural:
        spec["training"] = {
            "source_schedule": None,
            "task_schedule": [
                {"epoch": 1, "tasks": ["ae"], "weights": None},
                {"epoch": 2, "tasks": ["ae", "continuation"], "weights": None},
            ],
            "length_schedule": None,
        }
    path.write_text(json.dumps(spec))
    out = tmp_path / "warmup"
    result = run_pretraining(
        config, path, out, torch.device("cpu"), epochs=2, max_samples_per_epoch=8
    )
    checkpoint = load_model_checkpoint(result.final_checkpoint)
    logs = [
        json.loads(line)
        for p in out.glob("train-from-*.jsonl")
        for line in p.read_text().splitlines()
    ]
    first = [r for r in logs if r["epoch"] == 1]
    second = [r for r in logs if r["epoch"] == 2]
    assert first and second
    assert all(s["ae_nll"] is not None for r in first for s in r["samples"])
    if not natural:
        assert sum(s["ae_nll"] is not None for r in second for s in r["samples"]) == sum(
            s["lm_nll"] is not None for r in second for s in r["samples"]
        )
    for epoch_rows in (first, second):
        report = epoch_rows[0]["epoch_selection"]
        assert sum(report["actual_cells"].values()) == report["samples"]
    assert all(
        s["step"].item() == result.completed_steps
        for s in checkpoint.optimizer_state["state"].values()
    )
    summary = json.loads((out / "training-result.json").read_text())
    assert summary["complete"] and summary["completed_epochs"] == 2
    assert summary["final_checkpoint"] == str(result.final_checkpoint.resolve())
    assert (
        sum(e["steps"] for e in json.loads((out / "epoch-plan.json").read_text()))
        == result.completed_steps
    )

    resumed_out = tmp_path / "warmup-resumed"
    boundary = json.loads((out / "epoch-plan.json").read_text())[0]["steps"]
    # A pre-Accelerate checkpoint can continue with the unchanged canonical state.
    def legacy_trainer(config, backbone, writer, device, accelerator):
        return LegacyPretrainTrainer(config, backbone, writer, device)

    with monkeypatch.context() as patch:
        patch.setattr(training_module, "PretrainTrainer", legacy_trainer)
        first_segment = run_pretraining(
            config,
            path,
            resumed_out,
            torch.device("cpu"),
            epochs=2,
            max_samples_per_epoch=8,
            stop_after_steps=boundary,
        )
    assert not json.loads((resumed_out / "training-result.json").read_text())["complete"]
    resumed_run = run_pretraining(
        config,
        path,
        resumed_out,
        torch.device("cpu"),
        epochs=2,
        max_samples_per_epoch=8,
        resume=first_segment.final_checkpoint,
    )
    resumed = load_model_checkpoint(resumed_run.final_checkpoint)
    torch.testing.assert_close(checkpoint.model_state, resumed.model_state, rtol=0, atol=0)
    torch.testing.assert_close(checkpoint.optimizer_state, resumed.optimizer_state, rtol=0, atol=0)
    assert checkpoint.progress["sampler"] == resumed.progress["sampler"]


def test_epoch_sample_cap_keeps_exact_quotas_and_batch_divisibility():
    spec = json.loads(
        Path("configs/v1/pretrain/qwen2.5-3b-instruct_mixed-2048/selection.json").read_text()
    )
    sizes = []
    for epoch in range(1, 4):
        probabilities = epoch_distribution(spec["training"], epoch)
        available = {key: 100000 for key in probabilities}
        quotas, bottlenecks, unit = maximal_quotas(available, probabilities, 8, 32000)
        total = sum(quotas.values())
        assert total <= 32000 < total + unit
        assert total % 8 == 0 and not bottlenecks
        assert all(n == total * probabilities[key] for key, n in quotas.items())
        sizes.append(total)
    assert sizes == [32000, 31680, 31992]
    assert sum(sizes) // 8 == 11959
    probabilities = {"a": Fraction(1, 3), "b": Fraction(2, 3)}
    assert maximal_quotas({"a": 8, "b": 16}, probabilities, 8, 32000)[0] == {"a": 8, "b": 16}
    with pytest.raises(ValueError, match="no complete epoch"):
        maximal_quotas({"a": 8, "b": 16}, probabilities, 8, 23)
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer or null"):
            maximal_quotas({"a": 8, "b": 16}, probabilities, 8, limit)
