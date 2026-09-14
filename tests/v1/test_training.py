from __future__ import annotations

from dataclasses import replace
import json
import re
import pytest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig, LlamaForCausalLM

from latent_working_memory.v1.checkpoint import load_model_checkpoint
from latent_working_memory.v1.pretrain.evaluate import main as evaluate_main
from latent_working_memory.data_preparation.pretrain.pipeline import prepare_fineweb
from latent_working_memory.v1.pretrain.sampling import EpochSampler
from latent_working_memory.v1.pretrain.data_selection import select_experiment
from latent_working_memory.v1.pretrain.training import (
    PretrainTrainer,
    pretrain_forward,
    run_pretraining,
)


def test_joint_objective_uses_one_write_and_updates_all_four_modules(
    parquet_source,
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    components,
    monkeypatch,
    epoch_selection,
):
    preparation_records = [
        dict(row, text=re.sub(r"\b\w+\b", lambda m: m[0] + str(i), row["text"]))
        for i, row in enumerate(preparation_records)
    ]
    data = tmp_path / "data"
    prepare_fineweb(
        parquet_source(preparation_records),
        tokenizer,
        tiny_config,
        data,
        preparation_recipe,
    )
    spec = json.loads(epoch_selection(data, tiny_config).read_text())
    indices, _ = select_experiment(spec, tiny_config, tokenizer)
    sampler = EpochSampler(
        indices["train", "train"], tokenizer, tiny_config, spec["training"], 3, 2, 1
    )
    examples = sampler.sample_batch()
    backbone, writer = components
    calls = []
    original = backbone.text_features

    def traced(units, starts):
        calls.append(units)
        return original(units, starts)

    monkeypatch.setattr(backbone, "text_features", traced)
    output = pretrain_forward(tiny_config, backbone, writer, examples)
    assert calls == [[e.episode.input_ids for e in examples]]
    output.loss.backward()
    for parameter in (
        backbone.input_projection.weight,
        backbone.memory_projection.weight,
        writer.output_projection.weight,
    ):
        assert parameter.grad.abs().sum() > 0
    before = writer.output_projection.weight.detach().clone()
    trainer = PretrainTrainer(tiny_config, backbone, writer, torch.device("cpu"))
    metrics = trainer.step(examples)
    assert metrics["loss"] > 0 and metrics["gradient_norm"] > 0
    assert not torch.equal(before, writer.output_projection.weight)


def _assert_equal_nested(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_equal_nested(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            _assert_equal_nested(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("compression_mode", ["sample", "mean"])
def test_real_tiny_llama_train_evaluate_resume_matches_uninterrupted_run(
    parquet_source,
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    epoch_selection,
    compression_mode,
):
    model_dir = tmp_path / "tiny-llama"
    torch.manual_seed(3)
    model = LlamaForCausalLM(
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
            attention_dropout=0.0,
        )
    )
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    config = replace(
        tiny_config,
        model_name_or_path=str(model_dir),
        compression_mode=compression_mode,
        gradient_checkpointing=True,
        split_fractions=(0.6, 0.2, 0.2),
        warmup_steps=1,
        lr_decay_steps=4,
    )
    preparation_recipe = replace(
        preparation_recipe, samples_per_task=(16, 16, 16), candidates_per_document=16
    )
    preparation_records = [
        dict(row, text=re.sub(r"\b\w+\b", lambda m: m[0] + str(i), row["text"]))
        for i, row in enumerate(preparation_records)
    ]
    data = tmp_path / "data"
    prepare_fineweb(
        parquet_source(preparation_records),
        tokenizer,
        config,
        data,
        preparation_recipe,
    )
    selection = epoch_selection(data, config, {"dev": "semantic"})
    data = data / "semantic"
    full = run_pretraining(
        config,
        selection,
        tmp_path / "full",
        torch.device("cpu"),
        epochs=2,
        stop_after_steps=2,
        save_every=1,
    )
    first = run_pretraining(
        config,
        selection,
        tmp_path / "resumed",
        torch.device("cpu"),
        epochs=2,
        stop_after_steps=1,
        swanlab_mode="offline",
        swanlab_group="lwm-pretrain-test",
    )
    identity_path = tmp_path / "resumed/swanlab.json"
    first_swanlab_id = json.loads(identity_path.read_text())["id"]
    assert json.loads(identity_path.read_text())["project"] == "latent-working-memory-v1"
    resumed = run_pretraining(
        config,
        selection,
        tmp_path / "resumed",
        torch.device("cpu"),
        epochs=2,
        stop_after_steps=2,
        resume=first.final_checkpoint,
        swanlab_mode="offline",
        swanlab_group="lwm-pretrain-test",
    )
    expected, actual = (
        load_model_checkpoint(full.final_checkpoint),
        load_model_checkpoint(resumed.final_checkpoint),
    )
    _assert_equal_nested(expected.model_state, actual.model_state)
    _assert_equal_nested(expected.optimizer_state, actual.optimizer_state)
    _assert_equal_nested(expected.progress, actual.progress)
    assert full.dev_metrics == resumed.dev_metrics
    groups = resumed.dev_metrics["groups"]
    assert {"all/ae/memory", "all/ae/wrong_memory", "all/continuation/no_memory"} <= groups.keys()
    assert len([key for key in groups if key.startswith("length_ratio/")]) > 6
    assert groups["all/ae/memory"]["generated_reads"] > 0
    assert "bleu_4" in groups["all/ae/memory"]
    assert "all/continuation/full_context" in groups
    assert resumed.dev_metrics["training_input_tokens"] == actual.progress["input_tokens"]
    assert (tmp_path / "resumed/dev-step-000002.jsonl").exists()
    assert json.loads(identity_path.read_text())["id"] == first_swanlab_id
    evaluate_main(
        [
            "--checkpoint",
            str(resumed.final_checkpoint),
            "--data-dir",
            str(data),
            "--output-dir",
            str(tmp_path / "test-evaluation"),
            "--device",
            "cpu",
            "--split",
            "test",
            "--examples",
            "4",
            "--generation-examples",
            "1",
            "--swanlab-group",
            "lwm-pretrain-test",
            "--swanlab-mode",
            "offline",
        ]
    )
    test_metrics = json.loads((tmp_path / "test-evaluation/test-step-000002.json").read_text())
    assert json.loads((tmp_path / "test-evaluation/swanlab.json").read_text())["project"] == (
        "latent-working-memory-v1"
    )
    assert test_metrics["split"] == "test"
    assert test_metrics["training_input_tokens"] == actual.progress["input_tokens"]
    assert "nll_gap_to_full_context" in test_metrics["comparisons"]["all/continuation"]

    multi_selection = epoch_selection(
        data.parent, config, {"first": "semantic", "second": "random"}
    )
    multi = run_pretraining(
        config,
        multi_selection,
        tmp_path / "multi",
        torch.device("cpu"),
        epochs=2,
        stop_after_steps=1,
    )
    assert set(multi.dev_metrics) == {"first", "second"}
    assert (tmp_path / "multi/first/dev-step-000001.json").exists()
    assert (tmp_path / "multi/second/dev-step-000001.json").exists()

    for name, steps, resume_step in (
        ("distributed-full", 2, None),
        ("distributed-resumed", 1, None),
        ("distributed-resumed", 2, 1),
    ):
        mp.spawn(
            _distributed_train_worker,
            args=(
                str(tmp_path / f"rendezvous-{name}-{steps}"),
                config,
                multi_selection,
                tmp_path / name,
                steps,
                resume_step,
            ),
            nprocs=2,
            join=True,
        )
    expected = load_model_checkpoint(
        tmp_path / "distributed-full/checkpoints/pretrain-step-000002.pt"
    )
    actual = load_model_checkpoint(
        tmp_path / "distributed-resumed/checkpoints/pretrain-step-000002.pt"
    )
    _assert_equal_nested(expected.model_state, actual.model_state)
    _assert_equal_nested(expected.optimizer_state, actual.optimizer_state)
    _assert_equal_nested(expected.progress, actual.progress)
    assert actual.progress["run_identity"]["world_size"] == 2
    assert len(actual.progress["rank_rng_states"]) == 2


def _distributed_train_worker(rank, rendezvous, config, data, output, steps, resume_step):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    run_pretraining(
        config,
        data,
        output,
        torch.device("cpu"),
        epochs=2,
        stop_after_steps=steps,
        save_every=1,
        resume=output / f"checkpoints/pretrain-step-{resume_step:06d}.pt" if resume_step else None,
    )
    dist.destroy_process_group()
