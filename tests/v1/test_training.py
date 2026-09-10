from __future__ import annotations

from dataclasses import replace
import json

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from latent_working_memory.v1.checkpoint import load_model_checkpoint
from latent_working_memory.v1.data import EpisodeIndex
from latent_working_memory.v1.evaluate import main as evaluate_main
from latent_working_memory.data_preparation.pipeline import prepare_fineweb
from latent_working_memory.v1.sampling import PretrainSampler
from latent_working_memory.v1.training import PretrainTrainer, pretrain_forward, run_pretraining


def test_joint_objective_uses_one_write_and_updates_all_four_modules(
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    components,
    monkeypatch,
):
    data = tmp_path / "data"
    prepare_fineweb(
        preparation_records,
        tokenizer,
        tiny_config,
        data,
        preparation_recipe,
    )
    data = data / "semantic"
    sampler = PretrainSampler(EpisodeIndex(data / "train.jsonl"), tokenizer, tiny_config)
    examples = [sampler.sample(0), sampler.sample(0)]
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


def test_real_tiny_llama_train_evaluate_resume_matches_uninterrupted_run(
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
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
        gradient_checkpointing=True,
        split_fractions=(0.6, 0.2, 0.2),
    )
    preparation_recipe = replace(
        preparation_recipe, samples_per_task=(16, 16, 16), candidates_per_document=16
    )
    data = tmp_path / "data"
    prepare_fineweb(
        preparation_records,
        tokenizer,
        config,
        data,
        preparation_recipe,
    )
    data = data / "semantic"
    full = run_pretraining(
        config, data, tmp_path / "full", torch.device("cpu"), max_steps=2, save_every=1
    )
    first = run_pretraining(
        config,
        data,
        tmp_path / "resumed",
        torch.device("cpu"),
        max_steps=1,
        swanlab_mode="offline",
        swanlab_group="lwm-pretrain-test",
    )
    identity_path = tmp_path / "resumed/swanlab.json"
    first_swanlab_id = json.loads(identity_path.read_text())["id"]
    resumed = run_pretraining(
        config,
        data,
        tmp_path / "resumed",
        torch.device("cpu"),
        max_steps=2,
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
    assert len([key for key in groups if key.startswith("capacity/")]) > 6
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
    assert test_metrics["split"] == "test"
    assert test_metrics["training_input_tokens"] == actual.progress["input_tokens"]
    assert "nll_gap_to_full_context" in test_metrics["comparisons"]["all/continuation"]

    multi = run_pretraining(
        config,
        data,
        tmp_path / "multi",
        torch.device("cpu"),
        max_steps=1,
        evaluation_dirs={"first": data, "second": data.parent / "random"},
    )
    assert set(multi.dev_metrics) == {"first", "second"}
    assert (tmp_path / "multi/first/dev-step-000001.json").exists()
    assert (tmp_path / "multi/second/dev-step-000001.json").exists()
