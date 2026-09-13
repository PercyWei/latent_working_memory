import torch
from transformers import LlamaConfig, LlamaForCausalLM
from latent_working_memory.v1.training import run_pretraining
from latent_working_memory.v1.evaluate import main as evaluate_main
from latent_working_memory.v1.checkpoint import load_model_checkpoint
from dataclasses import replace
import json
import re

import pytest

from latent_working_memory.v1.data_selection import select_experiment
from latent_working_memory.data_preparation.pretrain.pipeline import prepare_fineweb
from latent_working_memory.v1.sampling import PretrainSampler
from latent_working_memory.v1.training import learning_rate_at


@pytest.mark.parametrize("shared_names", [True, False])
def test_selection_equal_cells_mixture_and_curriculum(
    parquet_source,
    tmp_path, tokenizer, tiny_config, preparation_records, preparation_recipe, shared_names
):
    config = replace(
        tiny_config,
        input_length_bounds=preparation_recipe.length_bounds,
        input_length_weights=(0.5, 0.3, 0.2),
        input_length_weights_end=(1 / 3,) * 3,
        input_length_curriculum_steps=10,
        max_input_tokens=64,
        max_continuation_tokens=64,
        warmup_steps=2,
        lr_decay_steps=10,
    )
    raw = tmp_path / "raw"
    prepare_fineweb(parquet_source(preparation_records), tokenizer, config, raw, preparation_recipe)
    names = ("first", "second") if shared_names else ("one", "two")
    spec = {
        "sources": {"first": str(raw / "semantic"), "second": str(raw / "random")},
        "runs": {
            names[0]: {"first": 1},
            names[1]: {"second": 1},
            "both": {"first": 0.5, "second": 0.5},
        },
        "seed": 3,
        "balance_task_lengths": True,
        "samples_per_split": {"train": 12, "dev": 6, "test": 6},
    }
    before = {str(p): p.stat().st_size for p in raw.rglob("*") if p.is_file()}
    indices, report = select_experiment(spec, config, tokenizer)
    q = 2
    assert {report["counts"][f"{name}/train"] for name in spec["runs"]} == {12}
    index = indices["both", "train"]
    counts = {}
    for i in range(len(index.offsets)):
        e = index[i]
        key = (
            e.sources[0].provenance["boundary_variant"],
            e.reads[0].task,
            next(b for b in config.input_length_bounds if len(e.input_ids) <= b),
        )
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 12 and set(counts.values()) == {q // 2}
    sampler = PretrainSampler(index, tokenizer, config)
    assert list(sampler.length_weights(0).values()) == [0.5, 0.3, 0.2]
    assert list(sampler.length_weights(10).values()) == [1 / 3] * 3
    assert list(sampler.length_weights(5).values()) == pytest.approx(
        [(a + 1 / 3) / 2 for a in [0.5, 0.3, 0.2]]
    )
    for step in range(5):
        sampler.sample(step)
    state = sampler.state_dict()
    expected = [sampler.sample(step).episode.episode_id for step in range(12)]
    restored = PretrainSampler(index, tokenizer, config)
    restored.load_state_dict(state)
    assert expected == [restored.sample(step).episode.episode_id for step in range(12)]
    assert learning_rate_at(config, 0) == config.learning_rate / 2
    assert learning_rate_at(config, 2) == config.learning_rate
    assert learning_rate_at(config, 10) == pytest.approx(
        config.learning_rate * config.min_lr_fraction
    )

    repeated, _ = select_experiment(spec, config, tokenizer)
    assert all(index.ids == repeated[key].ids for key, index in indices.items())
    assert before == {str(p): p.stat().st_size for p in raw.rglob("*") if p.is_file()}


def test_source_dataset_name_cannot_describe_a_different_mixture(tmp_path, tiny_config, tokenizer):
    spec = {
        "sources": {"first": "unused/first", "second": "unused/second"},
        "runs": {"first": {"first": 0.5, "second": 0.5}},
        "seed": 3,
        "balance_task_lengths": True,
        "samples_per_split": {"train": 12, "dev": 6, "test": 6},
    }
    with pytest.raises(ValueError, match="named after a source"):
        select_experiment(spec, tiny_config, tokenizer)


def test_count_all_and_insufficient_selection(
    parquet_source,
    tmp_path, tokenizer, tiny_config, preparation_records, preparation_recipe
):
    raw = tmp_path / "raw"
    prepare_fineweb(parquet_source(preparation_records), tokenizer, tiny_config, raw, preparation_recipe)
    spec = {
        "sources": {"semantic": str(raw / "semantic")},
        "runs": {"semantic": {"semantic": 1}},
        "seed": 5,
        "balance_task_lengths": False,
        "samples_per_split": {"train": None, "dev": None, "test": None},
    }
    all_indices, _ = select_experiment(spec, tiny_config, tokenizer)
    spec["samples_per_split"] = {"train": 5, "dev": 2, "test": 2}
    small, _ = select_experiment(spec, tiny_config, tokenizer)
    assert small["semantic", "train"].ids == all_indices["semantic", "train"].ids[:5] or set(
        small["semantic", "train"].ids
    ) <= set(all_indices["semantic", "train"].ids)
    assert len(small["semantic", "train"].ids) == 5
    spec["samples_per_split"]["train"] = 100000
    with pytest.raises(ValueError, match="insufficient data"):
        select_experiment(spec, tiny_config, tokenizer)


def test_selection_train_resume_and_evaluate(
    parquet_source,
    tmp_path, tokenizer, tiny_config, preparation_records, preparation_recipe
):
    model = tmp_path / "model"
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
    ).save_pretrained(model)
    tokenizer.save_pretrained(model)
    cfg = replace(
        tiny_config,
        model_name_or_path=str(model),
        eval_generation_examples=0,
        input_length_weights=None,
        split_fractions=(0.6, 0.2, 0.2),
    )
    raw = tmp_path / "raw"
    # Keep fragments from distinct fixture documents distinct after Parquet shuffling.
    preparation_records = [
        dict(row, text=re.sub(r"\b\w+\b", lambda m: m[0] + str(i), row["text"]))
        for i, row in enumerate(preparation_records)
    ]
    recipe = replace(preparation_recipe, samples_per_task=(16, 16, 16), candidates_per_document=16)
    prepare_fineweb(parquet_source(preparation_records), tokenizer, cfg, raw, recipe)
    spec = {
        "sources": {v: str(raw / v) for v in ("semantic", "random")},
        "runs": {"mixed": {"semantic": 0.5, "random": 0.5}},
        "seed": 7,
        "balance_task_lengths": False,
        "samples_per_split": {"train": 12, "dev": None, "test": None},
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(spec))
    run = tmp_path / "train"
    first = run_pretraining(
        cfg, None, run, torch.device("cpu"), max_steps=1, data_selection=path, data_run="mixed"
    )
    assert json.loads((run / "data-selection.json").read_text()) == spec
    result = run_pretraining(
        cfg,
        None,
        run,
        torch.device("cpu"),
        max_steps=2,
        data_selection=run / "data-selection.json",
        data_run="mixed",
        resume=first.final_checkpoint,
    )
    assert load_model_checkpoint(result.final_checkpoint).progress["next_step"] == 2
    out = tmp_path / "evaluation"
    evaluate_main(
        [
            "--checkpoint",
            str(result.final_checkpoint),
            "--data-selection",
            str(path),
            "--output-dir",
            str(out),
            "--split",
            "test",
            "--device",
            "cpu",
            "--generation-examples",
            "0",
        ]
    )
    assert all((out / v / "test-step-000002.json").exists() for v in spec["sources"])
    assert json.loads((out / "data-selection.json").read_text()) == spec

    single = tmp_path / "single-evaluation"
    evaluate_main(
        [
            "--checkpoint",
            str(result.final_checkpoint),
            "--data-selection",
            str(path),
            "--evaluation-source",
            "semantic",
            "--output-dir",
            str(single),
            "--split",
            "test",
            "--device",
            "cpu",
            "--generation-examples",
            "0",
        ]
    )
    assert (single / "test-step-000002.json").exists()
    assert not (single / "random").exists()
