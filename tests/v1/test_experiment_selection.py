from dataclasses import replace

import pytest

from latent_working_memory.data_preparation.experiment import prepare_experiment
from latent_working_memory.data_preparation.pipeline import prepare_fineweb
from latent_working_memory.v1.data import EpisodeIndex
from latent_working_memory.v1.sampling import PretrainSampler
from latent_working_memory.v1.training import learning_rate_at


def test_selection_equal_cells_mixture_and_curriculum(
    tmp_path, tokenizer, tiny_config, preparation_records, preparation_recipe
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
    prepare_fineweb(preparation_records, tokenizer, config, raw, preparation_recipe)
    spec = {
        "sources": {"first": str(raw / "semantic"), "second": str(raw / "random")},
        "runs": {"one": {"first": 1}, "two": {"second": 1}, "both": {"first": 0.5, "second": 0.5}},
        "seed": 3,
    }
    report = prepare_experiment(spec, config, tokenizer, tmp_path / "selected")
    assert len({r["samples"] for r in report["runs"].values()}) == 1
    q = report["quota_per_task_length"]["train"]
    index = EpisodeIndex(tmp_path / "selected/runs/both/train.jsonl")
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
