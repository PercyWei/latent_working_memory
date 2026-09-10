from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.data_preparation.fineweb import (
    SemanticSpans,
    data_contract,
    span_episode,
)
from latent_working_memory.data_preparation.truncation import RandomSpans
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.sampling import capacity_weights, read_tokens


def test_default_length_and_fraction_boundaries():
    recipe = PreparationConfig()
    assert recipe.accepts_lengths(32, None)
    assert recipe.accepts_lengths(4096, None)
    assert not recipe.accepts_lengths(31, None)
    assert not recipe.accepts_lengths(4097, None)
    for x, y in ((32, 32), (4096, 4096), (300, 700), (700, 300)):
        assert recipe.accepts_lengths(x, y)
    for x, y in ((32, 31), (31, 32), (4096, 4097), (300, 701), (700, 299)):
        assert not recipe.accepts_lengths(x, y)
    assert recipe.target_length_range(700)[0] == 300
    assert recipe.target_length_range(300)[1] == 700


@pytest.mark.parametrize(
    "values",
    [
        {"min_sample_tokens": 0},
        {"max_sample_tokens": 8192},
        {"lm_prefix_fraction": (0, 0.7)},
        {"lm_prefix_fraction": (0.7, 0.3)},
        {"lm_prefix_fraction": (0.3, float("nan"))},
    ],
)
def test_invalid_length_contract_is_rejected(values):
    with pytest.raises(ValueError):
        PreparationConfig(**values)


def test_balanced_quotas_and_recipe_round_trip(tmp_path):
    recipe = PreparationConfig(samples_per_task=(100, 5, 7))
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe.to_dict()))
    assert PreparationConfig.load(path) == recipe
    for split, quota in zip(("train", "dev", "test"), recipe.samples_per_task, strict=True):
        hist = recipe.balanced_histogram()[split]
        assert hist["ae"] == hist["continuation"]
        counts = list(hist["ae"].values())
        assert sum(counts) == quota and max(counts) - min(counts) <= 1


def test_two_4096_token_halves_are_legal_without_training_capacity_filter(
    tiny_config,
    tokenizer,
    source_records,
):
    sentence = "A " * 4095 + "."
    record = dict(source_records[0], text=sentence + "\n" + sentence)
    recipe = PreparationConfig()
    sampler = SemanticSpans(record, tokenizer, tiny_config, recipe)
    lm = sampler.sample("continuation", 4096, 4096, random.Random(0))
    assert lm is not None
    assert len(lm.input_ids) == 4096
    target = lm.reads[0].references[0].text
    assert len(tokenizer.encode(target, add_special_tokens=False)) == 4096
    p = lm.sources[0].provenance
    assert p["parent_char_span"] == [0, len(record["text"])]
    assert p["x_char_span"][1] == p["y_char_span"][0]
    ae = sampler.sample("ae", 4096, 4096, random.Random(0))
    assert ae is not None and ae.reads[0].references[0].text == sentence
    training_config = replace(
        tiny_config,
        max_input_tokens=4096,
        max_continuation_tokens=4096,
        write_context_tokens=8192,
        read_context_tokens=8192,
        k_limit=4096,
    )
    ae_tokens, lm_tokens = read_tokens(ae, tokenizer)
    assert set(capacity_weights(training_config, 4096, ae_tokens, lm_tokens, 0)) == {
        512,
        1024,
        2048,
    }
    # The same recipe applies to random spans, including an 8192-token parent.
    assert (
        span_episode(
            record,
            tokenizer,
            tiny_config,
            recipe,
            0,
            len(sentence),
            len(record["text"]),
            "random",
            "random",
        )
        is not None
    )
    random_sampler = RandomSpans(record, tokenizer, tiny_config, recipe)
    rng = random.Random(7)
    samples = [random_sampler.sample("continuation", 4096, 4096, rng) for _ in range(12)]
    samples = [row for row in samples if row is not None]
    assert samples
    for row in samples:
        target_length = len(
            tokenizer.encode(row.reads[0].references[0].text, add_special_tokens=False)
        )
        assert len(row.input_ids) == 4096 and 32 <= target_length <= 4096
        assert recipe.accepts_lengths(4096, target_length)


def test_lm_sampling_never_builds_an_ae_parent(
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    monkeypatch,
):
    tasks = []

    def observed_span(
        record, tokenizer, config, recipe, start, end, target_end, variant, granularity
    ):
        tasks.append("ae" if target_end is None else "continuation")
        return span_episode(
            record, tokenizer, config, recipe, start, end, target_end, variant, granularity
        )

    monkeypatch.setattr(
        "latent_working_memory.data_preparation.fineweb.span_episode", observed_span
    )
    sampler = SemanticSpans(preparation_records[0], tokenizer, tiny_config, preparation_recipe)
    rng = random.Random(5)
    results = [sampler.sample("continuation", 9, 32, rng) for _ in range(20)]
    assert any(row is not None for row in results)
    assert tasks and set(tasks) == {"continuation"}


@pytest.mark.parametrize("sampler_class", [SemanticSpans, RandomSpans])
def test_ae_draws_do_not_change_the_lm_random_stream(
    sampler_class,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
):
    sampler = sampler_class(preparation_records[0], tokenizer, tiny_config, preparation_recipe)
    baseline_rng = random.Random(9)
    expected = [sampler.sample("continuation", 9, 32, baseline_rng) for _ in range(12)]
    lm_rng, ae_rng = random.Random(9), random.Random(11)
    actual = []
    for _ in range(12):
        sampler.sample("ae", 9, 32, ae_rng)
        actual.append(sampler.sample("continuation", 9, 32, lm_rng))
    assert any(row is not None for row in expected)
    assert actual == expected


def test_memory_limit_increases_without_a_one_to_one_compression_ratio():
    config = ExperimentConfig()
    assert config.k_limit == 4096
    assert config.pretrain_compression_ratios == (2, 4, 8)
    assert data_contract(config) == data_contract(
        replace(config, k_limit=512, max_input_tokens=256)
    )
