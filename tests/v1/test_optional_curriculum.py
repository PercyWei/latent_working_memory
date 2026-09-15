from collections import Counter
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import random
from types import SimpleNamespace

import pytest

from latent_working_memory.v1.pretrain.curriculum import (
    distribution_at,
    epoch_distribution,
    maximal_quotas,
    tasks_at,
    validate_curriculum,
)
from latent_working_memory.v1.pretrain.data_selection import selection_metadata
from latent_working_memory.v1.pretrain.sampling import EpochSampler


def natural_training():
    return {
        "input_tokens": {"min": 16, "max": 64},
        "source_schedule": None,
        "task_schedule": None,
        "length_schedule": None,
    }


def make_index(cells):
    entries = []
    for (source, task, length), count in cells.items():
        for _ in range(count):
            i = len(entries)
            entries.append((source, i, str(i), str(i), str(i), str(i), task, length))
    return SimpleNamespace(entries=entries, sources={"semantic": None, "random": None})


def sampler_for(index, training, config, tokenizer, cap=None, seed=17, epochs=3):
    return EpochSampler(
        index, tokenizer, replace(config, input_length_bounds=(32, 64)), training,
        seed, 4, epochs, cap, prefetch_batches=0,
    )


def test_natural_pool_keeps_dataset_composition_and_inclusive_lengths(tiny_config, tokenizer):
    index = make_index({
        ("semantic", "ae", 15): 4,
        ("semantic", "ae", 16): 9,
        ("semantic", "ae", 64): 1,
        ("random", "continuation", 32): 2,
    })
    sampler = sampler_for(index, natural_training(), tiny_config, tokenizer)
    assert sampler.total_steps == 9
    assert sampler.plans[0][0] == {(None, None, None): Fraction(1)}
    assert "actual_cells" not in sampler.epoch_report(1)
    sampler.start_epoch()
    assert set(sampler.order) == set(range(4, 16))
    report = sampler.epoch_report(1)
    assert report["samples"] == 12 and report["quota_unit"] == 4
    assert report["actual_cells"] == {
        "semantic/ae/32": 9, "semantic/ae/64": 1, "random/continuation/32": 2,
    }
    # A missing source/task/length combination does not constrain natural sampling.
    capped = sampler_for(index, natural_training(), tiny_config, tokenizer, cap=11)
    capped.start_epoch()
    assert len(capped.order) == len(set(capped.order)) == 8
    assert set(capped.order) <= set(sampler.order)
    # Uniformly sample the single pool with the existing sampling RNG contract.
    rng = random.Random(17)
    expected = rng.sample(list(range(4, 16)), 8)
    rng.shuffle(expected)
    assert capped.order == expected
    with pytest.raises(ValueError, match="no complete epoch"):
        sampler_for(index, natural_training(), tiny_config, tokenizer, cap=3)


def test_warmup_changes_allowed_tasks_without_forcing_proportions(tiny_config, tokenizer):
    index = make_index({("semantic", "ae", 16): 8, ("random", "continuation", 64): 16})
    training = natural_training()
    training["task_schedule"] = [
        {"epoch": 1, "tasks": ["ae"], "weights": None},
        {"epoch": 2, "tasks": ["ae", "continuation"], "weights": None},
    ]
    sampler = sampler_for(index, training, tiny_config, tokenizer, epochs=2)
    assert [sampler.epoch_report(e)["samples"] for e in (1, 2)] == [8, 24]
    sampler.start_epoch()
    first = sampler.state_dict()
    assert all(index.entries[i][6] == "ae" for i in sampler.order)
    # Reject a checkpoint that substitutes a same-length plan with an excluded task.
    invalid = deepcopy(first)
    invalid["order"][0] = 8
    with pytest.raises(ValueError, match="ineligible"):
        sampler_for(index, training, tiny_config, tokenizer, epochs=2).load_state_dict(invalid)
    restored = sampler_for(index, training, tiny_config, tokenizer, seed=99, epochs=2)
    restored.load_state_dict(first)
    sampler.start_epoch()
    restored.start_epoch()
    assert restored.order == sampler.order
    assert Counter(index.entries[i][6] for i in sampler.order) == {"ae": 8, "continuation": 16}


@pytest.mark.parametrize("dimension", ["source", "task", "length"])
def test_only_constrained_dimensions_determine_quotas(tiny_config, tokenizer, dimension):
    index = make_index({
        ("semantic", "ae", 16): 12,
        ("random", "continuation", 64): 4,
    })
    training = natural_training()
    keys = {"source": ("semantic", "random"), "task": ("ae", "continuation"), "length": ("32", "64")}
    training[f"{dimension}_schedule"] = [{"epoch": 1, "weights": dict.fromkeys(keys[dimension], 1)}]
    sampler = sampler_for(index, training, tiny_config, tokenizer, cap=16)
    assert sampler.epoch_report(1)["samples"] == 8
    assert len(sampler.plans[0][1]) == 2  # No empty Cartesian cells for the other dimensions.
    sampler.start_epoch()
    assert Counter(index.entries[i][0] for i in sampler.order) == {"semantic": 4, "random": 4}


def test_null_transitions_are_discrete_and_numeric_lengths_still_interpolate():
    schedule = [
        {"epoch": 1, "weights": {"32": 3, "64": 1}},
        {"epoch": 3, "weights": {"32": 1, "64": 1}},
        {"epoch": 5, "weights": None},
        {"epoch": 7, "weights": {"32": 0, "64": 1}},
    ]
    assert distribution_at(schedule, 2, True) == {"32": Fraction(5, 8), "64": Fraction(3, 8)}
    assert distribution_at(schedule, 4, True) == {"32": Fraction(1, 2), "64": Fraction(1, 2)}
    assert distribution_at(schedule, 5, True) is None
    assert distribution_at(schedule, 6, True) is None
    assert distribution_at(schedule, 7, True) == {"32": 0, "64": 1}
    assert distribution_at(None, 1) is None


def test_switching_constraints_rebuilds_epoch_pools_and_restores(tiny_config, tokenizer):
    index = make_index({("semantic", "ae", 16): 12, ("random", "continuation", 64): 4})
    training = natural_training()
    training["source_schedule"] = [
        {"epoch": 1, "weights": {"semantic": 1, "random": 1}},
        {"epoch": 2, "weights": None},
        {"epoch": 3, "weights": {"semantic": 1, "random": 0}},
    ]
    sampler = sampler_for(index, training, tiny_config, tokenizer)
    assert [sampler.epoch_report(e)["samples"] for e in (1, 2, 3)] == [8, 16, 12]
    sampler.start_epoch()
    sampler.visits = 8
    sampler.start_epoch()
    sampler.cursor = 4
    sampler.visits += 4
    state = deepcopy(sampler.state_dict())
    restored = sampler_for(index, training, tiny_config, tokenizer, seed=999)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.epoch_report(2) == sampler.epoch_report(2)
    # Length filtering is enforced even when no distribution constraints remain.
    invalid = deepcopy(training)
    invalid["input_tokens"]["min"] = 32
    with pytest.raises(ValueError, match="no complete epoch"):
        sampler_for(index, invalid, tiny_config, tokenizer)


def test_legacy_three_distribution_sampling_order_is_unchanged(tiny_config, tokenizer):
    index = make_index({(s, t, b): 24 for s in ("semantic", "random") for t in ("ae", "continuation") for b in (32, 64)})
    training = {
        "source_schedule": [{"epoch": 1, "weights": {"semantic": 1, "random": 1}}],
        "task_schedule": [{"epoch": 1, "weights": {"ae": 1, "continuation": 1}}],
        "length_schedule": [{"epoch": 1, "weights": {"32": 1, "64": 1}}],
    }
    sampler = sampler_for(index, training, tiny_config, tokenizer, cap=32)
    pools = {}
    for i, entry in enumerate(index.entries):
        pools.setdefault((entry[0], entry[6], entry[7]), []).append(i)
    rng = random.Random(17)
    for epoch in (1, 2, 3):
        probabilities = epoch_distribution(training, epoch)
        quotas, _, _ = maximal_quotas({k: len(v) for k, v in pools.items()}, probabilities, 4, 32)
        expected = [i for key, n in quotas.items() for i in rng.sample(pools[key], n)]
        rng.shuffle(expected)
        sampler.start_epoch()
        assert sampler.order == expected


@pytest.mark.parametrize("change", [
    {"input_tokens": {"min": 0, "max": 64}},
    {"input_tokens": {"min": 32, "max": 16}},
    {"input_tokens": {"min": 1, "max": 65}},
    {"task_schedule": [{"epoch": 1, "tasks": [], "weights": None}]},
    {"task_schedule": [{"epoch": 1, "tasks": ["ae", "ae"], "weights": None}]},
    {"task_schedule": [{"epoch": 1, "tasks": ["qa"], "weights": None}]},
    {"task_schedule": [{"epoch": 1, "tasks": ["ae"], "weights": {"ae": 1, "continuation": 1}}]},
    {"source_schedule": [{"epoch": 1, "weights": {"unknown": 1}}]},
    {"source_schedule": [{"epoch": 2, "weights": None}]},
    {"length_schedule": []},
    {"task_schedule": [{"epoch": 1, "weights": {"ae": 0, "continuation": 0}}]},
])
def test_invalid_optional_schedule_contract(change):
    training = natural_training() | change
    with pytest.raises(ValueError):
        validate_curriculum(training, ("semantic", "random"), (32, 64), 64)


def test_explicit_task_subset_weights_and_natural_source_metadata():
    training = natural_training()
    training["task_schedule"] = [{"epoch": 1, "tasks": ["ae"], "weights": {"ae": 1}}]
    validate_curriculum(training, ("semantic", "random"), (32, 64), 64)
    assert tasks_at(training, 10) == ["ae"]
    assert epoch_distribution(training, 1) == {(None, "ae", None): 1}
    report = {"source_preparations": {}, "selection": {"sources": {"semantic": "s", "random": "r"}, "training": training}}
    assert selection_metadata(report, "train")["sources"] == ["random", "semantic"]
    training["source_schedule"] = [
        {"epoch": 1, "weights": {"semantic": 1, "random": 0}},
        {"epoch": 2, "weights": None},
    ]
    assert selection_metadata(report, "train")["sources"] == ["random", "semantic"]
