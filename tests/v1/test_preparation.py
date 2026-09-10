from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import pytest

from latent_working_memory.data_preparation.audit import audit_preparation, compare_preparations
from latent_working_memory.data_preparation.dedup import cluster_documents
from latent_working_memory.data_preparation.inspection import (
    sample_inspection,
    summarize_inspection,
)
from latent_working_memory.data_preparation.pipeline import (
    prepare_fineweb,
    prepare_sources,
    prepare_variant,
)
from latent_working_memory.data_preparation.quality import document_rejection_reason
from latent_working_memory.v1.data import EpisodeIndex, read_episodes
from latent_working_memory.v1.sampling import PretrainSampler


def test_independent_variants_balance_post_review_tasks_and_intervals(
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    accepting_scorer,
):
    result = prepare_fineweb(
        preparation_records,
        tokenizer,
        tiny_config,
        tmp_path / "data",
        preparation_recipe,
        accepting_scorer,
    )
    root = tmp_path / "data"
    assert result["semantic"]["source_pool_id"] == result["random"]["source_pool_id"]
    assert result["random"]["reference_preparation_id"] == result["semantic"]["preparation_id"]
    for variant in ("semantic", "random"):
        assert result[variant]["input_histogram"] == preparation_recipe.balanced_histogram()
        for group in result[variant]["audit"]["composition"].values():
            assert sum(c["sample_fraction"] for c in group.values()) == pytest.approx(1)
            assert sum(c["input_token_fraction"] for c in group.values()) == pytest.approx(1)
    different_exact_lengths = False
    for split, quota in zip(
        ("train", "dev", "test"), preparation_recipe.samples_per_task, strict=True
    ):
        a, b = [read_episodes(root / v / f"{split}.jsonl") for v in ("semantic", "random")]
        for rows in (a, b):
            assert Counter(e.reads[0].task for e in rows) == {"ae": quota, "continuation": quota}
            assert all("pair_id" not in e.sources[0].provenance for e in rows)
        different_exact_lengths |= Counter(len(e.input_ids) for e in a) != Counter(
            len(e.input_ids) for e in b
        )
    assert different_exact_lengths
    assert all(
        compare_preparations(
            root, tokenizer, tiny_config, json.loads((root / "random/preparation.json").read_text())
        )["checks"].values()
    )
    for variant in ("semantic", "random"):
        index = EpisodeIndex(root / variant / "train.jsonl")
        for task, weights in (("ae", {"lm_weight": 0}), ("continuation", {"ae_weight": 0})):
            only_task = PretrainSampler(index, tokenizer, replace(tiny_config, **weights))
            assert all(only_task.sample(0).episode.reads[0].task == task for _ in range(20))
        sampler = PretrainSampler(index, tokenizer, tiny_config)
        for i in range(7):
            sampler.sample(i)
        state = sampler.state_dict()
        expected = [sampler.sample(i) for i in range(10)]
        resumed = PretrainSampler(index, tokenizer, tiny_config)
        resumed.load_state_dict(state)
        assert [resumed.sample(i) for i in range(10)] == expected


@pytest.mark.parametrize("decision", ["reject", "error"])
def test_rejections_refill_quotas_independently(
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    accepting_scorer,
    decision,
):
    original = accepting_scorer.score_batch
    visits = Counter()

    def reject_some(samples):
        results = original(samples)
        for sample, result in zip(samples, results, strict=True):
            variant, task = sample["boundary_variant"], sample["task"]
            visits[(variant, task)] += 1
            if visits[(variant, task)] <= 3:
                result.update(decision=decision, reason="Test rejection before quota accounting")
        return results

    accepting_scorer.score_batch = reject_some
    result = prepare_fineweb(
        preparation_records,
        tokenizer,
        tiny_config,
        tmp_path / "data",
        preparation_recipe,
        accepting_scorer,
    )
    for variant in result:
        assert result[variant]["statistics"][f"review/{decision}"] > 0
        assert result[variant]["statistics"]["train/ae"] == preparation_recipe.samples_per_task[0]
        assert (
            result[variant]["statistics"]["train/continuation"]
            == preparation_recipe.samples_per_task[0]
        )


def test_source_pool_budget_and_near_duplicates_are_shared(
    tmp_path,
    tiny_config,
    preparation_records,
    preparation_recipe,
):
    base = [f"word{i}" for i in range(100)]
    records = [
        dict(preparation_records[i], text=" ".join(words))
        for i, words in enumerate(
            [base, ["changed"] + base[1:], ["changed"] + base[1:-1] + ["different"]]
        )
    ]
    recipe = replace(preparation_recipe, max_documents=3, near_duplicate_min_words=64)
    assert len(set(cluster_documents(records, recipe))) == 1
    consumed = []

    def source():
        for row in records + preparation_records:
            consumed.append(row)
            yield row

    prepare_sources(source(), tiny_config, tmp_path / "pool", recipe)
    rows = [json.loads(line) for line in (tmp_path / "pool/sources.jsonl").read_text().splitlines()]
    assert len(consumed) == 3
    assert len({r["split"] for r in rows}) == 1
    assert [r["status"] for r in rows].count("eligible") == 1


def test_failed_random_stage_preserves_completed_semantic(
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    accepting_scorer,
):
    root = tmp_path / "data"
    prepare_sources(preparation_records, tiny_config, root, preparation_recipe)
    prepare_variant(root, "semantic", tokenizer, tiny_config, preparation_recipe, accepting_scorer)
    before = (root / "semantic/preparation.json").read_bytes()
    accepting_scorer.score_batch = lambda samples: [
        {"decision": "uncertain", "reason": "Test uncertainty", "cache_key": "fixture"}
        for _ in samples
    ]
    with pytest.raises(ValueError, match="sample quotas not reached"):
        prepare_variant(
            root, "random", tokenizer, tiny_config, preparation_recipe, accepting_scorer
        )
    assert (root / "semantic/preparation.json").read_bytes() == before
    assert not (root / "random/preparation.json").exists()
    assert not (root / "comparison.json").exists()


def test_shared_registry_detects_cross_variant_split_leakage(
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    accepting_scorer,
):
    root = tmp_path / "data"
    prepare_fineweb(
        preparation_records, tokenizer, tiny_config, root, preparation_recipe, accepting_scorer
    )
    path = root / "random/train.jsonl"
    lines = path.read_text().splitlines()
    with (root / "random/test.jsonl").open("a") as out:
        out.write(lines[0] + "\n")
    path.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(ValueError, match="split mismatch"):
        compare_preparations(
            root, tokenizer, tiny_config, json.loads((root / "random/preparation.json").read_text())
        )


def test_source_text_audit_detects_offset_corruption(
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    accepting_scorer,
):
    root = tmp_path / "data"
    prepare_fineweb(
        preparation_records, tokenizer, tiny_config, root, preparation_recipe, accepting_scorer
    )
    path = root / "random/train.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["sources"][0]["provenance"]["x_char_span"][0] += 1
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    with pytest.raises(
        ValueError, match="source text|character span|reference tokens|input text key"
    ):
        audit_preparation(root / "random", tokenizer, tiny_config, preparation_recipe, root)


def test_only_explicit_properties_filter_before_model(
    tiny_config, tokenizer, preparation_records, semantic_examples
):
    record = dict(preparation_records[0], text="Read more. Privacy policy. Sign up for updates.")
    assert document_rejection_reason(record, 1) is None
    assert semantic_examples(record, tokenizer, tiny_config)
    assert document_rejection_reason(dict(record, text="  "), 1) == "too_short"


def test_independent_inspection_is_repeatable_and_does_not_mutate_data(
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    accepting_scorer,
):
    root = tmp_path / "data"
    prepare_fineweb(
        preparation_records, tokenizer, tiny_config, root, preparation_recipe, accepting_scorer
    )
    for variant in ("semantic", "random"):
        leaf = root / variant
        before = {p.name: p.read_bytes() for p in leaf.iterdir()}
        first, second = tmp_path / f"{variant}-inspect1", tmp_path / f"{variant}-inspect2"
        sample_inspection(leaf, first, examples=4)
        sample_inspection(leaf, second, examples=4)
        assert (first / "random-views.jsonl").read_bytes() == (
            second / "random-views.jsonl"
        ).read_bytes()
        rows = [
            json.loads(line) for line in (first / "random-views.jsonl").read_text().splitlines()
        ]
        for row, decision in zip(rows, ("pass", "fail", "uncertain", None), strict=True):
            row.update(judgment=decision, reviewer="test reviewer", review_reason="test reason")
        (first / "random-views.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary = summarize_inspection(first)["panels"]["random-views"]["all"]
        assert summary["failure_rate"] is None and summary["sample_failure_fraction_bounds"] == [
            0.25,
            0.75,
        ]
        assert before == {p.name: p.read_bytes() for p in leaf.iterdir()}


def test_failed_cross_variant_comparison_has_no_random_completion_record(
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    accepting_scorer,
    monkeypatch,
):
    root = tmp_path / "data"
    prepare_sources(preparation_records, tiny_config, root, preparation_recipe)
    prepare_variant(root, "semantic", tokenizer, tiny_config, preparation_recipe, accepting_scorer)

    def failed_comparison(*args):
        raise ValueError("test comparison failure")

    monkeypatch.setattr(
        "latent_working_memory.data_preparation.pipeline.compare_preparations", failed_comparison
    )
    with pytest.raises(ValueError, match="test comparison failure"):
        prepare_variant(
            root, "random", tokenizer, tiny_config, preparation_recipe, accepting_scorer
        )
    assert (root / "semantic/preparation.json").exists()
    assert not (root / "random/preparation.json").exists()


def test_existing_source_pool_rejects_a_changed_split_definition(
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    accepting_scorer,
):
    root = tmp_path / "data"
    prepare_sources(preparation_records, tiny_config, root, preparation_recipe)
    with pytest.raises(ValueError, match="source pool configuration differs"):
        prepare_variant(
            root,
            "semantic",
            tokenizer,
            replace(tiny_config, data_seed=2),
            preparation_recipe,
            accepting_scorer,
        )
    assert not (root / "semantic").exists()
