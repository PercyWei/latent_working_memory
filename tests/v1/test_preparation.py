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
from latent_working_memory.data_preparation.sources import load_sources
from latent_working_memory.data_preparation.quality import document_rejection_reason
from latent_working_memory.v1.prepared_data import pretraining_index
from latent_working_memory.data_preparation.text_samples import TextSample
from latent_working_memory.v1.sampling import PretrainSampler


def test_independent_variants_balance_tasks_and_intervals(
    parquet_source,
    monkeypatch,
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
):
    def repeated_audit(*args):
        pytest.fail("comparison must reuse completed audits")

    monkeypatch.setattr(
        "latent_working_memory.data_preparation.audit.audit_preparation", repeated_audit
    )
    result = prepare_fineweb(
        parquet_source(preparation_records),
        tokenizer,
        tiny_config,
        tmp_path / "data",
        preparation_recipe,
    )
    root = tmp_path / "data"
    assert result["semantic"]["source_pool_id"] == result["random"]["source_pool_id"]
    assert result["random"]["reference_preparation_id"] == result["semantic"]["preparation_id"]
    for variant in ("semantic", "random"):
        assert result[variant]["input_histogram"] == preparation_recipe.balanced_histogram()
        assert "audit" not in result[variant]
        assert {p.name for p in (root / variant).iterdir()} == {
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "preparation.json",
        }
    different_exact_lengths = False
    for split, quota in zip(
        ("train", "dev", "test"), preparation_recipe.samples_per_task, strict=True
    ):
        a, b = [
            [
                TextSample(**json.loads(line))
                for line in (root / v / f"{split}.jsonl").read_text().splitlines()
            ]
            for v in ("semantic", "random")
        ]
        for rows in (a, b):
            assert Counter(e.task for e in rows) == {"ae": quota, "continuation": quota}
            assert all("pair_id" not in e.to_record() for e in rows)
            assert all("granularity" not in e.to_record() for e in rows)
        different_exact_lengths |= Counter(e.reference_input_tokens for e in a) != Counter(
            e.reference_input_tokens for e in b
        )
    assert different_exact_lengths
    assert all(
        compare_preparations(root, json.loads((root / "random/preparation.json").read_text()))[
            "checks"
        ].values()
    )
    for variant in ("semantic", "random"):
        index = pretraining_index(root / variant / "train.jsonl", tokenizer, tiny_config)
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


def test_source_pool_budget_and_near_duplicates_are_shared(
    parquet_source,
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
    prepare_sources(parquet_source(records + records), tiny_config, tmp_path / "pool", recipe)
    rows = load_sources(tmp_path / "pool/source-pool.json")
    assert not (tmp_path / "pool/sources.jsonl").exists()
    assert len(rows) == 3
    assert len({r["split"] for r in rows}) == 1
    assert [r["status"] for r in rows].count("eligible") == 1


def test_failed_random_stage_preserves_completed_semantic(
    parquet_source,
    monkeypatch,
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
):
    root = tmp_path / "data"
    prepare_sources(parquet_source(preparation_records), tiny_config, root, preparation_recipe)
    prepare_variant(root, "semantic", tokenizer, tiny_config, preparation_recipe)
    before = (root / "semantic/preparation.json").read_bytes()
    monkeypatch.setattr(
        "latent_working_memory.data_preparation.pipeline.RandomSpans.available",
        lambda *args: False,
    )
    with pytest.raises(ValueError, match="sample quotas not reached"):
        prepare_variant(root, "random", tokenizer, tiny_config, preparation_recipe)
    assert (root / "semantic/preparation.json").read_bytes() == before
    assert not (root / "random/preparation.json").exists()
    assert not (root / "comparison.json").exists()


def test_shared_registry_detects_cross_variant_split_leakage(
    parquet_source,
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
):
    root = tmp_path / "data"
    prepare_fineweb(parquet_source(preparation_records), tokenizer, tiny_config, root, preparation_recipe)
    path = root / "random/train.jsonl"
    lines = path.read_text().splitlines()
    with (root / "random/test.jsonl").open("a") as out:
        out.write(lines[0] + "\n")
    path.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(ValueError, match="split mismatch"):
        audit_preparation(root / "random", tokenizer, tiny_config, preparation_recipe, root)


def test_source_text_audit_detects_offset_corruption(
    parquet_source,
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
):
    root = tmp_path / "data"
    prepare_fineweb(parquet_source(preparation_records), tokenizer, tiny_config, root, preparation_recipe)
    path = root / "random/train.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["x_char_span"][0] += 1
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    with pytest.raises(
        ValueError, match="source text|character span|reference tokens|input text key"
    ):
        audit_preparation(root / "random", tokenizer, tiny_config, preparation_recipe, root)


def test_only_explicit_properties_filter_sources(
    tiny_config, tokenizer, preparation_records, semantic_examples
):
    record = dict(preparation_records[0], text="Read more. Privacy policy. Sign up for updates.")
    assert document_rejection_reason(record, 1) is None
    assert semantic_examples(record, tokenizer, tiny_config)
    assert document_rejection_reason(dict(record, text="  "), 1) == "too_short"


def test_independent_inspection_is_repeatable_and_does_not_mutate_data(
    parquet_source,
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
):
    root = tmp_path / "data"
    prepare_fineweb(parquet_source(preparation_records), tokenizer, tiny_config, root, preparation_recipe)
    for variant in ("semantic",):
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
        assert all("source_granularity" not in row for row in rows)
        assert all("granularity" not in row for row in rows)
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
    parquet_source,
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    monkeypatch,
):
    root = tmp_path / "data"
    prepare_sources(parquet_source(preparation_records), tiny_config, root, preparation_recipe)
    prepare_variant(root, "semantic", tokenizer, tiny_config, preparation_recipe)

    def failed_comparison(*args):
        raise ValueError("test comparison failure")

    monkeypatch.setattr(
        "latent_working_memory.data_preparation.pipeline.compare_preparations", failed_comparison
    )
    with pytest.raises(ValueError, match="test comparison failure"):
        prepare_variant(root, "random", tokenizer, tiny_config, preparation_recipe)
    assert (root / "semantic/preparation.json").exists()
    assert not (root / "random/preparation.json").exists()


def test_existing_source_pool_rejects_a_changed_split_definition(
    parquet_source,
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
):
    root = tmp_path / "data"
    prepare_sources(parquet_source(preparation_records), tiny_config, root, preparation_recipe)
    with pytest.raises(ValueError, match="source pool configuration differs"):
        prepare_variant(
            root,
            "semantic",
            tokenizer,
            replace(tiny_config, data_seed=2),
            preparation_recipe,
        )
    assert not (root / "semantic").exists()


@pytest.mark.parametrize("count_type", ["task", "length_interval"])
def test_comparison_rejects_inconsistent_audited_counts(
    parquet_source,
    tmp_path, tiny_config, tokenizer, preparation_records, preparation_recipe, count_type
):
    root = tmp_path / "data"
    result = prepare_fineweb(parquet_source(preparation_records), tokenizer, tiny_config, root, preparation_recipe)
    if count_type == "task":
        result["random"]["statistics"]["train/ae"] += 1
    else:
        result["random"]["input_histogram"]["train"]["ae"][
            str(preparation_recipe.length_bounds[0])
        ] += 1
    with pytest.raises(ValueError, match="quotas"):
        compare_preparations(root, result["random"])
