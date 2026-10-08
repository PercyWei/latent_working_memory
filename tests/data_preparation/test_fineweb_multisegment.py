"""多段构造、参数化采样与跨数据集来源排除。"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import runpy
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig
from latent_working_memory.data_preparation.fineweb_multisegment import (
    prepare as preparation_module,
)
from latent_working_memory.data_preparation.fineweb_multisegment.prepare import (
    Document,
    build_samples,
    load_previous_sources,
    prepare_dataset,
    segment_lengths,
)
from latent_working_memory.data_preparation.fineweb_multisegment.records import MultisegmentSample
from latent_working_memory.data_preparation.pretrain.dedup import source_key
from latent_working_memory.data_preparation.pretrain.fineweb import document_split


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs/data_preparation/fineweb-multisegment"


@pytest.mark.parametrize("capacity,segment_bounds", [(64, (64, 192)), (512, (512, 1536))])
def test_official_config_cli_builds_self_contained_samples_with_unchanged_continuation(
    tmp_path, monkeypatch, capacity, segment_bounds
):
    official_path = CONFIG_DIR / f"fineweb-multisegment-k{capacity}-seg1to3x_train32k.json"
    original = official_path.read_bytes()
    config = DataPreparationConfig(**json.loads(original))
    assert (config.min_segment_tokens, config.max_segment_tokens) == segment_bounds
    assert config.continuation_tokens == 512
    assert config.dataset_name("smoke") == (
        f"fineweb-multisegment-k{capacity}-seg1to3x_train32k_smoke"
    )

    counts = {"train": 3, "dev": 1, "test": 1}
    records = []
    for split, count in counts.items():
        for index in range(count):
            identity = f"{split}-{index}"
            for suffix in range(1000):
                url = f"https://example.org/{identity}/{suffix}"
                if (
                    document_split(source_key(url), config.source_seed, config.split_fractions)
                    == split
                ):
                    break
            else:
                raise AssertionError("fixture split not found")
            records.append(
                {
                    "id": identity,
                    "url": url,
                    "text": (" ".join(f"{identity}word{word}" for word in range(8)) + " ") * 600,
                }
            )
    source = tmp_path / "fineweb.parquet"
    pq.write_table(pa.Table.from_pylist(records), source, row_group_size=2)
    small = replace(config, source_glob=str(source), source_batch_size=2, counts=counts)
    temporary_config = tmp_path / "configs" / "fineweb-multisegment" / official_path.name
    temporary_config.parent.mkdir(parents=True)
    temporary_config.write_text(json.dumps(asdict(small)))
    output_root = tmp_path / "datasets"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fineweb_multisegment",
            "--config",
            str(temporary_config),
            "--output-root",
            str(output_root),
            "--run-id",
            "smoke",
        ],
    )
    runpy.run_module(
        "latent_working_memory.data_preparation.fineweb_multisegment", run_name="__main__"
    )

    root = output_root / f"fineweb-multisegment-k{capacity}-seg1to3x_train3_smoke"
    metadata = json.loads((root / "preparation.json").read_text())
    assert metadata["config"] == json.loads(temporary_config.read_text())
    assert metadata["config"]["continuation_tokens"] == 512
    assert metadata["referenced_documents"] == sum(counts.values())
    original_text = {record["id"]: record["text"] for record in records}
    samples = read_samples(root)
    for split, rows in samples.items():
        assert len(rows) == counts[split]
        totals = []
        for sample in rows:
            sample.validate_plan(small)
            ends = sample.write_token_ends
            lengths = [b - a for a, b in zip((0,) + ends[:-1], ends, strict=True)]
            assert 3 <= len(lengths) <= 5
            assert all(segment_bounds[0] <= length <= segment_bounds[1] for length in lengths)
            total = sum(lengths)
            assert ends[-1] == total
            assert len(sample.text) == 6 * (total + 512)
            start, end = sample.source["char_span"]
            assert sample.text == original_text[sample.document_id][start:end]
            totals.append(total)
        assert metadata["statistics"][split]["planned_content_tokens"] == sum(totals)
    source.unlink()
    assert read_samples(root) == samples
    assert official_path.read_bytes() == original


def test_config_ratios_and_name_follow_actual_parameters():
    config = DataPreparationConfig("unused")
    assert config.dataset_name("20261008") == (
        "fineweb-multisegment-k512-seg1to3x_train32k_20261008"
    )
    changed = replace(
        config,
        capacity=511,
        min_segment_ratio=0.5,
        max_segment_ratio=1.5,
        counts={"train": 100000, "dev": 1, "test": 1},
    )
    assert (changed.min_segment_tokens, changed.max_segment_tokens) == (256, 766)
    assert "k511-seg0.5to1.5x" in changed.dataset_name("20261008")
    assert "train100k" in changed.dataset_name("20261008")


@pytest.mark.parametrize(
    "update",
    [
        {"min_segment_ratio": 3, "max_segment_ratio": 1},
        {"min_segment_ratio": float("nan")},
        {"capacity": 1, "min_segment_ratio": 0.1, "max_segment_ratio": 0.2},
        {"min_segments": 1},
        {"max_segments": 2},
        {"source_batch_size": 0},
        {"content_reserve_ratio": 0.5},
        {"counts": {"train": 0, "dev": 1, "test": 1}},
    ],
)
def test_invalid_construction_contracts(update):
    with pytest.raises(ValueError):
        DataPreparationConfig("unused", **update)


@pytest.mark.parametrize("run_id", ["", "../escape", "bad/name", "bad name", "-leading", "bad\n"])
def test_invalid_run_id_does_not_create_output(tmp_path, run_id):
    config = DataPreparationConfig("missing.parquet")
    output_root = tmp_path / "outputs"
    with pytest.raises(ValueError, match="run_id"):
        prepare_dataset(config, output_root, run_id=run_id)
    assert not output_root.exists()


@pytest.fixture
def fixed_creation_time(monkeypatch):
    # UTC 仍是前一天，用于确认默认名称采用 Asia/Shanghai 日期。
    instant = datetime(2026, 10, 7, 16, 5, 6, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)

    monkeypatch.setattr(preparation_module, "datetime", FixedDateTime)


@pytest.mark.parametrize("count,minimum,maximum", [(3, 2, 7), (5, 4, 4), (2, 1, 11)])
def test_segment_sampling_respects_available_budget_without_filling_it(count, minimum, maximum):
    for available in range(count * minimum, count * maximum + 5):
        parts = segment_lengths(count, minimum, maximum, available, random.Random(available))
        assert len(parts) == count and sum(parts) <= available
        assert all(minimum <= p <= maximum for p in parts)
    if minimum != maximum:
        totals = {
            sum(segment_lengths(count, minimum, maximum, count * maximum, random.Random(seed)))
            for seed in range(20)
        }
        assert len(totals) > 1
        assert min(totals) < count * maximum


def test_one_trajectory_per_source_with_total_derived_from_segments():
    config = DataPreparationConfig("unused", counts={"train": 100, "dev": 0, "test": 0})
    documents = [
        Document(
            f"doc{index}",
            "0123456789" * 5000,
            "train",
            f"cluster{index}",
            {"file": "unused.parquet", "row_group": 0, "row_index": index},
        )
        for index in range(100)
    ]
    samples = list(build_samples(documents, config, "train"))
    assert samples == list(build_samples(documents, config, "train"))
    assert len({s.document_id for s in samples}) == len(samples) == 100
    assert {len(sample.write_token_ends) for sample in samples} == {3, 4, 5}
    assert any(sample.write_token_ends[-1] > 5120 for sample in samples)
    assert max(sample.write_token_ends[-1] for sample in samples) <= 7680
    for sample in samples:
        sample.validate_plan(config)
        if len(sample.write_token_ends) == 3:
            assert sample.write_token_ends[-1] <= 4608
        start, end = sample.source["char_span"]
        assert sample.text == documents[0].text[start:end]
        assert len(sample.text) == config.candidate_chars(sample.write_token_ends[-1])
    # 更改其他划分的配额不能影响训练样本。
    other = replace(config, counts={"train": 100, "dev": 8, "test": 0})
    assert samples == list(build_samples(documents, other, "train"))
    assert list(build_samples(documents, config, "test")) == []


def test_exact_character_budget_accepts_decimal_reserve_at_boundary():
    config = DataPreparationConfig(
        "unused",
        capacity=5,
        min_segments=3,
        max_segments=3,
        min_segment_ratio=1,
        max_segment_ratio=1,
        continuation_tokens=1,
        content_reserve_ratio=1.1,
        counts={"train": 1, "dev": 0, "test": 0},
    )
    assert config.candidate_chars(15) == 71
    assert config.available_content_tokens(71) == 15
    assert config.available_content_tokens(70) == 14
    doc = Document(
        "id",
        "x" * 71,
        "train",
        "cluster",
        {"file": "source.parquet", "row_group": 0, "row_index": 0},
    )
    sample = next(build_samples([doc], config, "train"))
    assert sample.write_token_ends == (5, 10, 15)
    assert sample.text == doc.text


def test_short_eligible_source_limits_segment_count_instead_of_being_discarded():
    config = DataPreparationConfig("unused", counts={"train": 1, "dev": 0, "test": 0})
    chars = config.candidate_chars(config.min_segments * config.min_segment_tokens)
    doc = Document(
        "short",
        "x" * chars,
        "train",
        "short",
        {"file": "source.parquet", "row_group": 0, "row_index": 0},
    )
    sample = next(build_samples([doc], config, "train"))
    assert sample.write_token_ends == (512, 1024, 1536)
    assert sample.text == doc.text


def make_config(tmp_path):
    path = tmp_path / "fineweb.parquet"
    records = [
        {
            "id": str(i),
            "url": f"https://example.org/document/{i}",
            "text": f"article{i} " + "red blue sky water " * (8 + i % 4),
        }
        for i in range(100)
    ]
    pq.write_table(pa.Table.from_pylist(records), path, row_group_size=13)
    config = DataPreparationConfig(
        str(path),
        capacity=2,
        min_segment_ratio=1.5,
        max_segment_ratio=2.5,
        continuation_tokens=2,
        source_batch_size=100,
        split_fractions=(0.6, 0.2, 0.2),
        counts={"train": 8, "dev": 3, "test": 3},
    )
    return config, {record["id"]: record for record in records}


def read_samples(root):
    return {
        split: [
            MultisegmentSample(**json.loads(line))
            for line in (root / f"{split}.jsonl").read_text().splitlines()
        ]
        for split in ("train", "dev", "test")
    }


def test_saved_text_is_self_contained_and_metadata_records_only_used_documents(tmp_path):
    config, records = make_config(tmp_path)
    output_root = tmp_path / "datasets"
    root = output_root / config.dataset_name("self-contained")
    metadata = prepare_dataset(config, output_root, run_id="self-contained")
    assert {p.name for p in root.iterdir()} == {
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "preparation.json",
        "README.md",
    }
    samples = read_samples(root)
    used = set()
    split_documents = []
    for split, rows in samples.items():
        assert len(rows) == config.counts[split]
        assert len({row.document_id for row in rows}) == len(rows)
        split_documents.append({row.document_id for row in rows})
        for sample in rows:
            sample.validate_plan(config)
            start, end = sample.source["char_span"]
            assert sample.text == records[sample.document_id]["text"][start:end]
            used.add(sample.document_id)
            assert "capacity" not in asdict(sample)
        assert metadata["statistics"][split]["trajectories"] == len(rows)
    assert not split_documents[0] & split_documents[1]
    assert not split_documents[0] & split_documents[2]
    assert not split_documents[1] & split_documents[2]
    assert {item["document_id"] for item in metadata["used_sources"]} == used
    assert metadata["referenced_documents"] == len(used) == sum(config.counts.values())
    Path(config.source_glob).unlink()
    assert read_samples(root) == samples
    with pytest.raises(FileExistsError):
        prepare_dataset(config, output_root, run_id="self-contained")


def test_previous_datasets_exclude_used_sources_and_their_current_clusters(tmp_path):
    config, _ = make_config(tmp_path)
    output_root = tmp_path / "datasets"
    first_root = output_root / config.dataset_name("first")
    first = prepare_dataset(config, output_root, run_id="first")
    second_root = output_root / config.dataset_name("second")
    second = prepare_dataset(config, output_root, run_id="second", previous_datasets=(first_root,))
    assert first["config"] == second["config"] == asdict(config)
    assert first["previous_datasets"] == []
    assert second["previous_datasets"] == [str(first_root)]
    assert not (
        {s["document_id"] for s in first["used_sources"]}
        & {s["document_id"] for s in second["used_sources"]}
    )
    assert not (
        {s["dedup_cluster"] for s in first["used_sources"]}
        & {s["dedup_cluster"] for s in second["used_sources"]}
    )
    assert second["source_statistics"]["previously_used"] == len(first["used_sources"])
    # 只排除明确列出的第二份数据，不递归把第一份并入 used_sources。
    assert load_previous_sources((str(second_root),)) == second["used_sources"]
    assert {
        s["document_id"] for s in load_previous_sources((str(first_root), str(second_root)))
    } == ({s["document_id"] for s in first["used_sources"] + second["used_sources"]})


def test_missing_used_sources_is_not_silently_treated_as_no_exclusions(tmp_path):
    (tmp_path / "preparation.json").write_text("{}")
    with pytest.raises(KeyError, match="used_sources"):
        load_previous_sources((str(tmp_path),))


def test_no_long_source_does_not_leave_partial_output(tmp_path):
    config, _ = make_config(tmp_path)
    impossible = replace(config, capacity=512)
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="exhausted"):
        prepare_dataset(impossible, root, run_id="no-sources")
    assert not root.exists()


def test_cli_reuses_config_for_default_custom_and_source_exclusion_runs(
    tmp_path, monkeypatch, fixed_creation_time
):
    config, _ = make_config(tmp_path)
    config_path = tmp_path / "construction.json"
    original_config = json.dumps(asdict(config))
    config_path.write_text(original_config)
    output_root = tmp_path / "outputs"
    roots, metadata = {}, {}
    for run_id, previous in (
        (None, ()),
        ("repeat_2.a-1", ()),
        ("exclude-one", ("20261008",)),
        ("exclude-both", ("20261008", "exclude-one")),
    ):
        args = [
            "fineweb_multisegment",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ]
        if run_id is not None:
            args.extend(("--run-id", run_id))
        if previous:
            args.extend(("--previous-datasets", *(str(roots[key]) for key in previous)))
        monkeypatch.setattr(sys, "argv", args)
        runpy.run_module(
            "latent_working_memory.data_preparation.fineweb_multisegment", run_name="__main__"
        )
        identity = "20261008" if run_id is None else run_id
        root = output_root / config.dataset_name(identity)
        roots[identity] = root
        saved = json.loads((root / "preparation.json").read_text())
        metadata[identity] = saved
        assert saved["dataset"] == root.name
        assert saved["run_id"] == identity
        assert saved["created_at"] == "2026-10-08T00:05:06+08:00"
        assert saved["previous_datasets"] == [str(roots[key]) for key in previous]
        assert saved["config"] == json.loads(original_config)
        assert "run_id" not in saved["config"] and "previous_datasets" not in saved["config"]
        assert "创建时间：20261008 00:05:06 UTC+08:00" in (root / "README.md").read_text()

    assert config_path.read_text() == original_config
    # run_id 仅命名产物，不参与来源选择、分段或窗口起点的随机种子。
    assert read_samples(roots["20261008"]) == read_samples(roots["repeat_2.a-1"])
    used = {
        key: {item["document_id"] for item in value["used_sources"]}
        for key, value in metadata.items()
    }
    assert not used["20261008"] & used["exclude-one"]
    assert not (used["20261008"] | used["exclude-one"]) & used["exclude-both"]
    assert {item["document_id"] for item in metadata["exclude-both"]["excluded_sources"]} == (
        used["20261008"] | used["exclude-one"]
    )

    rejected_root = tmp_path / "rejected"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fineweb_multisegment",
            "--config",
            str(config_path),
            "--output-root",
            str(rejected_root),
            "--run-id",
            "../escape",
        ],
    )
    with pytest.raises(ValueError, match="run_id"):
        runpy.run_module(
            "latent_working_memory.data_preparation.fineweb_multisegment", run_name="__main__"
        )
    assert not rejected_root.exists()
