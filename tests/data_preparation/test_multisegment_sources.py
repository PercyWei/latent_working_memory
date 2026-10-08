from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig
from latent_working_memory.data_preparation.segmentation import SegmentationConfig
from latent_working_memory.data_preparation.fineweb_multisegment import sources
from latent_working_memory.data_preparation.pretrain.config import PreparationConfig
from latent_working_memory.data_preparation.pretrain.dedup import source_key
from latent_working_memory.data_preparation.pretrain.fineweb import document_split


SEED = 20260907
FRACTIONS = (1 / 3, 1 / 3, 1 / 3)


def config(counts, batch_size=2):
    return DataPreparationConfig(
        source_dir="unused",
        source_seed=SEED,
        source_batch_size=batch_size,
        split_counts=counts,
        window=SegmentationConfig(
            capacity=1,
            min_segments=2,
            max_segments=2,
            min_segment_ratio=1,
            max_segment_ratio=1,
            continuation_tokens=1,
            content_reserve_ratio=1,
        ),
    )


def record(name, split="train", text=None):
    for index in range(1000):
        url = f"https://example.org/{name}/{index}"
        if document_split(source_key(url), SEED, FRACTIONS) == split:
            return {"id": name, "url": url, "text": text or f"{name} source words " * 30}
    raise AssertionError("fixture split not found")


def source_stream(monkeypatch, records):
    state = {"calls": 0, "read_ids": [], "closed": False}

    def stream(paths, seed):
        state["calls"] += 1
        assert seed == SEED
        try:
            for index, item in enumerate(records):
                state["read_ids"].append(item["id"])
                yield item, {"source_file": "unused.parquet", "row_group": 0, "row_index": index}
        finally:
            state["closed"] = True

    monkeypatch.setattr(sources, "parquet_records", stream)
    return state


def test_quotas_continue_across_batches_without_restarting_or_reusing_sources(monkeypatch):
    records = [
        record("train-a"),
        record("train-b"),
        record("train-c"),
        record("dev", "dev"),
        record("test", "test"),
        record("train-d"),
        record("unread"),
    ]
    state = source_stream(monkeypatch, records)
    selected, statistics = sources.collect_documents(
        [], config({"train": 1, "dev": 1, "test": 1}), [], PreparationConfig()
    )
    assert {row["record"]["id"] for row in selected} == {"train-a", "dev", "test"}
    assert statistics["source_batches"] == 3
    assert statistics["scanned_documents"] == 6
    assert statistics["candidate_documents"] == 6
    assert statistics["available_by_split"] == {"train": 4, "dev": 1, "test": 1}
    assert statistics["selected_by_split"] == {"train": 1, "dev": 1, "test": 1}
    assert state == {
        "calls": 1,
        "read_ids": [row["id"] for row in records[:6]],
        "closed": True,
    }


def test_duplicate_ids_and_urls_across_batches_never_fill_additional_slots(monkeypatch):
    first = record("a")
    same_url = dict(record("b"), url=first["url"])
    records = [first, dict(first), same_url, record("c")]
    state = source_stream(monkeypatch, records)
    selected, statistics = sources.collect_documents(
        [], config({"train": 2, "dev": 0, "test": 0}), [], PreparationConfig()
    )
    assert [row["record"]["id"] for row in selected] == ["a", "c"]
    assert statistics["duplicate_id"] == 1
    assert statistics["duplicate"] == 1
    assert statistics["candidate_documents"] == 3
    assert state["calls"] == 1 and state["read_ids"] == [row["id"] for row in records]


def test_late_bridge_reclusters_prior_choices_and_reassigns_final_split(monkeypatch):
    words = [f"word{i:03d}" for i in range(120)]
    bridge = words.copy()
    bridge[20] = "changed"
    distant = bridge.copy()
    distant[70] = "altered"
    records = [
        record("z-a", "train", " ".join(words)),
        record("y-b", "dev", " ".join(distant)),
        record("a-bridge", "test", " ".join(bridge)),
        record("d-dev", "dev"),
        record("e-train", "train"),
        record("f-dev", "dev"),
    ]
    state = source_stream(monkeypatch, records)
    selected, statistics = sources.collect_documents(
        [], config({"train": 1, "dev": 1, "test": 1}), [], PreparationConfig()
    )
    by_id = {row["record"]["id"]: row for row in selected}
    assert set(by_id) == {"z-a", "d-dev", "e-train"}
    assert by_id["z-a"]["cluster"] == source_key(records[2]["url"])
    assert by_id["z-a"]["split"] == "test"
    assert statistics["duplicate"] == 2
    assert statistics["source_batches"] == 3
    assert state["read_ids"] == [row["id"] for row in records]


def test_previously_excluded_candidates_remain_available_as_later_bridges(monkeypatch):
    words = [f"word{i:03d}" for i in range(120)]
    near = words.copy()
    near[20] = "changed"
    distant = near.copy()
    distant[70] = "altered"
    previous_record = record("previous", text=" ".join(words))
    records = [
        record("near", text=" ".join(near)),
        record("distant", text=" ".join(distant)),
        record("unused"),
    ]
    state = source_stream(monkeypatch, records)
    previous = [{"document_id": "previous"}]
    references = []

    def load_previous(items):
        references.append(items)
        yield previous_record

    monkeypatch.setattr(sources, "referenced_records", load_previous)
    selected, statistics = sources.collect_documents(
        [], config({"train": 1, "dev": 0, "test": 0}, batch_size=1), previous, PreparationConfig()
    )
    assert [row["record"]["id"] for row in selected] == ["unused"]
    assert references == [previous]
    assert statistics["previously_used"] == 2
    assert statistics["source_batches"] == 3
    assert state["closed"]


def test_quota_overflow_candidate_remains_available_after_cluster_split_changes(monkeypatch):
    words = [f"word{i:03d}" for i in range(120)]
    near = words.copy()
    near[20] = "changed"
    records = [
        record("selected-train"),
        record("z-overflow", "train", " ".join(words)),
        record("a-near", "dev", " ".join(near)),
        record("final-test", "test"),
    ]
    source_stream(monkeypatch, records)
    selected, statistics = sources.collect_documents(
        [], config({"train": 1, "dev": 1, "test": 1}), [], PreparationConfig()
    )
    by_id = {row["record"]["id"]: row for row in selected}
    assert set(by_id) == {"selected-train", "z-overflow", "final-test"}
    assert by_id["z-overflow"]["split"] == "dev"
    assert by_id["z-overflow"]["cluster"] == source_key(records[2]["url"])
    assert statistics["candidate_documents"] == 4


def test_exhaustion_reports_each_unfilled_split_and_closes_stream(monkeypatch):
    state = source_stream(monkeypatch, [record("only-train")])
    with pytest.raises(ValueError, match="missing trajectories: train=1, dev=1, test=2"):
        sources.collect_documents(
            [], config({"train": 2, "dev": 1, "test": 2}), [], PreparationConfig()
        )
    assert state["calls"] == 1 and state["closed"]


def test_base_quality_and_minimum_window_filter_before_candidate_clustering(monkeypatch):
    cfg = replace(
        config({"train": 1, "dev": 0, "test": 0}),
        window=replace(config({"train": 1, "dev": 0, "test": 0}).window, capacity=10),
    )
    records = [record("basic", text="x"), record("short", text="longer than basic"), record("good")]
    source_stream(monkeypatch, records)
    selected, statistics = sources.collect_documents(
        [], cfg, [], PreparationConfig(min_document_chars=4)
    )
    assert [row["record"]["id"] for row in selected] == ["good"]
    assert statistics["basic_rejected"] == statistics["length_rejected"] == 1
    assert statistics["candidate_documents"] == statistics["selected"] == 1


def test_real_parquet_stream_locations_match_selected_records(tmp_path):
    path = tmp_path / "source.parquet"
    records = [record(f"document-{i}") for i in range(7)]
    pq.write_table(pa.Table.from_pylist(records), path, row_group_size=2)
    cfg = config({"train": 7, "dev": 0, "test": 0}, batch_size=2)
    selected, statistics = sources.collect_documents([path], cfg, [], PreparationConfig())
    assert statistics["scanned_documents"] == 7
    assert statistics["source_batches"] == 4
    assert len({row["record"]["id"] for row in selected}) == 7
    with pq.ParquetFile(path) as parquet:
        for row in selected:
            location = row["location"]
            assert Path(location["source_file"]) == path
            original = parquet.read_row_group(location["row_group"]).slice(location["row_index"], 1)
            assert original.to_pylist()[0] == row["record"]


def test_source_length_filter_sums_independently_rounded_windows(monkeypatch):
    cfg = replace(
        config({"train": 1, "dev": 0, "test": 0}),
        window=replace(config({"train": 1, "dev": 0, "test": 0}).window, content_reserve_ratio=1.1),
    )
    # 两段正文与续文分别 ceil(4 × 1 × 1.1) = 5，而整体 ceil 只得到 14。
    assert cfg.window.minimum_window_chars == 15
    source_stream(monkeypatch, [record("short", text="x" * 14), record("fits", text="y" * 15)])
    selected, statistics = sources.collect_documents(
        [], cfg, [], PreparationConfig(min_document_chars=4)
    )
    assert [row["record"]["id"] for row in selected] == ["fits"]
    assert statistics["length_rejected"] == 1
