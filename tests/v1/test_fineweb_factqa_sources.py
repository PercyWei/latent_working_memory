import copy
import json
import shutil
from contextlib import closing
from pathlib import Path
from threading import Event

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latent_working_memory.data_preparation.fineweb_factqa import pipeline, sources
from latent_working_memory.data_preparation.fineweb_factqa.storage import load_json
from latent_working_memory.data_preparation.fineweb_source import write_used_sources
from latent_working_memory.data_preparation.pretrain.dedup import source_key
from latent_working_memory.data_preparation.pretrain.fineweb import document_split
from latent_working_memory.data_preparation.segmentation import SegmentationConfig, sample_windows


SEED = 20260907
FRACTIONS = (0.8, 0.1, 0.1)
SPLITS = ("train", "dev", "test")


def _record(name, split="train", text=None):
    for i in range(1000):
        url = f"https://example.org/{name}/{i}"
        if document_split(source_key(url), SEED, FRACTIONS) == split:
            return {
                "id": name,
                "url": url,
                "text": text if text is not None else ((name + " document. ") * 30)[:120],
            }
    raise AssertionError("fixture split not found")


def _pool_config(tmp_path, records, monkeypatch, source_batch_size=None):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "000_00000.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    activity = {"opens": 0, "reads": [], "closes": 0}

    def stream(files, seed):
        assert files == [path] and seed == SEED
        activity["opens"] += 1
        try:
            for i, record in enumerate(records):
                activity["reads"].append(i)
                yield record, {"source_file": str(path), "row_group": 0, "row_index": i}
        finally:
            activity["closes"] += 1

    monkeypatch.setattr(sources, "parquet_records", stream)
    config = {
        "source_dir": str(raw),
        "source_batch_size": source_batch_size or len(records),
        "source_seed": SEED,
        "selection_seed": 31,
        "split_counts": {"train": 8, "dev": 1, "test": 1},
        "window": {
            "min_segments": 6,
            "max_segments": 10,
            "capacity": 4,
            "min_segment_ratio": 1,
            "max_segment_ratio": 1.5,
        },
        "batch_split_counts": {"train": 1, "dev": 1, "test": 1},
        "dataset_dir": str(tmp_path / "dataset"),
        "previous_datasets": [],
    }
    return config, activity


def _selection(config, index):
    return sources.prepare_selection(
        {"source_pool_dir": config["dataset_dir"], "batch_index": index}
    )


def _used(config):
    return [
        json.loads(line)
        for line in (Path(config["dataset_dir"]) / "used-sources.jsonl").read_text().splitlines()
    ]


def _reference(config, record, index):
    return {
        "document_id": record["id"],
        "dedup_cluster": source_key(record["url"]),
        "source": {
            "file": str(Path(config["source_dir"]) / "000_00000.parquet"),
            "row_group": 0,
            "row_index": index,
        },
    }


def test_expanding_pool_consumes_one_stream_and_preserves_frozen_splits_and_windows(
    tmp_path, monkeypatch
):
    records = [_record(f"{split}-{i}", split) for i in range(3) for split in SPLITS]
    config, activity = _pool_config(tmp_path, records, monkeypatch, source_batch_size=3)
    pool = sources.prepare_pool(config)
    assert not activity["reads"] and pool["documents"] == [] and _used(config) == []
    identity = pool["pool_id"]
    selections = []
    with closing(sources.source_records(pool)) as stream:
        for i in range(3):
            frozen = copy.deepcopy(pool["documents"])
            entry = sources.allocate_batch(config, pool, stream, config["batch_split_counts"])
            assert entry["batch_index"] == i
            assert pool["documents"][: len(frozen)] == frozen
            selected = _selection(config, i)
            selections.append(selected)
            assert selected["source_pool_id"] == identity
            assert selected["statistics"]["selected_by_split"] == dict.fromkeys(SPLITS, 1)
    assert activity == {"opens": 1, "reads": list(range(9)), "closes": 1}
    assert pool["statistics"]["source_batches"] == 3
    assert pool["statistics"]["scanned_documents"] == 9
    assert pool["next_offsets"] == dict.fromkeys(SPLITS, 3)
    ids = [d["document_id"] for selection in selections for d in selection["documents"]]
    assert len(set(ids)) == 9
    assert {row["document_id"] for row in _used(config)} == set(ids)
    for selection in selections:
        for document in selection["documents"]:
            frozen = pool["documents"][document["pool_index"]]
            assert {k: v for k, v in document.items() if k != "text"} == frozen
            raw = records[document["source"]["row_index"]]
            assert document["text"] == raw["text"][slice(*document["window_char_span"])]
    snapshot = (Path(config["dataset_dir"]) / "source-pool.json").read_bytes()
    assert sources.prepare_pool(config) == pool
    assert (Path(config["dataset_dir"]) / "source-pool.json").read_bytes() == snapshot
    changed = copy.deepcopy(config)
    changed["selection_seed"] += 1
    with pytest.raises(ValueError, match="configuration changed"):
        sources.prepare_pool(changed)


def test_partial_batches_consume_candidates_before_reading_another_source_batch(
    tmp_path, monkeypatch
):
    records = [_record("a"), _record("b"), _record("dev", "dev"), _record("test", "test")]
    config, activity = _pool_config(tmp_path, records, monkeypatch, source_batch_size=2)
    pool = sources.prepare_pool(config)
    with closing(sources.source_records(pool)) as stream:
        for index in (0, 1):
            sources.allocate_batch(config, pool, stream, config["batch_split_counts"])
            assert _selection(config, index)["statistics"]["selected_by_split"] == {
                "train": 1,
                "dev": 0,
                "test": 0,
            }
            assert activity["reads"] == [0, 1]
        sources.allocate_batch(config, pool, stream, {"train": 0, "dev": 1, "test": 1})
        assert _selection(config, 2)["statistics"]["selected_by_split"] == {
            "train": 0,
            "dev": 1,
            "test": 1,
        }
        with pytest.raises(ValueError, match="exhausted; missing candidates: train=1"):
            sources.allocate_batch(config, pool, stream, {"train": 1, "dev": 0, "test": 0})
    assert pool["exhausted"]
    assert activity["reads"] == [0, 1, 2, 3]


def test_one_source_supplies_disjoint_batches_and_is_referenced_only_once(tmp_path, monkeypatch):
    records = [
        _record("long", text="unique-long article " * 40),
        dict(_record("fresh", text="fresh content " * 30), id="long"),
    ]
    config, activity = _pool_config(tmp_path, records, monkeypatch, source_batch_size=1)
    config["window"].update(
        capacity=16, min_segment_ratio=1, max_segment_ratio=1, min_segments=2, max_segments=2
    )
    references, read_references = [], sources.referenced_records

    def capture(values):
        values = list(values)
        references.append([row["document_id"] for row in values])
        yield from read_references(values)

    monkeypatch.setattr(sources, "referenced_records", capture)
    pool = sources.prepare_pool(config)
    selected = []
    with closing(sources.source_records(pool)) as stream:
        source_trajectories = len(records[0]["text"]) // 128
        for index in range(source_trajectories):
            sources.allocate_batch(config, pool, stream, {"train": 1, "dev": 0, "test": 0})
            selected.extend(_selection(config, index)["documents"])
            assert activity["reads"] == [0]
            assert len(_used(config)) == 1
        with pytest.raises(ValueError, match="exhausted"):
            sources.allocate_batch(config, pool, stream, {"train": 1, "dev": 0, "test": 0})
    assert [document["window_char_span"] for document in selected] == [
        [index * 128, (index + 1) * 128] for index in range(source_trajectories)
    ]
    assert len({document["trajectory_id"] for document in selected}) == source_trajectories
    assert {document["document_id"] for document in selected} == {"long"}
    assert pool["statistics"]["frozen_trajectories"] == source_trajectories
    assert pool["statistics"]["frozen_source_documents"] == 1
    assert references.count(["long"]) == 1


def test_ledger_records_assigned_sources_before_annotation_not_unused_or_rejected(
    tmp_path, monkeypatch
):
    records = [
        _record("first"),
        _record("unused"),
        _record("short", text="tiny"),
        _record("length-short", text="z" * 80),
    ]
    config, _ = _pool_config(tmp_path, records, monkeypatch)
    pool = sources.prepare_pool(config)
    with closing(sources.source_records(pool)) as stream:
        sources.allocate_batch(config, pool, stream, {"train": 1, "dev": 0, "test": 0})
    assert len(pool["documents"]) == 2
    assert pool["statistics"]["basic_rejected_documents"] == 1
    assert pool["statistics"]["length_failed"] == 1
    assigned = _selection(config, 0)["documents"][0]["document_id"]
    assert [row["document_id"] for row in _used(config)] == [assigned]
    # No annotation result exists: even a later annotation failure must retain this assignment.
    assert not (Path(config["dataset_dir"]) / "preparation.json").exists()
    sources.prepare_pool(config)
    assert [row["document_id"] for row in _used(config)] == [assigned]


def test_late_bridge_and_same_id_do_not_change_old_splits_or_add_duplicate_sources(
    tmp_path, monkeypatch
):
    a = _record("a", "train", text=" ".join(f"first{i}" for i in range(120)))
    b = _record("b", "test", text=" ".join(f"second{i}" for i in range(120)))
    records = [
        a,
        b,
        _record("dev", "dev"),
        dict(_record("bridge"), url=a["url"], text=b["text"]),
        dict(_record("same-id"), id=a["id"]),
        _record("survivor"),
    ]
    config, activity = _pool_config(tmp_path, records, monkeypatch, source_batch_size=3)
    pool = sources.prepare_pool(config)
    with closing(sources.source_records(pool)) as stream:
        sources.allocate_batch(config, pool, stream, dict.fromkeys(SPLITS, 100))
        old = copy.deepcopy(pool["documents"])
        sources.allocate_batch(config, pool, stream, {"train": 1, "dev": 0, "test": 0})
    assert pool["documents"][: len(old)] == old
    assert pool["documents"][len(old)]["document_id"] == "survivor"
    assert {doc["document_id"]: doc["split"] for doc in old} == {
        "a": "train",
        "b": "test",
        "dev": "dev",
    }
    assert pool["statistics"]["previous_pool_duplicate_documents"] == 2
    assert len(_used(config)) == 4 and activity["reads"] == list(range(6))


def test_previous_ledger_excludes_ids_urls_text_and_transitive_near_clusters_without_pool(
    tmp_path, monkeypatch
):
    words = [f"word{i:03d}" for i in range(120)]
    near = words.copy()
    near[20] = "changed"
    transitive = near.copy()
    transitive[70] = "altered"
    used = _record("used", text=" ".join(words))
    records = [
        used,
        _record("near", text=" ".join(near)),
        _record("transitive", text=" ".join(transitive)),
        _record("same-text", text=used["text"]),
        dict(_record("same-url"), url=used["url"]),
        _record("unused", text=" ".join(f"novel{i}" for i in range(120))),
    ]
    config, _ = _pool_config(tmp_path, records, monkeypatch)
    previous = tmp_path / "previous"
    reference = _reference(config, used, 0)
    write_used_sources(previous, [reference])
    config["previous_datasets"] = [str(previous)]
    pool = sources.prepare_pool(config)
    assert pool["excluded_sources"] == [reference]
    shutil.rmtree(previous)
    assert sources.prepare_pool(config) == pool
    with closing(sources.source_records(pool)) as stream:
        sources.allocate_batch(config, pool, stream, {"train": 1, "dev": 0, "test": 0})
    assert {d["document_id"] for d in pool["documents"]} == {"unused"}
    assert pool["statistics"]["previously_used_documents"] == 5
    assert [d["document_id"] for d in _used(config)] == ["unused"]


def test_empty_filtered_batches_continue_and_exhaustion_is_persisted(tmp_path, monkeypatch):
    records = [
        _record("short", text="tiny"),
        _record("length-short", text="z" * 80),
        _record("usable"),
    ]
    config, activity = _pool_config(tmp_path, records, monkeypatch, source_batch_size=1)
    pool = sources.prepare_pool(config)
    with closing(sources.source_records(pool)) as stream:
        sources.allocate_batch(config, pool, stream, {"train": 1, "dev": 0, "test": 0})
        assert activity["reads"] == [0, 1, 2]
        with pytest.raises(ValueError, match="exhausted"):
            sources.allocate_batch(config, pool, stream, {"train": 0, "dev": 1, "test": 0})
    assert sources.prepare_pool(config)["exhausted"]
    assert pool["statistics"]["source_batches"] == 3


def test_resume_skips_saved_source_prefix_and_repairs_interrupted_ledger(tmp_path, monkeypatch):
    records = [_record("first"), _record("second")]
    config, activity = _pool_config(tmp_path, records, monkeypatch, source_batch_size=1)
    pool = sources.prepare_pool(config)
    real_write = sources.write_used_sources
    monkeypatch.setattr(
        sources,
        "write_used_sources",
        lambda *_: (_ for _ in ()).throw(OSError("ledger interrupted")),
    )
    with closing(sources.source_records(pool)) as stream:
        with pytest.raises(OSError, match="ledger interrupted"):
            sources.allocate_batch(config, pool, stream, {"train": 1, "dev": 0, "test": 0})
    saved = load_json(Path(config["dataset_dir"]) / "source-pool.json")
    assert len(saved["batches"]) == 1 and _used(config) == []
    monkeypatch.setattr(sources, "write_used_sources", real_write)
    resumed = sources.prepare_pool(config)
    assert resumed["pool_id"] == pool["pool_id"]
    assert [d["document_id"] for d in _used(config)] == ["first"]
    with closing(sources.source_records(resumed)) as stream:
        sources.allocate_batch(config, resumed, stream, {"train": 1, "dev": 0, "test": 0})
    assert [d["document_id"] for d in _selection(config, 1)["documents"]] == ["second"]
    assert activity == {"opens": 2, "reads": [0, 0, 1], "closes": 2}
    assert resumed["statistics"]["scanned_documents"] == 2
    assert len(_used(config)) == 2


@pytest.mark.parametrize("length", [27648, 27649, 30000, 36864, 49152, 61440, 80000])
def test_variable_character_windows_always_fit_and_reproduce(length):
    config = SegmentationConfig(
        capacity=512,
        min_segment_ratio=1.5,
        max_segment_ratio=2,
        min_segments=6,
        max_segments=10,
        content_reserve_ratio=1.5,
    )
    for seed in range(40):
        windows = list(sample_windows(length, "document", seed, config))
        assert windows == list(sample_windows(length, "document", seed, config))
        cursor = 0
        for start, parts, end in windows:
            assert start == cursor
            assert 6 <= len(parts) <= min(10, (length - start) // 4608)
            assert 0 <= start < end <= length
            assert all(4608 <= config.reserved_chars(part) <= 6144 for part in parts)
            assert end - start == sum(config.reserved_chars(part) for part in parts)
            cursor = end
        assert length - cursor < config.minimum_window_chars
        if length == 27648:
            assert windows == [(0, [768] * 6, 27648)]


def test_long_documents_cover_all_segment_counts_and_lengths_vary():
    config = SegmentationConfig(
        capacity=512, min_segment_ratio=1.5, max_segment_ratio=2, min_segments=6, max_segments=10
    )
    windows = [next(sample_windows(60000, f"doc-{i}", 17, config)) for i in range(100)]
    assert {len(parts) for _, parts, _ in windows} == set(range(6, 11))
    assert len({tuple(parts) for _, parts, _ in windows}) == 100


@pytest.mark.parametrize("reserve_ratio", [1, 1.5, 1.1])
def test_factqa_pool_uses_shared_window_and_freezes_character_boundaries(
    tmp_path, monkeypatch, reserve_ratio
):
    record = _record("first", text="abcdefghij " * 100)
    config, _ = _pool_config(tmp_path, [record], monkeypatch)
    config["window"]["content_reserve_ratio"] = reserve_ratio
    segmentation = SegmentationConfig(**config["window"], continuation_tokens=0)
    pool = sources.prepare_pool(config)
    with closing(sources.source_records(pool)) as stream:
        sources.allocate_batch(config, pool, stream, {"train": 1, "dev": 0, "test": 0})
    frozen = pool["documents"][0]
    start, parts, end = next(
        sample_windows(len(record["text"]), record["id"], config["selection_seed"], segmentation)
    )
    assert frozen["window_char_span"] == [start, end]
    assert frozen["trajectory_id"] == f"first:{start}:{end}"
    assert [s["char_span"][1] - s["char_span"][0] for s in frozen["segments"]] == [
        segmentation.reserved_chars(n) for n in parts
    ]
    if reserve_ratio == 1.1:
        assert any((s["char_span"][1] - s["char_span"][0]) % 4 for s in frozen["segments"])
    assert "text" not in frozen


def test_generation_and_supplement_use_complete_reserved_segments(tmp_path, monkeypatch):
    original = _record("new", text="abcdefghij " * 100)
    config, _ = _pool_config(tmp_path, [original], monkeypatch)
    config["window"]["content_reserve_ratio"] = 1.5
    pool = sources.prepare_pool(config)
    with closing(sources.source_records(pool)) as stream:
        sources.allocate_batch(config, pool, stream, {"train": 1, "dev": 0, "test": 0})
    selected = _selection(config, 0)["documents"][0]
    calls = []

    class NoFactsClient:
        stop = Event()
        batch_root = tmp_path

        def call(self, stage, content, schema):
            assert stage == "generate"
            calls.append(content)
            return {"qas": [], "skip_reason": "No facts returned by the offline fixture."}

    result = {
        "rounds": [],
        "candidates": [],
        "local_candidates": [],
        "review_decisions": [],
        "assembly": None,
    }
    annotation = {
        "qa": {
            "role_seed": 17,
            "max_answer_chars": 128,
            "max_supplement_rounds": 1,
            "supplement_surplus": 2,
        }
    }
    for _ in range(2):
        pipeline._run_round(selected, result, annotation, NoFactsClient())
    expected = {
        s["segment_id"]: selected["text"][slice(*s["char_span"])] for s in selected["segments"]
    }
    assert len(calls) == 2 * len(expected)
    assert {(call["round_index"], call["segment_id"]) for call in calls} == {
        (i, s) for i in (0, 1) for s in expected
    }
    assert all(call["segment_text"] == expected[call["segment_id"]] for call in calls)
    assert selected["text"] == original["text"][slice(*selected["window_char_span"])]


def test_source_length_filter_includes_reserve_and_per_segment_rounding(tmp_path, monkeypatch):
    records = [_record("short", text="x" * 107), _record("exact", text="y" * 108)]
    config, _ = _pool_config(tmp_path, records, monkeypatch)
    config["window"]["content_reserve_ratio"] = 1.1
    pool = sources.prepare_pool(config)
    with closing(sources.source_records(pool)) as stream:
        sources.allocate_batch(config, pool, stream, {"train": 1, "dev": 0, "test": 0})
    assert pool["statistics"]["length_failed"] == 1
    assert [doc["document_id"] for doc in pool["documents"]] == ["exact"]
    assert pool["documents"][0]["window_char_span"] == [0, 108]
    assert all(
        s["char_span"][1] - s["char_span"][0] == 18 for s in pool["documents"][0]["segments"]
    )


@pytest.mark.parametrize("continuation", [0, 1])
def test_factqa_continuation_is_not_configurable(tmp_path, monkeypatch, continuation):
    config, _ = _pool_config(tmp_path, [_record("new")], monkeypatch)
    config["window"]["continuation_tokens"] = continuation
    with pytest.raises(ValueError, match="no configurable continuation_tokens"):
        sources.prepare_pool(config)


@pytest.mark.parametrize("index", [-1, 0, True, 0.5])
def test_unallocated_batch_indices_fail(index):
    with pytest.raises(ValueError, match="batch_index"):
        sources.batch_ranges({"batches": []}, index)


def test_invalid_window_bounds_fail_before_source_reading(tmp_path, monkeypatch):
    config, activity = _pool_config(tmp_path, [_record("new")], monkeypatch)
    config["window"]["min_segments"] = 11
    with pytest.raises(ValueError, match="segment counts"):
        sources.prepare_pool(config)
    assert not activity["reads"]
