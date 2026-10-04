import copy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latent_working_memory.data_preparation.fineweb_qa import sources
from latent_working_memory.data_preparation.pretrain.dedup import source_key
from latent_working_memory.data_preparation.pretrain.fineweb import document_split


SEED = 20260907
FRACTIONS = (0.8, 0.1, 0.1)


def _record(name, split="train", text=None):
    for i in range(1000):
        url = f"https://example.org/{name}/{i}"
        if document_split(source_key(url), SEED, FRACTIONS) == split:
            return {"id": name, "url": url, "text": text or (name + " document. ") * 30}
    raise AssertionError("fixture split not found")


def _pool_config(tmp_path, records, monkeypatch, batch_counts=None):
    path = tmp_path / "000_00000.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)

    def stream(files, seed):
        assert files == [path] and seed == SEED
        yield from (
            (record, {"source_file": str(path), "row_group": 0, "row_index": i})
            for i, record in enumerate(records)
        )

    monkeypatch.setattr(sources, "parquet_records", stream)
    return {
        "source": {
            "raw_dir": str(tmp_path),
            "file_count": 1,
            "data_seed": SEED,
            "split_fractions": list(FRACTIONS),
            "old_pool_documents": 1,
            "scan_documents": len(records) - 1,
            "near_duplicate_threshold": 0.9,
            "near_duplicate_min_words": 1000,
            "selection_seed": 31,
        },
        "window": {
            "min_segments": 6,
            "max_segments": 10,
            "min_segment_chars": 16,
            "max_segment_chars": 24,
        },
        "batch_counts": batch_counts or {"train": 1, "dev": 1, "test": 1},
        "pool_dir": str(tmp_path / "pool"),
    }


def test_pool_freezes_all_splits_and_batches_do_not_recluster_or_overlap(tmp_path, monkeypatch):
    records = [_record("old")] + [
        _record(f"{split}-{i}", split) for split in FRACTION_NAMES for i in range(3)
    ]
    config = _pool_config(tmp_path, records, monkeypatch)
    pool = sources.prepare_pool(config)
    assert pool["split_counts"] == {"train": 3, "dev": 3, "test": 3}
    assert all("text" not in d for d in pool["documents"])
    original = Path(config["pool_dir"], "source-pool.json").read_bytes()
    monkeypatch.setattr(sources, "parquet_records", lambda *_: pytest.fail("pool must not rebuild"))
    assert sources.prepare_pool(config) == pool
    batches = [
        sources.prepare_selection(
            {
                "source_pool_dir": config["pool_dir"],
                "source_offsets": dict.fromkeys(FRACTION_NAMES, 0),
                "batch_index": i,
            }
        )
        for i in range(3)
    ]
    ids = [d["document_id"] for b in batches for d in b["documents"]]
    assert len(ids) == len(set(ids)) == 9
    assert all(
        b["statistics"]["selected_by_split"] == {"train": 1, "dev": 1, "test": 1} for b in batches
    )
    for batch in batches:
        assert batch["source_pool_id"] == pool["pool_id"]
        for d in batch["documents"]:
            frozen = pool["documents"][d["pool_index"]]
            assert {k: v for k, v in d.items() if k != "text"} == frozen
            assert (
                d["text"]
                == records[d["source"]["row_index"]]["text"][slice(*d["window_char_span"])]
            )
    assert Path(config["pool_dir"], "source-pool.json").read_bytes() == original
    changed = copy.deepcopy(config)
    changed["source"]["scan_documents"] += 1
    with pytest.raises(ValueError, match="configuration changed"):
        sources.prepare_pool(changed)
    with pytest.raises(ValueError, match="exhausted"):
        sources.prepare_selection(
            {
                "source_pool_dir": config["pool_dir"],
                "source_offsets": dict.fromkeys(FRACTION_NAMES, 0),
                "batch_index": 3,
            }
        )


FRACTION_NAMES = ("train", "dev", "test")


def test_exact_and_transitive_near_matches_to_old_pool_are_excluded(tmp_path, monkeypatch):
    words = [f"word{i:03d}" for i in range(120)]
    near = words.copy()
    near[20] = "changed"
    transitive = near.copy()
    transitive[70] = "altered"
    old = _record("old", text=" ".join(words))
    records = [
        old,
        _record("near", text=" ".join(near)),
        _record("transitive", text=" ".join(transitive)),
        _record("exact", text=old["text"]),
        _record("survivor", text=" ".join(f"item{i}" for i in range(120))),
    ]
    config = _pool_config(tmp_path, records, monkeypatch)
    config["source"]["near_duplicate_min_words"] = 64
    pool = sources.prepare_pool(config)
    assert [d["document_id"] for d in pool["documents"]] == ["survivor"]
    assert pool["statistics"]["old_pool_duplicate_documents"] == 3


def test_pool_reports_short_and_duplicate_documents_and_partial_final_batch(tmp_path, monkeypatch):
    a = _record("a")
    records = [
        _record("old"),
        a,
        dict(a, id="copy"),
        _record("short", text="tiny"),
        _record("length-short", text="z" * 80),
        _record("b"),
        _record("dev", "dev"),
    ]
    config = _pool_config(tmp_path, records, monkeypatch)
    pool = sources.prepare_pool(config)
    assert pool["statistics"]["within_scan_duplicate_documents"] == 1
    assert pool["statistics"]["basic_rejected_documents"] == 1
    assert pool["statistics"]["length_failed"] == 1
    first = sources.prepare_selection(
        {
            "source_pool_dir": config["pool_dir"],
            "source_offsets": dict.fromkeys(FRACTION_NAMES, 0),
            "batch_index": 0,
        }
    )
    second = sources.prepare_selection(
        {
            "source_pool_dir": config["pool_dir"],
            "source_offsets": dict.fromkeys(FRACTION_NAMES, 0),
            "batch_index": 1,
        }
    )
    assert first["statistics"]["selected_by_split"] == {"train": 1, "dev": 1, "test": 0}
    assert second["statistics"]["selected_by_split"] == {"train": 1, "dev": 0, "test": 0}
    assert all(row["stop"] >= row["start"] for row in second["ranges"].values())


@pytest.mark.parametrize("length", [18432, 18433, 20000, 24576, 32768, 40960, 80000])
def test_variable_character_windows_always_fit_and_reproduce(length):
    config = {
        "min_segments": 6,
        "max_segments": 10,
        "min_segment_chars": 3072,
        "max_segment_chars": 4096,
    }
    for seed in range(40):
        cuts = sources._window_cuts("x" * length, "document", seed, config)
        assert cuts == sources._window_cuts("x" * length, "document", seed, config)
        assert 6 <= len(cuts) - 1 <= min(10, length // 3072)
        assert 0 <= cuts[0] < cuts[-1] <= length
        assert all(3072 <= b - a <= 4096 for a, b in zip(cuts, cuts[1:]))
        if length == 18432:
            assert cuts == list(range(0, 18433, 3072))


def test_long_documents_cover_all_segment_counts_and_lengths_vary():
    config = {
        "min_segments": 6,
        "max_segments": 10,
        "min_segment_chars": 3072,
        "max_segment_chars": 4096,
    }
    cuts = [sources._window_cuts("x" * 60000, f"doc-{i}", 17, config) for i in range(100)]
    assert {len(c) - 1 for c in cuts} == set(range(6, 11))
    assert len({tuple(b - a for a, b in zip(c, c[1:])) for c in cuts}) == 100


def test_invalid_pool_bounds_and_batch_index_fail(tmp_path, monkeypatch):
    config = _pool_config(tmp_path, [_record("old"), _record("new")], monkeypatch)
    config["window"]["min_segments"] = 11
    with pytest.raises(ValueError, match="segment counts"):
        sources.prepare_pool(config)
    config["window"]["min_segments"] = 6
    sources.prepare_pool(config)
    with pytest.raises(ValueError, match="batch_index"):
        sources.prepare_selection(
            {
                "source_pool_dir": config["pool_dir"],
                "source_offsets": dict.fromkeys(FRACTION_NAMES, 0),
                "batch_index": -1,
            }
        )


def test_next_run_restarts_batch_zero_after_all_previously_selected_sources(tmp_path, monkeypatch):
    records = [_record("old")] + [
        _record(f"{split}-{i}", split) for split in FRACTION_NAMES for i in range(3)
    ]
    config = _pool_config(tmp_path, records, monkeypatch, {"train": 2, "dev": 3, "test": 3})
    pool = sources.prepare_pool(config)
    first = sources.prepare_selection(
        {
            "source_pool_dir": config["pool_dir"],
            "source_offsets": dict.fromkeys(FRACTION_NAMES, 0),
            "batch_index": 0,
        }
    )
    second_config = {
        "source_pool_dir": config["pool_dir"],
        "source_offsets": {s: r["stop"] for s, r in first["ranges"].items()},
        "batch_index": 0,
    }
    second = sources.prepare_selection(second_config)
    assert first["batch_index"] == second["batch_index"] == 0
    assert {d["document_id"] for d in first["documents"]}.isdisjoint(
        d["document_id"] for d in second["documents"]
    )
    assert second["statistics"]["selected_by_split"] == {"train": 1, "dev": 0, "test": 0}
    assert second["ranges"]["train"] == {"start": 2, "stop": 3, "requested": 2, "selected": 1}
    assert second == sources.prepare_selection(second_config)
    assert sources.prepare_pool(config) == pool


@pytest.mark.parametrize("offset", [-1, 4, True, 0.5])
def test_source_offsets_must_be_valid_pool_positions(offset):
    pool = {
        "split_counts": {s: 3 for s in FRACTION_NAMES},
        "config": {"batch_counts": {s: 1 for s in FRACTION_NAMES}},
    }
    with pytest.raises(ValueError, match="source_offsets.train"):
        sources.batch_ranges(pool, {"train": offset, "dev": 0, "test": 0}, 0)


def test_used_raw_text_is_excluded_with_new_ids_urls_and_near_copies(tmp_path, monkeypatch):
    words = [f"word{i:03d}" for i in range(120)]
    used = _record("used", text=" ".join(words))
    near = words.copy()
    near[20] = "changed"
    records = [
        _record("old"),
        used,
        _record("new-id-same-text", text=used["text"]),
        dict(_record("same-url", text="unrelated text " * 100), url=used["url"]),
        _record("near", text=" ".join(near)),
        _record("unused", text=" ".join(f"novel{i}" for i in range(120))),
    ]
    config = _pool_config(tmp_path, records, monkeypatch)
    config["source"]["near_duplicate_min_words"] = 64
    excluded = [
        {
            "document_id": used["id"],
            "dedup_cluster": source_key(used["url"]),
            "source": {"file": str(tmp_path / "000_00000.parquet"), "row_group": 0, "row_index": 1},
        }
    ]
    pool = sources.prepare_pool(config, excluded, ["previous-dataset"])
    assert [d["document_id"] for d in pool["documents"]] == ["unused"]
    assert pool["statistics"]["previously_used_documents"] == 4
    assert pool["excluded_sources"] == excluded
    assert sources.prepare_pool(config, excluded, ["previous-dataset"]) == pool
    with pytest.raises(ValueError, match="configuration changed"):
        sources.prepare_pool(config)


def test_unchanged_pool_snapshot_filters_whole_documents_without_reading_raw(tmp_path, monkeypatch):
    records = [_record("old"), _record("a"), _record("b")]
    config = _pool_config(tmp_path, records, monkeypatch)
    pool = sources.prepare_pool(config)
    used = {key: pool["documents"][0][key] for key in ("document_id", "dedup_cluster", "source")}
    monkeypatch.setattr(
        sources, "parquet_records", lambda *_: pytest.fail("snapshot must not rescan raw text")
    )
    new = sources.pool_after_exclusions(pool, [used], ["previous-dataset"])
    assert new["pool_id"] != pool["pool_id"]
    assert len(new["documents"]) == 1 and new["documents"][0]["document_id"] != used["document_id"]
    assert new["documents"][0]["pool_index"] == 0
    assert len(pool["documents"]) == 2
