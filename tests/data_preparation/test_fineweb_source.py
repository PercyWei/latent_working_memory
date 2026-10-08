"""公共来源目录、配额比例与已用原文登记。"""

import json

import pytest

from latent_working_memory.data_preparation import fineweb_source
from latent_working_memory.data_preparation.fineweb_source import (
    load_previous_sources,
    source_files,
    split_fractions,
    write_used_sources,
)


def reference(identity, row=0):
    return {
        "document_id": identity,
        "dedup_cluster": f"example/{identity}",
        "source": {"file": "source.parquet", "row_group": 0, "row_index": row},
    }


def test_source_directory_uses_only_sorted_direct_parquet_files(tmp_path):
    for name in ("b.parquet", "a.parquet", "c.parquet.part", "readme.txt"):
        (tmp_path / name).touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.parquet").touch()
    (tmp_path / "directory.parquet").mkdir()
    assert source_files(tmp_path) == [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    with pytest.raises(ValueError, match="no FineWeb Parquet"):
        source_files(tmp_path / "empty")


@pytest.mark.parametrize(
    "counts",
    [
        None,
        {},
        {"train": 0, "dev": 1, "test": 1},
        {"train": True, "dev": 0, "test": 0},
        {"train": 1.0, "dev": 0, "test": 0},
        {"train": 1, "dev": -1, "test": 0},
        {"train": 1, "dev": 0, "test": 0, "extra": 1},
    ],
)
def test_split_counts_reject_invalid_contract(counts):
    with pytest.raises(ValueError, match="split_counts"):
        split_fractions(counts)


def test_split_probabilities_derive_only_from_counts_in_canonical_order():
    assert split_fractions({"test": 1, "train": 8, "dev": 1}) == (0.8, 0.1, 0.1)
    assert split_fractions({"train": 1, "dev": 0, "test": 0}) == (1, 0, 0)
    assert split_fractions({"train": 32000, "dev": 128, "test": 128}) == pytest.approx(
        (32000 / 32256, 128 / 32256, 128 / 32256)
    )


def test_ledgers_and_legacy_metadata_merge_only_explicit_directories(tmp_path):
    current, legacy, history = (tmp_path / name for name in ("current", "legacy", "history"))
    write_used_sources(current, [reference("b", 1), reference("a"), reference("a")])
    write_used_sources(history, [reference("history")])
    legacy.mkdir()
    (legacy / "preparation.json").write_text(
        json.dumps(
            {
                "used_sources": [reference("b", 1), reference("c", 2)],
                "previous_datasets": [str(history)],
            }
        )
    )
    (current / "preparation.json").write_text(json.dumps({"used_sources": [reference("ignored")]}))
    assert load_previous_sources([current, legacy, current]) == [
        reference("a"),
        reference("b", 1),
        reference("c", 2),
    ]
    assert [
        json.loads(line) for line in (current / "used-sources.jsonl").read_text().splitlines()
    ] == [reference("a"), reference("b", 1)]


def test_location_conflict_preserves_existing_ledger_and_rejects_cross_dataset_merge(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_used_sources(first, [reference("a")])
    before = (first / "used-sources.jsonl").read_bytes()
    with pytest.raises(ValueError, match="inconsistent source locations"):
        write_used_sources(first, [reference("a"), reference("a", 9)])
    assert (first / "used-sources.jsonl").read_bytes() == before
    write_used_sources(second, [reference("a", 9)])
    with pytest.raises(ValueError, match="inconsistent source locations"):
        load_previous_sources([first, second])


def test_failed_atomic_replacement_keeps_prior_ledger_and_removes_temporary(tmp_path, monkeypatch):
    write_used_sources(tmp_path, [reference("a")])
    before = (tmp_path / "used-sources.jsonl").read_bytes()

    def fail_replace(*_):
        raise OSError("replacement failed")

    monkeypatch.setattr(fineweb_source.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        write_used_sources(tmp_path, [reference("b")])
    assert (tmp_path / "used-sources.jsonl").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["used-sources.jsonl"]


def test_missing_new_ledger_does_not_fall_back_to_stale_embedded_sources(tmp_path):
    (tmp_path / "preparation.json").write_text(
        json.dumps({"used_sources_file": "used-sources.jsonl", "used_sources": [reference("old")]})
    )
    with pytest.raises(FileNotFoundError, match="missing used source ledger"):
        load_previous_sources([tmp_path])
