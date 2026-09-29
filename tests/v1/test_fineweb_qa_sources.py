from __future__ import annotations

from pathlib import Path

import pytest

from latent_working_memory.data_preparation.fineweb_qa import sources
from latent_working_memory.data_preparation.pretrain.config import PreparationConfig
from latent_working_memory.data_preparation.pretrain.dedup import (
    cluster_documents,
    source_key,
)
from latent_working_memory.data_preparation.pretrain.fineweb import document_split


DATA_SEED = 20260907
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)


def _url_for_split(name: str, split: str) -> str:
    for index in range(1000):
        url = f"https://example.org/{name}/{index}"
        if document_split(source_key(url), DATA_SEED, SPLIT_FRACTIONS) == split:
            return url
    raise AssertionError(f"no {split} URL found for fixture")


def _paragraph_text(marker: str) -> str:
    assert len(marker) == 7
    words = [f"word{index:03d}" for index in range(120)]
    words[0] = marker
    return _paragraphs(words)


def _paragraphs(words: list[str]) -> str:
    assert len(words) == 120 and all(len(word) == 7 for word in words)
    return (
        "\n\n".join(" ".join(words[start : start + 15]) + "." for start in range(0, 120, 15))
        + "\n\n"
    )


def _record(name: str, text: str, split: str = "train") -> dict:
    return {"id": name, "url": _url_for_split(name, split), "text": text}


def _located(records: list[dict], source_file: Path) -> list[tuple[dict, dict]]:
    return [
        (
            record,
            {"source_file": str(source_file), "row_group": index // 3, "row_index": index % 3},
        )
        for index, record in enumerate(records)
    ]


def _config(
    tmp_path: Path,
    scan_documents: int,
    max_train_inspections: int,
    count: int,
    old_pool_documents: int = 1,
    near_duplicate_min_words: int = 1000,
) -> dict:
    raw_dir = tmp_path / "fineweb"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "000_00000.parquet").touch()
    return {
        "source": {
            "raw_dir": str(raw_dir),
            "file_count": 1,
            "data_seed": DATA_SEED,
            "split_fractions": list(SPLIT_FRACTIONS),
            "old_pool_documents": old_pool_documents,
            "scan_documents": scan_documents,
            "max_train_inspections": max_train_inspections,
            "near_duplicate_threshold": 0.9,
            "near_duplicate_min_words": near_duplicate_min_words,
            "selection_seed": 17,
        },
        "window": {
            "count": count,
            "segments": 8,
            "min_segment_chars": 118,
            "max_segment_chars": 123,
        },
    }


def _patch_records(monkeypatch, located: list[tuple[dict, dict]], source_file: Path) -> None:
    def fake_parquet_records(files, seed):
        assert files == [source_file]
        assert seed == DATA_SEED
        yield from located

    monkeypatch.setattr(sources, "parquet_records", fake_parquet_records)


def test_selection_order_and_budget_are_stable_under_scan_reordering(tmp_path, monkeypatch):
    config = _config(tmp_path, scan_documents=5, max_train_inspections=3, count=2)
    source_file = Path(config["source"]["raw_dir"]) / "000_00000.parquet"
    old = _record("old", _paragraph_text("old0000"))
    candidates = [
        _record(f"candidate-{index}", _paragraph_text(f"mark{index:03d}")) for index in range(6)
    ]
    located = _located([old, *candidates], source_file)

    _patch_records(monkeypatch, located, source_file)
    first = sources.prepare_selection(config)
    _patch_records(monkeypatch, [located[0], *reversed(located[1:6]), located[6]], source_file)
    second = sources.prepare_selection(config)

    first_ids = [document["document_id"] for document in first["documents"]]
    second_ids = [document["document_id"] for document in second["documents"]]
    assert first_ids == second_ids
    assert len(first_ids) == 2
    assert set(first_ids) <= {record["id"] for record in candidates[:5]}
    assert "candidate-5" not in first_ids
    assert first["statistics"]["old_pool_documents"] == 1
    assert first["statistics"]["scanned_documents"] == 5
    assert first["statistics"]["frozen_documents"] == 2
    assert all(document["split"] == "train" for document in first["documents"])
    assert all(
        Path(document["source"]["file"]).name == source_file.name for document in first["documents"]
    )


def test_old_pool_excludes_exact_and_transitively_near_duplicate_documents(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        scan_documents=4,
        max_train_inspections=4,
        count=2,
        near_duplicate_min_words=64,
    )
    source_file = Path(config["source"]["raw_dir"]) / "000_00000.parquet"
    original = [f"word{index:03d}" for index in range(120)]
    one_change = original.copy()
    one_change[20] = "changed"
    two_changes = one_change.copy()
    two_changes[70] = "altered"
    old = _record("old", _paragraphs(original))
    near_old = _record("near-old", _paragraphs(one_change))
    transitive = _record("transitive", _paragraphs(two_changes))
    exact_old = _record("exact-old", old["text"])
    survivor = _record("survivor", _paragraphs([f"item{index:03d}" for index in range(120)]))
    recipe = PreparationConfig(near_duplicate_threshold=0.9, near_duplicate_min_words=64)
    assert len(set(cluster_documents([old, near_old, transitive], recipe))) == 1
    assert len(set(cluster_documents([old, transitive], recipe))) == 2

    _patch_records(
        monkeypatch,
        _located([old, near_old, transitive, exact_old, survivor], source_file),
        source_file,
    )
    result = sources.prepare_selection(config)

    assert [document["document_id"] for document in result["documents"]] == ["survivor"]
    assert result["statistics"]["scanned_documents"] == 4
    assert result["statistics"]["old_pool_duplicate_documents"] == 3
    assert result["statistics"]["within_scan_duplicate_documents"] == 0
    assert result["statistics"]["frozen_documents"] == 1


def test_old_pool_matching_reaches_later_batches_and_excludes_new_cluster(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        old_pool_documents=2,
        scan_documents=3,
        max_train_inspections=3,
        count=2,
        near_duplicate_min_words=64,
    )
    source_file = Path(config["source"]["raw_dir"]) / "000_00000.parquet"
    original = [f"word{index:03d}" for index in range(120)]
    one_change = original.copy()
    one_change[20] = "changed"
    two_changes = one_change.copy()
    two_changes[70] = "altered"
    records = [
        _record("old-unrelated", _paragraphs([f"item{index:03d}" for index in range(120)])),
        _record("old-match", _paragraphs(original)),
        _record("near-old", _paragraphs(one_change)),
        _record("transitive", _paragraphs(two_changes)),
        _record("survivor", _paragraphs([f"gold{index:03d}" for index in range(120)])),
    ]
    located = _located(records, source_file)
    reads: list[list[str]] = []

    def fake_parquet_records(files, seed):
        assert files == [source_file]
        assert seed == DATA_SEED
        current: list[str] = []
        reads.append(current)
        for record, location in located:
            current.append(record["id"])
            yield record, location

    monkeypatch.setattr(sources, "parquet_records", fake_parquet_records)
    monkeypatch.setattr(sources, "_OLD_BATCH_SIZE", 1)

    result = sources.prepare_selection(config)

    assert [document["document_id"] for document in result["documents"]] == ["survivor"]
    assert result["statistics"]["old_pool_duplicate_documents"] == 2
    assert result["statistics"]["old_pool_documents"] == 2
    assert result["statistics"]["scanned_documents"] == 3
    assert reads == [
        ["old-unrelated", "old-match", "near-old", "transitive", "survivor"],
        ["old-unrelated", "old-match"],
    ]


def test_window_is_eight_contiguous_original_text_segments_with_relative_offsets(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, scan_documents=1, max_train_inspections=1, count=1)
    source_file = Path(config["source"]["raw_dir"]) / "000_00000.parquet"
    original_text = "Short preface.\n\n" + _paragraph_text("target0")
    target = _record("target", original_text)
    _patch_records(
        monkeypatch,
        _located([_record("old", _paragraph_text("old0000")), target], source_file),
        source_file,
    )

    result = sources.prepare_selection(config)
    assert len(result["documents"]) == 1
    document = result["documents"][0]
    start, end = document["window_char_span"]
    assert 0 <= start < end <= len(original_text)
    assert document["text"] == original_text[start:end]
    assert 8 * 118 <= end - start <= 8 * 123
    assert len(document["segments"]) == 8
    assert [segment["segment_id"] for segment in document["segments"]] == [
        f"seg{index}" for index in range(8)
    ]
    spans = [segment["char_span"] for segment in document["segments"]]
    assert spans[0][0] == 0
    assert spans[-1][1] == len(document["text"])
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
    assert all(118 <= b - a <= 123 for a, b in spans)
    assert [original_text[start + a : start + b] for a, b in spans] == [
        document["text"][a:b] for a, b in spans
    ]


@pytest.mark.parametrize("length", [944, 945, 950, 984, 1000, 4000])
def test_arbitrary_character_windows_always_fit_and_are_reproducible(length):
    config = {"segments": 8, "min_segment_chars": 118, "max_segment_chars": 123}
    text = "x" * length
    for seed in range(30):
        cuts = sources._window_cuts(text, "document", seed, config)
        assert cuts == sources._window_cuts(text, "document", seed, config)
        assert len(cuts) == 9
        assert 0 <= cuts[0] < cuts[-1] <= length
        assert all(118 <= b - a <= 123 for a, b in zip(cuts, cuts[1:]))
        if length == 944:
            assert cuts == list(range(0, 945, 118))


def test_documents_have_different_random_lengths_and_start_positions():
    config = {"segments": 8, "min_segment_chars": 3072, "max_segment_chars": 4096}
    cuts = [sources._window_cuts("x" * 40000, f"doc-{i}", 17, config) for i in range(30)]
    lengths = {tuple(b - a for a, b in zip(row, row[1:])) for row in cuts}
    assert len(lengths) == 30
    assert len({row[0] for row in cuts}) > 20


def test_shortage_statistics_keep_rejections_and_non_train_sources_visible(tmp_path, monkeypatch):
    config = _config(tmp_path, scan_documents=5, max_train_inspections=5, count=3)
    source_file = Path(config["source"]["raw_dir"]) / "000_00000.parquet"
    records = [
        _record("old", _paragraph_text("old0000")),
        _record("basic-short", "tiny"),
        _record("no-boundary", "x" * 974),
        _record("length-short", "X" * 100),
        _record("usable", _paragraph_text("usable0")),
        _record("dev", _paragraph_text("dev0000"), split="dev"),
    ]
    _patch_records(monkeypatch, _located(records, source_file), source_file)

    result = sources.prepare_selection(config)

    assert {document["document_id"] for document in result["documents"]} == {
        "usable",
        "no-boundary",
    }
    assert result["statistics"] == {
        "old_pool_documents": 1,
        "scanned_documents": 5,
        "old_pool_duplicate_documents": 0,
        "basic_rejected_documents": 1,
        "within_scan_duplicate_documents": 0,
        "non_train_documents": 1,
        "train_candidates": 3,
        "train_inspected": 3,
        "length_failed": 1,
        "train_uninspected": 0,
        "frozen_documents": 2,
    }
