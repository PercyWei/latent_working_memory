from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latent_working_memory.data_preparation.fineweb_qa import sources
from latent_working_memory.data_preparation.pretrain.dedup import source_key
from latent_working_memory.data_preparation.pretrain.fineweb import document_split


def _text_settings() -> dict:
    return {
        "segments": 3,
        "min_segment_chars": 80,
        "target_segment_chars": 100,
        "max_segment_chars": 140,
        "min_trajectory_chars": 240,
        "max_trajectory_chars": 420,
    }


def _config(tmp_path: Path, old_count: int, files: list[str] | None = None) -> dict:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "source_seed": 20260907,
                "source_pool_candidates": old_count,
                "source_files": files or ["/former/server/000.parquet"],
            }
        )
    )
    return {
        "source": {
            "source_report": str(report),
            "raw_dir": str(tmp_path),
            "seed": 20260928,
            "split": "train",
            "scan_document_limit": 20,
            "candidate_document_limit": 20,
            "trajectory_limit": 10,
            "dedup_batch_documents": 1,
        },
        "text": _text_settings(),
    }


def _train_url(name: str) -> str:
    for i in range(100):
        url = f"https://example.com/{name}-{i}"
        if document_split(source_key(url), 20260907, (0.9, 0.05, 0.05)) == "train":
            return url
    raise AssertionError("could not make a train URL")


def _story(word: str) -> str:
    return "\n\n".join(
        " ".join(f"{word}{paragraph}{i}" for i in range(10)) + "."
        for paragraph in range(8)
    )


def _record(name: str, text: str | None = None, url: str | None = None) -> dict:
    return {"id": name, "url": url or _train_url(name), "text": text or _story(name)}


def _mock_records(monkeypatch, rows: list[dict]):
    calls = []

    def stream(files, seed):
        calls.append((files, seed))
        for i, row in enumerate(rows):
            yield row, {"source_file": str(files[0]), "row_group": 0, "row_index": i}

    monkeypatch.setattr(sources, "parquet_records", stream)
    return calls


def test_paragraph_windows_preserve_exact_text_and_eight_contiguous_segments():
    text = "  " + "\n\n".join(
        f"Paragraph {i}. " + ("A" * (3560 + i)) + "." for i in range(12)
    ) + "  "
    settings = {
        "segments": 8,
        "min_segment_chars": 3072,
        "target_segment_chars": 3584,
        "max_segment_chars": 4096,
        "min_trajectory_chars": 24576,
        "max_trajectory_chars": 32768,
    }
    window = sources.select_window(text, "document", 20260928, settings)
    assert window is not None
    assert window == sources.select_window(text, "document", 20260928, settings)
    assert window["text"] == text[window["char_start"] : window["char_end"]]
    assert len(window["segments"]) == 8
    fragments = []
    position = 0
    for i, segment in enumerate(window["segments"], 1):
        assert segment["segment_id"] == i
        assert segment["char_start"] == position
        position = segment["char_end"]
        fragment = window["text"][segment["char_start"] : position]
        assert 3072 <= len(fragment) <= 4096
        fragments.append(fragment)
    assert position == len(window["text"])
    assert "".join(fragments) == window["text"]


def test_sentence_fallback_does_not_cut_an_unfinished_sentence():
    text = " ".join(
        f"Sentence {i} describes a separate event with complete contextual information."
        for i in range(20)
    )
    settings = {**_text_settings(), "min_segment_chars": 60, "min_trajectory_chars": 180}
    window = sources.select_window(text, "document", 1, settings)
    assert window is not None
    for segment in window["segments"]:
        value = window["text"][segment["char_start"] : segment["char_end"]]
        assert value.rstrip().endswith(".")
    assert sources.select_window("A" * 1000, "unfinished", 1, settings) is None


def test_old_pool_exclusion_accumulates_and_propagates_to_new_cluster(monkeypatch, tmp_path):
    old_a = _record("old-a", _story("alpha"))
    old_b = _record("old-b", _story("beta"))
    # The first new document matches old A by text. Its new URL also joins a
    # different-text new record, so the exclusion must propagate through the new cluster.
    new_a = _record("new-a", old_a["text"], _train_url("shared"))
    new_b = _record("new-b", _story("gamma"), new_a["url"])
    # Near duplicate of old B, not an identical text, ID or URL.
    new_c = _record("new-c", old_b["text"].replace("beta00", "changedword", 1))
    survivor = _record("survivor", _story("unique"))
    calls = _mock_records(monkeypatch, [old_a, old_b, new_a, new_b, new_c, survivor])
    config = _config(tmp_path, 2)

    result = sources.prepare_sources(config)

    assert result["statistics"]["new_documents_matching_old_pool"] == 3
    assert result["statistics"]["new_clusters_matching_old_pool"] == 2
    assert {row["document_id"] for row in result["trajectories"]} == {"survivor"}
    assert len(calls) == 2
    assert all(seed == 20260907 for _, seed in calls)
    assert all(files == [tmp_path / "000.parquet"] for files, _ in calls)
    assert result["source_provenance"]["old_candidate_pool_size"] == 2
    assert result["statistics"]["trajectory_shortfall"] == 9


def test_scan_budget_and_fixed_seed_selection(monkeypatch, tmp_path):
    rows = [_record("old"), *[_record(f"candidate{i}") for i in range(10)]]
    _mock_records(monkeypatch, rows)
    config = _config(tmp_path, 1)
    config["source"].update(scan_document_limit=4, candidate_document_limit=2, trajectory_limit=1)

    result = sources.prepare_sources(config)

    assert result == sources.prepare_sources(config)
    assert result["statistics"]["new_documents_scanned"] == 4
    assert result["statistics"]["candidate_documents_selected"] == 2
    assert len(result["trajectories"]) == 1
    row = result["trajectories"][0]
    raw = next(record for record in rows if record["id"] == row["document_id"])
    assert row["text"] == raw["text"][
        row["source"]["window_char_start"] : row["source"]["window_char_end"]
    ]
    assert row["source"]["row_index"] in range(1, 5)


def test_source_exhaustion_and_bad_settings_are_errors(monkeypatch, tmp_path):
    _mock_records(monkeypatch, [_record("only")])
    config = _config(tmp_path, 2)
    with pytest.raises(ValueError, match="complete old candidate pool"):
        sources.prepare_sources(config)
    config["source"]["scan_document_limit"] = 0
    with pytest.raises(ValueError, match="scan_document_limit"):
        sources.prepare_sources(config)


def test_empty_continuation_reports_shortfall(monkeypatch, tmp_path):
    calls = _mock_records(monkeypatch, [_record("old")])
    result = sources.prepare_sources(_config(tmp_path, 1))
    assert result["trajectories"] == []
    assert result["statistics"]["new_documents_scanned"] == 0
    assert result["statistics"]["trajectory_shortfall"] == 10
    assert len(calls) == 1


def test_real_local_parquet_keeps_original_row_location(tmp_path):
    rows = [_record(f"local{i}") for i in range(8)]
    parquet_path = tmp_path / "000.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path, row_group_size=3)
    config = _config(tmp_path, 2)
    config["source"]["dedup_batch_documents"] = 2
    ordered = list(sources.parquet_records([parquet_path], 20260907))
    excluded_ids = {record["id"] for record, _ in ordered[:2]}

    result = sources.prepare_sources(config)

    assert result["statistics"]["new_documents_scanned"] == 6
    assert not excluded_ids.intersection(row["document_id"] for row in result["trajectories"])
    assert result["trajectories"]
    for trajectory in result["trajectories"]:
        location = trajectory["source"]
        row = (
            pq.ParquetFile(location["source_file"])
            .read_row_group(location["row_group"])
            .to_pylist()[location["row_index"]]
        )
        assert row["id"] == trajectory["document_id"]
