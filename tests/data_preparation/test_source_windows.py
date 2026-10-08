"""同一原文的多轨迹身份、完整窗口不重叠与划分隔离。"""

import pytest

from latent_working_memory.data_preparation.fineweb_source import SourceWindowTracker


def record(start, end, document_id="doc", split="train", cluster="cluster", row=0):
    return {
        "document_id": document_id,
        "dedup_cluster": cluster,
        "split": split,
        "source": {"file": "source.parquet", "row_group": 0, "row_index": row},
        "window_char_span": [start, end],
    }


def test_nonoverlap_check_handles_out_of_order_adjacent_and_gapped_windows():
    tracker = SourceWindowTracker()
    for start, end in ((40, 50), (0, 10), (20, 30), (10, 20)):
        tracker.add(record(start, end))
    assert tracker.windows["doc"] == [(0, 10), (10, 20), (20, 30), (40, 50)]
    tracker.add(record(0, 10, document_id="other", row=1))


@pytest.mark.parametrize("span", [(0, 10), (5, 15), (15, 25), (10, 20), (0, 30)])
def test_partial_contained_and_duplicate_windows_are_rejected(span):
    tracker = SourceWindowTracker()
    tracker.add(record(10, 20))
    if span == (0, 10):
        tracker.add(record(*span))
    else:
        with pytest.raises(ValueError, match="overlapping source windows"):
            tracker.add(record(*span))


@pytest.mark.parametrize("start,overlap", [(20, True), (25, True), (30, False)])
def test_continuation_is_part_of_the_reserved_source_window(start, overlap):
    tracker = SourceWindowTracker()
    tracker.add(record(0, 20), continuation_chars=10)
    if overlap:
        with pytest.raises(ValueError, match="overlapping source windows"):
            tracker.add(record(start, start + 20), continuation_chars=10)
    else:
        tracker.add(record(start, start + 20), continuation_chars=10)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"split": "dev"}, "source document changes"),
        ({"cluster": "other"}, "source document changes"),
        ({"row": 1}, "source document changes"),
        ({"document_id": "other"}, "source row belongs"),
        ({"document_id": "other", "split": "dev", "row": 1}, "multiple dataset splits"),
    ],
)
def test_source_identity_and_cluster_split_are_consistent(changes, message):
    tracker = SourceWindowTracker()
    tracker.add(record(0, 10))
    with pytest.raises(ValueError, match=message):
        tracker.add(record(20, 30, **changes))
