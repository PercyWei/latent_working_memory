"""Session evidence must follow the exact benchmark history after sorting."""

import json

import pytest

from benchmarks.longmemeval.data import LongMemEvalInput, Session, Turn, format_history
from latent_working_memory.v3_2.exploration.data_statistics.adapters.longmemeval import (
    iter_longmemeval,
)


def _sample():
    return {
        "question_id": "q1",
        "question_type": "temporal-reasoning",
        "question": "When did we meet Alice?",
        "question_date": "2026/11/26 (Thu) 12:00",
        "answer": "Monday",
        "answer_session_ids": ["early"],
        "haystack_session_ids": ["late", "early"],
        "haystack_dates": ["2026/11/25 (Wed) 12:00", "2026/11/23 (Mon) 12:00"],
        "haystack_sessions": [
            [{"role": "assistant", "content": "The later message.", "has_answer": True}],
            [{"role": "user", "content": "We met Alice.\nThen left.", "has_answer": False}],
        ],
    }


def test_sorted_session_evidence_matches_benchmark_format(tmp_path):
    row = _sample()
    path = tmp_path / "longmemeval_oracle.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    (document,) = iter_longmemeval(tmp_path, "longmemeval_oracle")
    assert document.source_ref["path"] == str(path)
    inputs = LongMemEvalInput(
        tuple(
            Session(
                row["haystack_dates"][index],
                tuple(
                    Turn(turn["role"], turn["content"]) for turn in row["haystack_sessions"][index]
                ),
            )
            for index in (1, 0)
        ),
        row["question"],
        row["question_date"],
    )
    assert document.text == format_history(inputs)
    assert document.source_ref["sessions"] == [
        {"session_index": 1, "session_id": "early"},
        {"session_index": 0, "session_id": "late"},
    ]
    (qa,) = document.qas
    assert qa.exclusion_reason is None
    assert qa.evidence_granularity == "session"
    assert qa.evidence_spans == (document.unit_spans[0],)
    start, end = qa.evidence_spans[0]
    assert (
        document.text[start:end]
        == "Session date: 2026/11/23 (Mon) 12:00\nuser: We met Alice.\nThen left."
    )
    assert "has_answer" not in document.text


@pytest.mark.parametrize(
    "answer_ids, reason, evidence_indices",
    [
        (["early"], None, (1,)),
        (["early", "late"], "evidence_locator_ambiguous", ()),
    ],
)
def test_repeated_session_id_keeps_history_and_requires_unique_evidence(
    tmp_path, answer_ids, reason, evidence_indices
):
    row = _sample()
    row["haystack_session_ids"].append("late")
    row["haystack_dates"].append("2026/11/22 (Sun) 12:00")
    row["haystack_sessions"].append(row["haystack_sessions"][0])
    row["answer_session_ids"] = answer_ids
    path = tmp_path / "longmemeval_s_cleaned.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    (document,) = iter_longmemeval(tmp_path, "longmemeval_s_cleaned")

    assert document.source_ref["sessions"] == [
        {"session_index": 2, "session_id": "late"},
        {"session_index": 1, "session_id": "early"},
        {"session_index": 0, "session_id": "late"},
    ]
    assert [document.text[start:end] for start, end in document.unit_spans] == [
        "Session date: 2026/11/22 (Sun) 12:00\nassistant: The later message.",
        "Session date: 2026/11/23 (Mon) 12:00\nuser: We met Alice.\nThen left.",
        "Session date: 2026/11/25 (Wed) 12:00\nassistant: The later message.",
    ]
    (qa,) = document.qas
    assert qa.exclusion_reason == reason
    assert qa.evidence_spans == tuple(document.unit_spans[index] for index in evidence_indices)


@pytest.mark.parametrize(
    "annotation, qid, reason",
    [
        (None, "q1", "missing_evidence_annotation"),
        ([], "q1", "missing_evidence_annotation"),
        (["early", "unknown"], "q1", "evidence_locator_failed"),
        (["early"], "q1_abs", "unanswerable"),
    ],
)
def test_missing_or_unlocatable_evidence_never_keeps_partial_sessions(
    tmp_path,
    annotation,
    qid,
    reason,
):
    row = _sample()
    row["question_id"] = qid
    if annotation is None:
        del row["answer_session_ids"]
    else:
        row["answer_session_ids"] = annotation
    path = tmp_path / "longmemeval_s_cleaned.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    (document,) = iter_longmemeval(tmp_path, "longmemeval_s_cleaned")
    (qa,) = document.qas
    assert qa.exclusion_reason == reason
    assert qa.evidence_spans == ()
    assert document.source_ref["source_qa_count"] == document.source_ref["selected_qa_count"] == 1
