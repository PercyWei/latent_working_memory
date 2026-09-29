"""Deterministic diagnostic sampling, answer scores, and review materials."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import string


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)
_CANDIDATE_FIELDS = (
    "qa_id",
    "segment_id",
    "fact_statement",
    "question",
    "answer",
    "evidence_quote",
    "evidence_char_span",
    "answer_char_span",
)

_PANEL_FIELDS = (
    "qa_id",
    "document_index",
    "trajectory_id",
    "split",
    "segment_count",
    "segment_id",
    "question",
    "answer",
    "evidence_char_span",
    "evidence_text",
    "fact_group_id",
    "role",
)


def _rank(seed: int, trajectory_id: str, qa_id: str) -> bytes:
    identity = json.dumps([seed, trajectory_id, qa_id], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(identity.encode()).digest()


def _results_by_index(
    selection_documents: list[dict], document_results: list[dict]
) -> dict[int, dict]:
    indexed = {}
    for result in document_results:
        index = result["document_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(selection_documents)
        ):
            raise ValueError(f"invalid document_index: {index}")
        if index in indexed:
            raise ValueError(f"duplicate document result for index {index}")
        indexed[index] = result
    return indexed


def panel_items(document: dict, document_index: int, assembly: dict) -> list[dict]:
    trajectory = assembly["trajectory"]
    roles = {qa["qa_id"]: qa["role"] for qa in trajectory["qas"]} if trajectory else {}
    items = []
    spans = {segment["segment_id"]: segment["char_span"] for segment in document["segments"]}
    for qa in assembly["eligible_qas"]:
        start, end = qa["evidence_char_span"]
        left, right = spans[qa["segment_id"]]
        if not left <= start < end <= right <= len(document["text"]):
            raise ValueError(f"evidence outside segment: {qa['qa_id']}")
        items.append(
            {
                "qa_id": qa["qa_id"],
                "document_index": document_index,
                "trajectory_id": document["trajectory_id"],
                "split": document["split"],
                "segment_count": len(document["segments"]),
                "segment_id": qa["segment_id"],
                "question": qa["question"],
                "answer": qa["answer"],
                "evidence_char_span": [start, end],
                "evidence_text": document["text"][start:end],
                "fact_group_id": qa["fact_group_id"],
                "role": roles.get(qa["qa_id"]),
            }
        )
    return items


def select_review_panel(
    selection_documents: list[dict], document_results: list[dict], qas_per_segment: int, seed: int
) -> list[dict]:
    """Freeze sampling per document and segment, with no global sample-size cap."""
    if type(qas_per_segment) is not int or qas_per_segment <= 0:
        raise ValueError("qas_per_segment must be a positive integer")
    indexed = _results_by_index(selection_documents, document_results)
    panel = []
    for index, result in sorted(indexed.items()):
        document = selection_documents[index]
        items = panel_items(document, index, result["assembly"])
        for segment in document["segments"]:
            candidates = [qa for qa in items if qa["segment_id"] == segment["segment_id"]]
            candidates.sort(
                key=lambda qa: (_rank(seed, qa["trajectory_id"], qa["qa_id"]), qa["qa_id"])
            )
            panel.extend(candidates[:qas_per_segment])
    return panel


def _normalize_answer(answer: str) -> str:
    lower = answer.lower().translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", lower).split())


def _score(prediction: str, reference: str) -> dict[str, float]:
    predicted = _normalize_answer(prediction)
    expected = _normalize_answer(reference)
    left, right = predicted.split(), expected.split()
    overlap = sum((Counter(left) & Counter(right)).values())
    f1 = 2 * overlap / (len(left) + len(right)) if left and right else float(left == right)
    return {"em": float(predicted == expected), "f1": f1}


def score_answers(panel: list[dict], answers: dict[str, dict[str, str]]) -> dict:
    """Score both answer conditions with English SQuAD normalization."""
    qa_ids = [qa["qa_id"] for qa in panel]
    if len(set(qa_ids)) != len(qa_ids) or set(answers) != set(qa_ids):
        raise ValueError("answers must cover each panel QA exactly once")
    items = []
    for qa in panel:
        qa_id = qa["qa_id"]
        responses = answers[qa_id]
        if set(responses) != {"evidence", "question_only"} or any(
            not isinstance(value, str) for value in responses.values()
        ):
            raise ValueError(f"invalid diagnostic answers for {qa_id}")
        items.append(
            {
                "qa_id": qa_id,
                "evidence": {
                    "answer": responses["evidence"],
                    **_score(responses["evidence"], qa["answer"]),
                },
                "question_only": {
                    "answer": responses["question_only"],
                    **_score(responses["question_only"], qa["answer"]),
                },
            }
        )

    def aggregate(condition: str) -> dict[str, float | None]:
        if not items:
            return {"em": None, "f1": None}
        return {
            metric: sum(item[condition][metric] for item in items) / len(items)
            for metric in ("em", "f1")
        }

    return {
        "count": len(items),
        "evidence": aggregate("evidence"),
        "question_only": aggregate("question_only"),
        "items": items,
    }


def review_material(
    document: dict, document_index: int, result: dict, panel: list[dict], answers: dict
) -> dict:
    """One independent document context, excluding generator and verifier judgments."""
    return {
        "document_index": document_index,
        "trajectory_id": document["trajectory_id"],
        "text": document["text"],
        "segments": document["segments"],
        "candidates": [
            {field: qa[field] for field in _CANDIDATE_FIELDS} for qa in result["candidates"]
        ],
        "qas": [
            {
                **{field: qa[field] for field in _PANEL_FIELDS},
                "diagnostic_answers": answers[qa["qa_id"]],
            }
            for qa in panel
        ],
    }
