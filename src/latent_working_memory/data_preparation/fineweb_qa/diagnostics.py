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


def select_review_panel(
    selection_documents: list[dict], document_results: list[dict], sample_size: int, seed: int
) -> list[dict]:
    """Sample eligible facts evenly across frozen documents and source segments."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
        raise ValueError("sample_size must be a nonnegative integer")
    indexed = _results_by_index(selection_documents, document_results)
    remaining = []
    seen_qa_ids = set()
    for document_index, result in sorted(indexed.items()):
        document = selection_documents[document_index]
        text = document["text"]
        segment_spans = {
            segment["segment_id"]: segment["char_span"] for segment in document["segments"]
        }
        trajectory = result["assembly"]["trajectory"]
        roles = (
            {qa["qa_id"]: qa["role"] for qa in trajectory["qas"]} if trajectory is not None else {}
        )
        for qa in result["assembly"]["eligible_qas"]:
            qa_id = qa["qa_id"]
            if qa_id in seen_qa_ids:
                raise ValueError(f"duplicate eligible QA ID: {qa_id}")
            seen_qa_ids.add(qa_id)
            start, end = qa["evidence_char_span"]
            segment_start, segment_end = segment_spans[qa["segment_id"]]
            if not segment_start <= start < end <= segment_end <= len(text):
                raise ValueError(f"evidence is outside its segment for {qa_id}")
            remaining.append(
                (
                    {
                        "qa_id": qa_id,
                        "document_index": document_index,
                        "trajectory_id": document["trajectory_id"],
                        "segment_id": qa["segment_id"],
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "evidence_char_span": [start, end],
                        "evidence_text": text[start:end],
                        "fact_group_id": qa["fact_group_id"],
                        "role": roles.get(qa_id),
                    },
                    _rank(seed, document["trajectory_id"], qa_id),
                )
            )

    selected = []
    document_counts: Counter[int] = Counter()
    document_segment_counts: Counter[tuple[int, str]] = Counter()
    segment_counts: Counter[str] = Counter()
    for _ in range(min(sample_size, len(remaining))):
        choice = min(
            range(len(remaining)),
            key=lambda index: (
                document_counts[remaining[index][0]["document_index"]],
                document_segment_counts[
                    (remaining[index][0]["document_index"], remaining[index][0]["segment_id"])
                ],
                segment_counts[remaining[index][0]["segment_id"]],
                remaining[index][1],
                remaining[index][0]["qa_id"],
            ),
        )
        item, _ = remaining.pop(choice)
        selected.append(item)
        document_index = item["document_index"]
        segment_id = item["segment_id"]
        document_counts[document_index] += 1
        document_segment_counts[document_index, segment_id] += 1
        segment_counts[segment_id] += 1
    return selected


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


def review_shards(
    panel: list[dict],
    selection_documents: list[dict],
    document_results: list[dict],
    answers: dict[str, dict[str, str]],
) -> tuple[dict, dict]:
    """Package two independent review assignments without prior model judgments."""
    if len(panel) > 96:
        raise ValueError("the fixed review panel has at most 96 QAs")
    indexed = _results_by_index(selection_documents, document_results)
    if len({qa["qa_id"] for qa in panel}) != len(panel) or set(answers) != {
        qa["qa_id"] for qa in panel
    }:
        raise ValueError("answers must cover each panel QA exactly once")

    def shard(number: int, assigned: list[dict]) -> dict:
        qas = []
        for qa in assigned:
            responses = answers[qa["qa_id"]]
            if set(responses) != {"evidence", "question_only"} or any(
                not isinstance(value, str) for value in responses.values()
            ):
                raise ValueError(f"invalid diagnostic answers for {qa['qa_id']}")
            qas.append(
                {
                    **{field: qa[field] for field in _PANEL_FIELDS},
                    "diagnostic_answers": {
                        "evidence": responses["evidence"],
                        "question_only": responses["question_only"],
                    },
                }
            )

        documents = []
        for document_index in sorted({qa["document_index"] for qa in assigned}):
            document = selection_documents[document_index]
            result = indexed[document_index]
            text = document["text"]
            documents.append(
                {
                    "document_index": document_index,
                    "trajectory_id": document["trajectory_id"],
                    "document_id": document["document_id"],
                    "text": text,
                    "segments": [
                        {
                            "segment_id": segment["segment_id"],
                            "char_span": list(segment["char_span"]),
                            "text": text[slice(*segment["char_span"])],
                        }
                        for segment in document["segments"]
                    ],
                    "candidates": [
                        {field: candidate[field] for field in _CANDIDATE_FIELDS}
                        for candidate in result["candidates"]
                    ],
                }
            )
        return {"shard": number, "qas": qas, "documents": documents}

    targets = [(len(panel) + 1) // 2, len(panel) // 2]
    assigned_counts = [0, 0]
    assignment = {}
    by_document = {}
    for qa in panel:
        by_document.setdefault(qa["document_index"], []).append(qa["qa_id"])
    for _document_index, qa_ids in sorted(
        by_document.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        room = [target - count for target, count in zip(targets, assigned_counts, strict=True)]
        fitting = [index for index in (0, 1) if len(qa_ids) <= room[index]]
        if fitting:
            chosen = max(fitting, key=lambda index: (room[index], -index))
            for qa_id in qa_ids:
                assignment[qa_id] = chosen
            assigned_counts[chosen] += len(qa_ids)
        else:
            first_count = min(len(qa_ids), room[0])
            for qa_id in qa_ids[:first_count]:
                assignment[qa_id] = 0
            for qa_id in qa_ids[first_count:]:
                assignment[qa_id] = 1
            assigned_counts[0] += first_count
            assigned_counts[1] += len(qa_ids) - first_count
    return (
        shard(1, [qa for qa in panel if assignment[qa["qa_id"]] == 0]),
        shard(2, [qa for qa in panel if assignment[qa["qa_id"]] == 1]),
    )
