"""Read LongMemEval using complete annotated sessions as evidence."""

from collections.abc import Iterator
import json
from pathlib import Path

from benchmarks.longmemeval.data import LONGMEMEVAL_FILES, SESSION_HEADER, TURN_FORMAT

from ..schema import Document, Question


def iter_longmemeval(dataset_dir: Path, split: str) -> Iterator[Document]:
    """Keep dates and complete sessions; turn-level ``has_answer`` is not used.

    Session evidence is deliberately coarse and includes the date header.
    Abstention, absent annotation, and unlocatable session IDs remain distinct
    QA exclusions. All answer sessions must exist; there is no partial fallback.
    """
    if split not in LONGMEMEVAL_FILES:
        raise ValueError(f"unsupported LongMemEval split: {split!r}")
    path = dataset_dir.expanduser().resolve() / LONGMEMEVAL_FILES[split]
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    for record_index, row in enumerate(rows):
        qid = row["question_id"]
        sessions = list(
            zip(
                row["haystack_dates"],
                row["haystack_session_ids"],
                row["haystack_sessions"],
                strict=True,
            )
        )
        if len({session_id for _, session_id, _ in sessions}) != len(sessions):
            raise ValueError(f"{qid}: duplicate haystack_session_ids")
        order = sorted(range(len(sessions)), key=lambda index: sessions[index][0])
        parts, units, references, spans = [], [], [], {}
        cursor = 0
        for index in order:
            date, session_id, turns = sessions[index]
            lines = [SESSION_HEADER.format(date=date)]
            lines.extend(
                TURN_FORMAT.format(role=turn["role"], content=turn["content"]) for turn in turns
            )
            text = "\n".join(lines)
            span = (cursor, cursor + len(text))
            parts.append(text)
            units.append(span)
            spans[session_id] = span
            references.append({"session_index": index, "session_id": session_id})
            cursor += len(text) + 2

        answer_ids = row.get("answer_session_ids")
        reason = None
        if "_abs" in qid:
            reason = "unanswerable"
        elif not answer_ids:
            reason = "missing_evidence_annotation"
        elif any(session_id not in spans for session_id in answer_ids):
            reason = "evidence_locator_failed"
        evidence = (
            ()
            if reason
            else tuple(span for session_id, span in spans.items() if session_id in answer_ids)
        )
        yield Document(
            doc_id=qid,
            dataset="longmemeval",
            split=split,
            text="\n\n".join(parts),
            unit_spans=tuple(units),
            unit_kind="session",
            qas=(Question(qid, evidence, "session", reason),),
            source_ref={
                "path": str(path),
                "record_index": record_index,
                "question_id": qid,
                "sessions": references,
                "answer_session_ids": answer_ids,
                "source_qa_count": 1,
                "selected_qa_count": 1,
                "qa_selection_rule": "all",
            },
        )
