"""Read existing datasets without changing their text or evidence offsets."""

from collections.abc import Iterator
import json
from pathlib import Path
import re

from ..schema import Document, Question

FINEWEB_FACTQA_FILES = {split: f"{split}.jsonl" for split in ("train", "dev", "test")}


def _fineweb_question(qa: dict, text: str) -> Question:
    evidence = qa.get("evidence_char_span")
    if evidence is None or evidence == []:
        return Question(qa["qa_id"], (), "quoted_span", "missing_evidence_annotation")
    start, end = evidence
    answer_start, answer_end = qa["answer_char_span"]
    if (
        not start <= answer_start < answer_end <= end
        or text[answer_start:answer_end] != qa["answer"]
    ):
        return Question(qa["qa_id"], (), "quoted_span", "evidence_locator_failed")
    return Question(qa["qa_id"], ((start, end),), "quoted_span")


def iter_fineweb_factqa(dataset_dir: Path, split: str) -> Iterator[Document]:
    """Keep every document; select QA roles and use nonempty newline paragraphs.

    FineWeb's stored ``segments`` are artificial annotation ranges, not natural
    paragraphs. A text without line breaks remains one unit. Offsets always
    refer to the unmodified JSONL ``text`` string.
    """
    if split not in FINEWEB_FACTQA_FILES:
        raise ValueError("FineWeb FactQA split must be train, dev or test")
    path = dataset_dir.expanduser().resolve() / FINEWEB_FACTQA_FILES[split]
    role = "train" if split == "train" else "evaluation"
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row["split"] != split:
                raise ValueError(f"{path}:{line_number}: expected split {split!r}")
            text = row["text"]
            units = tuple(
                match.span() for match in re.finditer(r"[^\r\n]+", text) if match.group().strip()
            )
            qas = tuple(_fineweb_question(qa, text) for qa in row["qas"] if qa["role"] == role)
            yield Document(
                doc_id=row["trajectory_id"],
                dataset="fineweb_factqa",
                split=split,
                text=text,
                unit_spans=units,
                qas=qas,
                unit_kind="newline_paragraph",
                source_ref={
                    "path": str(path),
                    "line": line_number,
                    "trajectory_id": row["trajectory_id"],
                    "document_id": row["document_id"],
                    "dedup_cluster": row["dedup_cluster"],
                    "source": row["source"],
                    "window_char_span": row["window_char_span"],
                    "qa_role": role,
                    "qa_selection_rule": f"role:{role}",
                    "source_qa_count": len(row["qas"]),
                    "selected_qa_count": len(qas),
                },
            )
