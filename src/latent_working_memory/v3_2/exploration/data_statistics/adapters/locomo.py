"""Read official LoCoMo conversations and locate turn-level evidence."""

from collections.abc import Iterator
import json
from pathlib import Path
import re

from benchmarks.data import LOCOMO_CATEGORIES, _locomo_history

from ..schema import Document, Question


def iter_locomo(dataset_dir: Path) -> Iterator[Document]:
    """Keep whole conversations; select QA categories 1–4."""
    path = dataset_dir.expanduser().resolve() / "locomo10.json"
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    for row_index, row in enumerate(rows):
        location = f"{path}[{row_index}]"
        conversation = row["conversation"]
        text, _, _ = _locomo_history(conversation, location)
        session_keys = sorted(
            (key for key in conversation if re.fullmatch(r"session_\d+", key)),
            key=lambda key: int(key.split("_")[1]),
        )
        units, by_id = [], {}
        cursor = 0
        for key in session_keys:
            for turn in conversation[key]:
                rendered = f'{turn["speaker"]} said, "{turn["text"]}"'
                if "blip_caption" in turn:
                    rendered += f" and shared {turn['blip_caption']}."
                start = text.index(rendered, cursor)
                cursor = start + len(rendered)
                if turn["dia_id"] in by_id:
                    raise ValueError(f"{location}: duplicate dia_id {turn['dia_id']!r}")
                by_id[turn["dia_id"]] = (start, cursor)
                units.append((start, cursor))

        qas = []
        for qa_index, qa in enumerate(row["qa"]):
            if qa["category"] not in LOCOMO_CATEGORIES:
                continue
            qa_id = f"locomo/{row['sample_id']}/{qa_index}"
            evidence = qa.get("evidence")
            if not evidence:
                qas.append(Question(qa_id, (), "turn", "missing_evidence_annotation"))
                continue
            references = [
                reference
                for value in evidence
                for reference in re.split(r"[;\s]+", value.strip())
                if reference
            ]
            if not references or any(reference not in by_id for reference in references):
                qas.append(Question(qa_id, (), "turn", "evidence_locator_failed"))
                continue
            qas.append(Question(qa_id, tuple(by_id[reference] for reference in references), "turn"))

        yield Document(
            doc_id=row["sample_id"],
            dataset="locomo",
            split=None,
            text=text,
            unit_spans=tuple(units),
            qas=tuple(qas),
            source_ref={
                "path": str(path),
                "row_index": row_index,
                "sample_id": row["sample_id"],
                "qa_selection_rule": "category:1-4",
                "source_qa_count": len(row["qa"]),
                "selected_qa_count": len(qas),
            },
            unit_kind="turn",
        )
