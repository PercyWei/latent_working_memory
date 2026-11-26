"""Read official QASPER JSON using one complete paragraph annotation per QA.

Text is title, abstract, then section titles and paragraphs, joined with two
newlines. Only the abstract and body paragraphs are natural units. Evidence
must match a complete paragraph after stripping outer whitespace; returned
spans retain the original paragraph. Answer spans and sentence highlights are
never used as evidence.
"""

from collections.abc import Iterator
import json
from pathlib import Path

from ..schema import Document, Question, Span


def _question(qa: dict, paragraph_spans: dict[str, list[Span]]) -> tuple[Question, dict]:
    annotations = qa["answers"]
    selection = {
        "selected_annotation_index": None,
        "annotation_count": len(annotations),
        "rejected_annotations": [],
    }
    first_answerable_failure = None
    for index, annotation in enumerate(annotations):
        answer = annotation["answer"]
        evidence = answer["evidence"]
        if answer["unanswerable"]:
            reason = "unanswerable"
        elif not evidence:
            reason = "missing_evidence_annotation"
        elif any(paragraph.strip().startswith("FLOAT SELECTED") for paragraph in evidence):
            reason = "figure_or_table_evidence"
        else:
            reason = None
            spans = []
            for paragraph in evidence:
                locations = paragraph_spans.get(paragraph.strip(), [])
                if not locations:
                    reason = "evidence_locator_failed"
                    break
                if len(locations) > 1:
                    reason = "evidence_locator_ambiguous"
                    break
                spans.append(locations[0])
            if reason is None:
                selection["selected_annotation_index"] = index
                # Repeated labels for the same paragraph are one evidence piece.
                question = Question(qa["question_id"], tuple(dict.fromkeys(spans)), "paragraph")
                return question, selection
        selection["rejected_annotations"].append({"index": index, "reason": reason})
        if not answer["unanswerable"] and first_answerable_failure is None:
            first_answerable_failure = reason

    reason = first_answerable_failure or (
        "unanswerable" if annotations else "missing_evidence_annotation"
    )
    return Question(qa["question_id"], (), "paragraph", reason), selection


def iter_qasper(dataset_dir: Path, split: str) -> Iterator[Document]:
    """Use the first usable answer annotation in source order, never their union.

    A usable annotation is answerable, contains no figure/table evidence, and
    locates every evidence paragraph uniquely. If none is usable, the first
    answerable annotation determines the exclusion reason; only an entirely
    unanswerable set is labeled unanswerable. Selection metadata keeps original
    annotation indices and rejected reasons without copying evidence text.
    """
    if split not in ("train", "dev", "test"):
        raise ValueError("QASPER split must be train, dev or test")
    path = dataset_dir.expanduser().resolve() / f"qasper-{split}-v0.3.json"
    with path.open(encoding="utf-8") as stream:
        papers = json.load(stream)
    for paper_id, paper in papers.items():
        components = [(paper["title"], False), (paper["abstract"], True)]
        for section in paper["full_text"]:
            components.append((section["section_name"], False))
            components.extend((paragraph, True) for paragraph in section["paragraphs"])

        text_parts = []
        unit_spans = []
        paragraph_spans = {}
        position = 0
        for part, is_paragraph in components:
            if not part or not part.strip():
                continue
            if text_parts:
                position += 2
            span = (position, position + len(part))
            text_parts.append(part)
            if is_paragraph:
                unit_spans.append(span)
                paragraph_spans.setdefault(part.strip(), []).append(span)
            position = span[1]

        questions = []
        annotation_selection = {}
        for qa in paper["qas"]:
            question, selection = _question(qa, paragraph_spans)
            questions.append(question)
            annotation_selection[question.qa_id] = selection
        yield Document(
            doc_id=paper_id,
            dataset="qasper",
            split=split,
            text="\n\n".join(text_parts),
            unit_spans=tuple(unit_spans),
            qas=tuple(questions),
            source_ref={
                "path": str(path),
                "paper_id": paper_id,
                "source_qa_count": len(paper["qas"]),
                "selected_qa_count": len(questions),
                "qa_selection_rule": "all",
                "text_rule": "title_abstract_sections_double_newline",
                "evidence_selection_rule": "first_usable_paragraph_annotation_in_source_order",
                "annotation_selection": annotation_selection,
            },
            unit_kind="paragraph",
        )
