"""QASPER keeps one annotation and locates whole paragraphs without answer leakage."""

import json

from latent_working_memory.v3_2.exploration.data_statistics.adapters.qasper import iter_qasper


def _annotation(evidence, unanswerable=False):
    return {"answer": {"unanswerable": unanswerable, "evidence": evidence}}


def test_uses_first_usable_annotation_without_merging_alternatives(tmp_path):
    first = "The first textual evidence paragraph is deliberately longer than the second."
    second = "Short alternative."
    paper = {
        "title": "Paper title",
        "abstract": "Paper abstract.",
        "full_text": [{"section_name": "Results", "paragraphs": [first, second]}],
        "qas": [
            {
                "question_id": "q",
                "answers": [
                    _annotation([], True),
                    _annotation(["FLOAT SELECTED: Table 1"]),
                    _annotation([first, first]),
                    _annotation([second]),
                ],
            }
        ],
    }
    path = tmp_path / "qasper-dev-v0.3.json"
    path.write_text(json.dumps({"paper-id": paper}), encoding="utf-8")
    document = next(iter_qasper(tmp_path, "dev"))
    assert document.source_ref["path"] == str(path)
    question = document.qas[0]
    assert question.exclusion_reason is None
    assert question.evidence_granularity == "paragraph"
    assert [document.text[a:b] for a, b in question.evidence_spans] == [first]
    assert document.source_ref["annotation_selection"]["q"] == {
        "selected_annotation_index": 2,
        "annotation_count": 4,
        "rejected_annotations": [
            {"index": 0, "reason": "unanswerable"},
            {"index": 1, "reason": "figure_or_table_evidence"},
        ],
    }
    assert document.source_ref["source_qa_count"] == 1
    assert document.source_ref["selected_qa_count"] == 1
    assert document.source_ref["qa_selection_rule"] == "all"


def test_exact_paragraph_offsets_and_unusable_annotations_remain_visible(tmp_path):
    abstract = "Evidence in the abstract."
    paragraph = "  A complete evidence paragraph.  "
    repeated = "Repeated paragraph."
    substring = _annotation(["evidence paragraph"])
    substring["answer"].update(extractive_spans=[paragraph], highlighted_evidence=[paragraph])
    paper = {
        "title": "Paper title",
        "abstract": abstract,
        "full_text": [
            {"section_name": paragraph, "paragraphs": [paragraph, repeated]},
            {"section_name": None, "paragraphs": [f" {repeated} "]},
        ],
        "qas": [
            {"question_id": "abstract", "answers": [_annotation([abstract])]},
            {
                "question_id": "paragraph",
                "answers": [
                    _annotation(["\tA complete evidence paragraph.\n"]),
                ],
            },
            {"question_id": "ambiguous", "answers": [_annotation([repeated])]},
            {"question_id": "substring", "answers": [substring]},
            {
                "question_id": "mixed",
                "answers": [
                    _annotation([], True),
                    _annotation([paragraph, "FLOAT SELECTED: Figure 1"]),
                ],
            },
            {"question_id": "unanswerable", "answers": [_annotation([], True)]},
            {"question_id": "missing", "answers": []},
        ],
    }
    path = tmp_path / "qasper-dev-v0.3.json"
    path.write_text(json.dumps({"paper-id": paper}), encoding="utf-8")
    document = next(iter_qasper(tmp_path, "dev"))
    assert document.text == "\n\n".join(
        [
            "Paper title",
            abstract,
            paragraph,
            paragraph,
            repeated,
            f" {repeated} ",
        ]
    )
    assert [document.text[a:b] for a, b in document.unit_spans] == [
        abstract,
        paragraph,
        repeated,
        f" {repeated} ",
    ]
    abstract_qa, paragraph_qa, *excluded = document.qas
    assert abstract_qa.evidence_spans == (document.unit_spans[0],)
    assert paragraph_qa.evidence_spans == (document.unit_spans[1],)
    assert [qa.exclusion_reason for qa in excluded] == [
        "evidence_locator_ambiguous",
        "evidence_locator_failed",
        "figure_or_table_evidence",
        "unanswerable",
        "missing_evidence_annotation",
    ]
