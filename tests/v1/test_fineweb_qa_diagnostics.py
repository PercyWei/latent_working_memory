import copy
from collections import Counter
import json

import pytest

from latent_working_memory.data_preparation.fineweb_qa.diagnostics import (
    review_material,
    score_answers,
    select_review_panel,
)


def _documents_and_results(count: int) -> tuple[list[dict], list[dict]]:
    documents, results = [], []
    for document_index in range(count):
        text_parts, segments, candidates = [], [], []
        offset = 0
        for segment_index in range(8):
            segment_id = f"seg{segment_index}"
            segment_text = ""
            for item_index in range(2):
                qa_id = f"doc{document_index}:{segment_id}:qa{item_index}"
                answer = f"value {document_index}-{segment_index}-{item_index}"
                line = f"Café ☕ fact {segment_index}-{item_index} has {answer}.\n"
                evidence_start = offset + len(segment_text)
                answer_start = evidence_start + line.index(answer)
                candidates.append(
                    {
                        "qa_id": qa_id,
                        "segment_id": segment_id,
                        "fact_statement": f"Fact {segment_index}-{item_index} has {answer}",
                        "question": f"What does fact {segment_index}-{item_index} have?",
                        "answer": answer,
                        "evidence_quote": line,
                        "evidence_char_span": [evidence_start, evidence_start + len(line)],
                        "answer_char_span": [answer_start, answer_start + len(answer)],
                        "reason": "SECRET PRIOR VERIFIER REASON",
                    }
                )
                segment_text += line
            text_parts.append(segment_text)
            segments.append(
                {"segment_id": segment_id, "char_span": [offset, offset + len(segment_text)]}
            )
            offset += len(segment_text)
        document = {
            "trajectory_id": f"trajectory-{document_index}",
            "document_id": f"document-{document_index}",
            "split": "train",
            "text": "".join(text_parts),
            "segments": segments,
        }
        eligible = [
            {
                key: candidate[key]
                for key in ("qa_id", "segment_id", "question", "answer", "evidence_char_span")
            }
            | {"fact_group_id": candidate["qa_id"]}
            for candidate in candidates
        ]
        trajectory = (
            {"qas": [{"qa_id": qa["qa_id"], "role": "train"} for qa in eligible[:4]]}
            if document_index % 2 == 0
            else None
        )
        results.append(
            {
                "document_index": document_index,
                "assembly": {
                    "ok": trajectory is not None,
                    "trajectory": trajectory,
                    "eligible_qas": eligible,
                },
                "candidates": candidates,
                "verification_decisions": [{"reason": "SECRET PRIOR VERIFIER REASON"}],
            }
        )
        documents.append(document)
    return documents, results


def test_review_panel_is_deterministic_and_spread_across_documents_and_segments():
    documents, results = _documents_and_results(4)
    first = select_review_panel(documents, results, qas_per_segment=1, seed=20260928)
    reordered = copy.deepcopy(results[::-1])
    for result in reordered:
        result["assembly"]["eligible_qas"].reverse()
    second = select_review_panel(documents, reordered, qas_per_segment=1, seed=20260928)
    assert first == second

    by_document = Counter(qa["document_index"] for qa in first)
    by_segment = Counter(qa["segment_id"] for qa in first)
    assert by_document == {0: 8, 1: 8, 2: 8, 3: 8}
    assert by_segment == {f"seg{index}": 4 for index in range(8)}
    assert all(
        len({qa["segment_id"] for qa in first if qa["document_index"] == index}) == 8
        for index in range(4)
    )
    assert {qa["document_index"] for qa in first} == {0, 1, 2, 3}
    assert any(qa["document_index"] == 1 and qa["role"] is None for qa in first)
    for qa in first:
        document = documents[qa["document_index"]]
        start, end = qa["evidence_char_span"]
        assert qa["evidence_text"] == document["text"][start:end]
        assert qa["evidence_text"].startswith("Café ☕")


def test_squad_normalized_scores_include_each_condition_and_item():
    panel = [
        {"qa_id": "one", "answer": "The Orion, Alpha!"},
        {"qa_id": "two", "answer": "a red car"},
    ]
    answers = {
        "one": {"evidence": "orion alpha", "question_only": "Orion"},
        "two": {"evidence": "red car", "question_only": ""},
    }
    result = score_answers(panel, answers)
    assert result["count"] == 2
    assert result["evidence"] == {"em": 1.0, "f1": 1.0}
    assert result["question_only"]["em"] == 0.0
    assert result["question_only"]["f1"] == pytest.approx(1 / 3)
    assert result["items"][0]["question_only"] == {
        "answer": "Orion",
        "em": 0.0,
        "f1": pytest.approx(2 / 3),
    }
    assert score_answers([], {}) == {
        "count": 0,
        "evidence": {"em": None, "f1": None},
        "question_only": {"em": None, "f1": None},
        "items": [],
    }


def test_review_material_covers_each_document_and_hides_prior_judgments():
    documents, results = _documents_and_results(20)
    panel = select_review_panel(documents, results, qas_per_segment=1, seed=29)
    assert len(panel) == 160  # Coverage scales beyond the previous 96-question cap.
    panel[0]["prior_reason"] = "SECRET PRIOR VERIFIER REASON"
    answers = {q["qa_id"]: {"evidence": q["answer"], "question_only": ""} for q in panel}
    materials = [
        review_material(d, i, results[i], [q for q in panel if q["document_index"] == i], answers)
        for i, d in enumerate(documents)
    ]
    assert all(len(m["qas"]) == 8 for m in materials)
    assert len({q["qa_id"] for m in materials for q in m["qas"]}) == 160
    assert "SECRET PRIOR VERIFIER REASON" not in json.dumps(materials)
    assert materials[0]["text"] == documents[0]["text"]
    assert materials[0]["qas"][0]["diagnostic_answers"]["evidence"] == panel[0]["answer"]


def test_review_panel_rejects_qa_span_outside_its_source_segment():
    documents, results = _documents_and_results(1)
    qa = results[0]["assembly"]["eligible_qas"][0]
    qa["evidence_char_span"] = documents[0]["segments"][1]["char_span"]
    with pytest.raises(ValueError, match="outside segment"):
        select_review_panel(documents, results, qas_per_segment=1, seed=1)
