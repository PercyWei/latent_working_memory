import copy

import pytest

from latent_working_memory.data_preparation.fineweb_qa.assembly import (
    assemble_document,
    validate_trajectory,
)
from latent_working_memory.data_preparation.fineweb_qa.finalization import (
    apply_resolved_reviews,
    validate_resolved_reviews,
)


QA_CONFIG = {
    "role_seed": 20260928,
    "train_per_segment": [4] * 8,
    "gate_per_segment": [8, 4, 4, 4, 4, 4, 4, 0],
}


def example_document(extra_first_segment=0):
    segments, candidates, parts = [], [], []
    offset = 0
    for segment_index in range(8):
        segment_start = offset
        count = 12 if segment_index == 0 else 4 if segment_index == 7 else 8
        count += extra_first_segment if segment_index == 0 else 0
        for item_index in range(count):
            qa_id = f"q{segment_index}-{item_index}"
            answer = f"value-{segment_index}-{item_index}"
            line = f"The value for {qa_id} is {answer}.\n"
            answer_start = offset + line.index(answer)
            candidates.append(
                {
                    "qa_id": qa_id,
                    "segment_id": f"seg{segment_index}",
                    "fact_statement": f"{qa_id} has {answer}",
                    "question": f"What is the value for {qa_id}?",
                    "answer": answer,
                    "evidence_char_span": [offset, offset + len(line)],
                    "answer_char_span": [answer_start, answer_start + len(answer)],
                }
            )
            parts.append(line)
            offset += len(line)
        segments.append({"segment_id": f"seg{segment_index}", "char_span": [segment_start, offset]})
    document = {
        "trajectory_id": "trajectory-1",
        "document_id": "document-1",
        "dedup_cluster": "example.org/article",
        "split": "train",
        "source": {"file": "000_00000.parquet", "row_group": 0, "row_index": 0},
        "window_char_span": [10, 10 + offset],
        "text": "".join(parts),
        "segments": segments,
    }
    result = {
        "local_candidates": candidates,
        "review_decisions": [
            {"qa_id": qa["qa_id"], "accepted": True, "reason": "", "fact_group_id": qa["qa_id"]}
            for qa in candidates
        ],
    }
    return document, result


def reviewed(qa_id, accepted=True, reason="", same_fact_with=None):
    return {
        "qa_id": qa_id,
        "accepted": accepted,
        "reason": reason,
        "same_fact_with": [] if same_fact_with is None else same_fact_with,
        "evidence_prediction_correct": None,
    }


def test_reviewer_rejection_recomputes_quota_without_mutating_model_result():
    document, result = example_document()
    original = copy.deepcopy(result)
    panel = [{"qa_id": "q0-0"}]
    resolved = validate_resolved_reviews(
        panel, {"decisions": [reviewed("q0-0", accepted=False, reason="evidence is ambiguous")]}
    )
    final = apply_resolved_reviews(document, result, resolved, QA_CONFIG)
    assert not final["ok"]
    assert final["trajectory"] is None
    assert final["shortfalls"][0]["segment_id"] == "seg0"
    assert final["shortfalls"][0]["missing_total"] == 1
    assert final["statistics"]["review_rejected_count"] == 1
    assert result == original


def test_reviewer_merge_reallocates_cross_role_fact_before_final_assembly():
    document, result = example_document(extra_first_segment=1)
    initial = assemble_document(
        document, result["local_candidates"], result["review_decisions"], QA_CONFIG
    )
    first_segment = [qa for qa in initial["trajectory"]["qas"] if qa["segment_id"] == "seg0"]
    train_id = next(qa["qa_id"] for qa in first_segment if qa["role"] == "train")
    gate_id = next(qa["qa_id"] for qa in first_segment if qa["role"] == "gate")
    resolved = validate_resolved_reviews(
        [{"qa_id": gate_id}],
        {"decisions": [reviewed(gate_id, same_fact_with=[train_id])]},
    )
    final = apply_resolved_reviews(document, result, resolved, QA_CONFIG)
    assert final["ok"]
    assert final["statistics"]["duplicate_fact_count"] == 1
    trajectory = final["trajectory"]
    validate_trajectory(trajectory, QA_CONFIG)
    assert len(trajectory["qas"]) == 64
    assert len({qa["fact_group_id"] for qa in trajectory["qas"]}) == 64
    merged = [qa for qa in trajectory["qas"] if qa["qa_id"] in {train_id, gate_id}]
    assert len(merged) == 1
    candidate_order = {qa["qa_id"]: index for index, qa in enumerate(result["local_candidates"])}
    assert merged[0]["fact_group_id"] == min(
        (train_id, gate_id), key=lambda qa_id: candidate_order[qa_id]
    )
    unaffected = next(qa for qa in trajectory["qas"] if qa["qa_id"] not in {train_id, gate_id})
    assert unaffected["fact_group_id"] == unaffected["qa_id"]


@pytest.mark.parametrize(
    "decisions,error",
    [
        ([reviewed("q0-0")], "cover the panel"),
        ([reviewed("q0-0"), reviewed("q0-0")], "duplicate resolved review"),
        ([reviewed("q0-0"), reviewed("q9-9")], "cover the panel"),
        ([reviewed("q0-0", accepted=False)], "needs a reason"),
    ],
)
def test_resolved_review_requires_exact_panel_coverage(decisions, error):
    panel = [{"qa_id": "q0-0"}, {"qa_id": "q0-1"}]
    with pytest.raises(ValueError, match=error):
        validate_resolved_reviews(panel, {"decisions": decisions})


def test_cross_document_and_rejected_fact_merge_targets_fail():
    document, result = example_document()
    cross_document = validate_resolved_reviews(
        [{"qa_id": "q0-0"}],
        {"decisions": [reviewed("q0-0", same_fact_with=["another-document-qa"])]},
    )
    with pytest.raises(ValueError, match="not in this document"):
        apply_resolved_reviews(document, result, cross_document, QA_CONFIG)

    rejected_target = validate_resolved_reviews(
        [{"qa_id": "q0-0"}, {"qa_id": "q0-1"}],
        {
            "decisions": [
                reviewed("q0-0", same_fact_with=["q0-1"]),
                reviewed("q0-1", accepted=False, reason="unsupported"),
            ]
        },
    )
    with pytest.raises(ValueError, match="not retained"):
        apply_resolved_reviews(document, result, rejected_target, QA_CONFIG)


def test_panel_cannot_restore_model_rejected_candidate():
    document, result = example_document()
    result["review_decisions"][0].update(accepted=False, reason="model rejected", fact_group_id="")
    resolved = validate_resolved_reviews([{"qa_id": "q0-0"}], {"decisions": [reviewed("q0-0")]})
    final = apply_resolved_reviews(document, result, resolved, QA_CONFIG)
    assert not final["ok"]
    assert "q0-0" not in {qa["qa_id"] for qa in final["eligible_qas"]}


def test_historical_review_links_survive_later_model_rejection_of_a_bridge():
    document, result = example_document(extra_first_segment=2)
    resolved = validate_resolved_reviews(
        [{"qa_id": "q0-0"}, {"qa_id": "q0-1"}],
        {
            "decisions": [
                reviewed("q0-0", same_fact_with=["q0-1"]),
                reviewed("q0-1", same_fact_with=["q0-2"]),
            ]
        },
    )
    result["review_decisions"][1].update(accepted=False, reason="later rejection", fact_group_id="")
    final = apply_resolved_reviews(document, result, resolved, QA_CONFIG)
    assert final["ok"]
    remaining = {q["qa_id"] for q in final["eligible_qas"]}
    assert "q0-1" not in remaining
    assert len(remaining & {"q0-0", "q0-2"}) == 1
