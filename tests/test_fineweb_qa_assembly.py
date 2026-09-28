import copy
import json
from pathlib import Path

import pytest

from latent_working_memory.data_preparation.fineweb_qa.assembly import (
    assemble_trajectory,
    validate_trajectory,
)


@pytest.fixture
def assembly_input():
    config_path = Path(__file__).resolve().parents[1] / (
        "configs/data_preparation/fineweb-factqa-8192-doc2k.json"
    )
    config = json.loads(config_path.read_text())
    text = ""
    segments = []
    qas = []
    for segment_id, count in enumerate(config["annotation"]["candidate_counts"], 1):
        start = len(text)
        for number in range(1, count + 1):
            answer = f"value-{segment_id}-{number}"
            quote = f"Fact {segment_id}-{number} has {answer}."
            evidence_start = len(text)
            text += quote + "\n"
            answer_start = evidence_start + quote.index(answer)
            qas.append(
                {
                    "qa_id": f"trajectory-1:s{segment_id}:q{number}",
                    "segment_id": segment_id,
                    "fact_statement": quote,
                    "question": f"What value belongs to fact {segment_id}-{number}?",
                    "answer": answer,
                    "evidence_quote": quote,
                    "evidence_span": [evidence_start, evidence_start + len(quote)],
                    "answer_span": [answer_start, answer_start + len(answer)],
                }
            )
        text += " " * (config["text"]["target_segment_chars"] - (len(text) - start))
        segments.append({"segment_id": segment_id, "char_start": start, "char_end": len(text)})
    trajectory = {
        "trajectory_id": "trajectory-1",
        "document_id": "document-1",
        "dedup_cluster": "cluster-1",
        "split": "train",
        "source": {"path": "local.parquet", "row_index": 7, "char_start": 0, "char_end": len(text)},
        "text": text,
        "segments": segments,
    }
    review = {
        "decisions": [
            {"qa_id": qa["qa_id"], "accepted": True, "reason": "supported"} for qa in qas
        ],
        "same_fact_groups": [],
    }
    return trajectory, qas, review, config


def test_complete_trajectory_preserves_evidence_and_builds_all_seven_updates(assembly_input):
    trajectory, qas, review, config = assembly_input
    before = copy.deepcopy(assembly_input)
    result = assemble_trajectory(trajectory, qas, review, config)
    record = result["trajectory"]
    assert result["complete"]
    assert len(result["accepted_qas"]) == 80
    assert len(result["selected_qas"]) == len(record["qas"]) == 64
    assert result["removals"] == []
    assert all(item["shortfall"] == 0 for item in result["per_segment"])
    assert record["text"] == trajectory["text"]
    assert record["source"] == trajectory["source"]
    assert len({qa["fact_group"] for qa in record["qas"]}) == 64
    assert all("evidence_quote" not in qa for qa in record["qas"])
    assert record["evaluation_qa_ids"] == []
    by_id = {qa["qa_id"]: qa for qa in record["qas"]}
    assert (
        sum(len(step["new_qa_ids"]) + len(step["old_qa_ids"]) for step in record["schedule"]) == 60
    )
    assert sum(len(step["gate_qa_ids"]) for step in record["schedule"]) * 3 == 168
    assert record["schedule"][0]["gate_qa_ids"] == []
    assert {by_id[qa_id]["segment_id"] for qa_id in record["schedule"][1]["gate_qa_ids"]} == {1}
    for step in record["schedule"][2:]:
        current = step["segment_id"]
        all_ids = step["new_qa_ids"] + step["old_qa_ids"] + step["gate_qa_ids"]
        assert len(set(all_ids)) == 16
        assert sum(by_id[qa_id]["segment_id"] == current - 1 for qa_id in step["old_qa_ids"]) == 2
        assert sum(by_id[qa_id]["segment_id"] == current - 1 for qa_id in step["gate_qa_ids"]) == 4
        assert all(
            by_id[qa_id]["segment_id"] < current
            for qa_id in step["old_qa_ids"] + step["gate_qa_ids"]
        )
    early_at_third = set(record["schedule"][2]["gate_qa_ids"][4:])
    early_at_fourth = set(record["schedule"][3]["gate_qa_ids"][4:])
    assert not early_at_third.intersection(early_at_fourth)
    assert assembly_input == before


def test_same_fact_uses_earliest_accepted_segment_and_numeric_candidate_order(assembly_input):
    trajectory, qas, review, config = assembly_input
    first, second, tenth, later = qas[0], qas[1], qas[9], qas[15]
    review["same_fact_groups"] = [[later["qa_id"], tenth["qa_id"], first["qa_id"], second["qa_id"]]]
    review["decisions"][0].update(accepted=False, reason="ambiguous")
    result = assemble_trajectory(trajectory, list(reversed(qas)), review, config)
    by_id = {qa["qa_id"]: qa for qa in result["accepted_qas"]}
    assert second["qa_id"] in by_id
    assert by_id[second["qa_id"]]["fact_group"] == second["qa_id"]
    assert all(qa["qa_id"] not in by_id for qa in (first, tenth, later))
    removed = {item["qa_id"]: item for item in result["removals"]}
    assert removed[first["qa_id"]]["stage"] == "document_review"
    assert removed[first["qa_id"]]["reason"] == "ambiguous"
    assert removed[tenth["qa_id"]]["representative_qa_id"] == second["qa_id"]
    assert removed[later["qa_id"]]["representative_qa_id"] == second["qa_id"]


def test_selection_and_schedule_do_not_depend_on_review_or_input_order(assembly_input):
    trajectory, qas, review, config = assembly_input
    expected = assemble_trajectory(trajectory, qas, review, config)
    review["decisions"].reverse()
    actual = assemble_trajectory(trajectory, list(reversed(qas)), review, config)
    assert actual == expected


def test_shortfall_keeps_diagnostic_candidates_but_never_emits_partial_trajectory(assembly_input):
    trajectory, qas, review, config = assembly_input
    for decision in review["decisions"][:4]:
        decision.update(accepted=False, reason="unsupported")
    result = assemble_trajectory(trajectory, qas, review, config)
    assert not result["complete"]
    assert result["trajectory"] is None
    assert len(result["accepted_qas"]) == 76
    assert len(result["selected_qas"]) == 63
    assert result["per_segment"][0] == {
        "segment_id": 1,
        "accepted_count": 11,
        "required_count": 12,
        "selected_count": 11,
        "shortfall": 1,
    }
    assert len(result["removals"]) == 4


def test_empty_review_results_are_reported_as_shortfalls(assembly_input):
    trajectory, _, _, config = assembly_input
    result = assemble_trajectory(trajectory, [], {"decisions": [], "same_fact_groups": []}, config)
    assert not result["complete"]
    assert result["trajectory"] is None
    assert result["accepted_qas"] == result["selected_qas"] == []
    assert sum(item["shortfall"] for item in result["per_segment"]) == 64


@pytest.mark.parametrize("split", ["dev", "test"])
def test_evaluation_uses_task_pool_without_training_roles(assembly_input, split):
    trajectory, qas, review, config = assembly_input
    trajectory["split"] = split
    record = assemble_trajectory(trajectory, qas, review, config)["trajectory"]
    assert len(record["evaluation_qa_ids"]) == 32
    assert {qa["role"] for qa in record["qas"]} == {"evaluation", "gate"}
    assert set(record["evaluation_qa_ids"]) == {
        qa_id for step in record["schedule"] for qa_id in step["new_qa_ids"]
    }


@pytest.mark.parametrize(
    "error",
    ["missing", "duplicate", "unknown", "non_boolean", "overlap", "group_unknown", "singleton"],
)
def test_malformed_full_document_review_is_a_contract_error(assembly_input, error):
    trajectory, qas, review, config = assembly_input
    first, second, third = [qa["qa_id"] for qa in qas[:3]]
    if error == "missing":
        review["decisions"].pop()
    elif error == "duplicate":
        review["decisions"].append(copy.deepcopy(review["decisions"][0]))
    elif error == "unknown":
        review["decisions"][0]["qa_id"] = "not-a-candidate"
    elif error == "non_boolean":
        review["decisions"][0]["accepted"] = "true"
    elif error == "overlap":
        review["same_fact_groups"] = [[first, second], [second, third]]
    elif error == "group_unknown":
        review["same_fact_groups"] = [[first, "not-a-candidate"]]
    else:
        review["same_fact_groups"] = [[first]]
    with pytest.raises(ValueError):
        assemble_trajectory(trajectory, qas, review, config)


@pytest.mark.parametrize(
    "error",
    [
        "future",
        "duplicate",
        "fact_collision",
        "answer_span",
        "evidence_segment",
        "gap",
        "quota",
        "evaluation_list",
    ],
)
def test_persisted_trajectory_validation_rejects_corrupted_records(assembly_input, error):
    trajectory, qas, review, config = assembly_input
    record = assemble_trajectory(trajectory, qas, review, config)["trajectory"]
    if error == "future":
        future_gate = next(
            qa for qa in record["qas"] if qa["segment_id"] == 2 and qa["role"] == "gate"
        )
        record["schedule"][1]["gate_qa_ids"][0] = future_gate["qa_id"]
    elif error == "duplicate":
        record["schedule"][1]["gate_qa_ids"][1] = record["schedule"][1]["gate_qa_ids"][0]
    elif error == "fact_collision":
        record["qas"][1]["fact_group"] = record["qas"][0]["fact_group"]
    elif error == "answer_span":
        record["qas"][0]["answer_span"][0] += 1
    elif error == "evidence_segment":
        record["qas"][0]["evidence_span"][1] = record["segments"][1]["char_end"]
    elif error == "gap":
        record["segments"][1]["char_start"] += 1
    elif error == "quota":
        record["qas"].pop()
    else:
        record["evaluation_qa_ids"] = [record["qas"][0]["qa_id"]]
    with pytest.raises(ValueError):
        validate_trajectory(record, config)
