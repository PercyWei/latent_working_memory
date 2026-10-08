import copy

import pytest

from latent_working_memory.data_preparation.fineweb_factqa.assembly import (
    assemble_document,
    validate_trajectory,
    qa_quotas,
)


QA_CONFIG = {
    "role_seed": 20260928,
    "max_answer_chars": 128,
}


def example_document(
    extra_per_segment=None, repeat_fact=False, segment_count=8, split="train", answer_suffix=""
):
    extras = extra_per_segment or {}
    text_parts, segments, candidates = [], [], []
    offset = 0
    for segment_index in range(segment_count):
        segment_id = f"seg{segment_index}"
        lines = []
        _, tasks, gates = qa_quotas(segment_count)
        target_count = tasks[segment_index] + gates[segment_index]
        for item_index in range(target_count + extras.get(segment_index, 0)):
            qa_id = f"q{segment_index}-{item_index}"
            fact_name = f"{segment_index}-{item_index}"
            if repeat_fact and (segment_index, item_index) == (1, 0):
                fact_name = "0-1"
            answer = f"value-{fact_name}{answer_suffix}"
            line = f"Fact {fact_name} has {answer}.\n"
            evidence_start = offset + sum(len(part) for part in lines)
            answer_start = evidence_start + line.index(answer)
            candidates.append(
                {
                    "qa_id": qa_id,
                    "segment_id": segment_id,
                    "fact_statement": f"Fact {fact_name} has {answer}",
                    "question": f"What does fact {fact_name} have?",
                    "answer": answer,
                    "evidence_char_span": [evidence_start, evidence_start + len(line)],
                    "answer_char_span": [answer_start, answer_start + len(answer)],
                }
            )
            lines.append(line)
        segment_text = "".join(lines)
        text_parts.append(segment_text)
        segments.append(
            {"segment_id": segment_id, "char_span": [offset, offset + len(segment_text)]}
        )
        offset += len(segment_text)
    text = "".join(text_parts)
    document = {
        "trajectory_id": "trajectory-1",
        "document_id": "document-1",
        "dedup_cluster": "example.org/article",
        "split": split,
        "source": {"file": "sample/000_00000.parquet", "row_group": 2, "row_index": 7},
        "window_char_span": [100, 100 + len(text)],
        "text": text,
        "segments": segments,
    }
    decisions = [
        {"qa_id": qa["qa_id"], "accepted": True, "reason": "", "fact_group_id": qa["qa_id"]}
        for qa in candidates
    ]
    return document, candidates, decisions


def test_complete_trajectory_has_fixed_quota_and_first_gate():
    document, candidates, decisions = example_document()
    result = assemble_document(document, candidates, decisions, QA_CONFIG)
    assert result["ok"] and not result["shortfalls"]
    trajectory = result["trajectory"]
    assert len(trajectory["qas"]) == 64
    assert sum(qa["role"] == "train" for qa in trajectory["qas"]) == 32
    assert sum(qa["role"] == "gate" for qa in trajectory["qas"]) == 32
    assert trajectory["estimated_tokens"] == len(document["text"]) / 4
    assert trajectory["text_char_length"] == len(document["text"])
    assert all("evidence_quote" not in qa for qa in trajectory["qas"])

    usage = trajectory["usage"]
    by_id = {qa["qa_id"]: qa for qa in trajectory["qas"]}
    assert usage[0]["gate_qa_ids"] == []
    assert usage[0]["task_old_qa_ids"] == []
    assert len(usage[0]["task_new_qa_ids"]) == 4
    assert len(usage[1]["gate_qa_ids"]) == 8
    assert {by_id[qa_id]["segment_id"] for qa_id in usage[1]["gate_qa_ids"]} == {"seg0"}
    assert len(usage[1]["task_old_qa_ids"]) == 4
    for step, row in enumerate(usage):
        assert len(row["task_new_qa_ids"]) == 4
        assert len(row["task_old_qa_ids"]) == (0 if step == 0 else 4)
        assert len(row["gate_qa_ids"]) == (0 if step == 0 else 8)
        all_ids = row["task_new_qa_ids"] + row["task_old_qa_ids"] + row["gate_qa_ids"]
        assert len(all_ids) == len(set(all_ids))
        assert all(by_id[qa_id]["segment_id"] == f"seg{step}" for qa_id in row["task_new_qa_ids"])
        assert all(
            int(by_id[qa_id]["segment_id"][3:]) < step
            for qa_id in row["task_old_qa_ids"] + row["gate_qa_ids"]
        )
    validate_trajectory(trajectory, QA_CONFIG)


def test_review_fact_group_uses_earliest_complete_segment_then_candidate_order():
    document, candidates, decisions = example_document({1: 1}, repeat_fact=True)
    for decision in decisions:
        if decision["qa_id"] in {"q0-1", "q1-0"}:
            decision["fact_group_id"] = "repeated-fact"
    repeated_early = copy.deepcopy(candidates[0])
    repeated_early["qa_id"] = "q0-0-reworded"
    repeated_early["question"] = "Which value belongs to fact 0-0?"
    candidates.append(repeated_early)
    decisions.append(
        {
            "qa_id": repeated_early["qa_id"],
            "accepted": True,
            "reason": "",
            "fact_group_id": "q0-0",
        }
    )

    result = assemble_document(document, candidates, decisions, QA_CONFIG)
    assert result["ok"]
    by_group = {qa["fact_group_id"]: qa for qa in result["eligible_qas"]}
    assert by_group["q0-0"]["qa_id"] == "q0-0"
    assert by_group["repeated-fact"]["qa_id"] == "q0-1"
    assert by_group["repeated-fact"]["segment_id"] == "seg0"
    assert result["statistics"]["duplicate_fact_count"] == 2
    assert len({qa["fact_group_id"] for qa in result["trajectory"]["qas"]}) == 64


def test_segment_shortfall_does_not_backfill_from_another_segment():
    document, candidates, decisions = example_document({1: 1})
    candidates = [qa for qa in candidates if qa["qa_id"] != "q0-0"]
    decisions = [decision for decision in decisions if decision["qa_id"] != "q0-0"]
    result = assemble_document(document, candidates, decisions, QA_CONFIG)
    assert not result["ok"]
    assert result["trajectory"] is None
    assert result["shortfalls"] == [
        {
            "segment_id": "seg0",
            "available": 11,
            "required_task": 4,
            "required_gate": 8,
            "task_missing": 0,
            "gate_missing": 1,
            "missing_total": 1,
        }
    ]
    assert len(result["eligible_qas"]) == 64
    assert result["statistics"]["eligible_by_segment"]["seg1"] == 9


def test_review_must_cover_every_qa_and_accepted_facts_need_groups():
    document, candidates, decisions = example_document()
    with pytest.raises(ValueError, match="cover every candidate"):
        assemble_document(document, candidates, decisions[:-1], QA_CONFIG)
    decisions[0]["fact_group_id"] = ""
    with pytest.raises(ValueError, match="fact_group_id"):
        assemble_document(document, candidates, decisions, QA_CONFIG)


@pytest.mark.parametrize(
    "change,error",
    [
        (
            lambda trajectory: trajectory["usage"][1]["gate_qa_ids"].__setitem__(
                0, trajectory["usage"][2]["task_new_qa_ids"][0]
            ),
            "usage schedule",
        ),
        (
            lambda trajectory: trajectory["qas"][-1].__setitem__(
                "fact_group_id", trajectory["qas"][0]["fact_group_id"]
            ),
            "fact group",
        ),
        (
            lambda trajectory: trajectory["qas"][0].__setitem__("answer_char_span", [0, 1]),
            "answer is outside evidence|answer span does not match",
        ),
        (
            lambda trajectory: trajectory.__setitem__("estimated_tokens", 1),
            "estimated_tokens",
        ),
    ],
)
def test_validation_rejects_schedule_fact_and_offset_corruption(change, error):
    document, candidates, decisions = example_document()
    trajectory = assemble_document(document, candidates, decisions, QA_CONFIG)["trajectory"]
    change(trajectory)
    with pytest.raises(ValueError, match=error):
        validate_trajectory(trajectory, QA_CONFIG)


def test_role_order_is_stable_when_candidate_listing_is_reordered():
    document, candidates, decisions = example_document({0: 2, 3: 2})
    first = assemble_document(document, candidates, decisions, QA_CONFIG)
    second = assemble_document(document, list(reversed(candidates)), decisions, QA_CONFIG)
    assert first["trajectory"]["qas"] == second["trajectory"]["qas"]
    assert first["trajectory"]["usage"] == second["trajectory"]["usage"]


@pytest.mark.parametrize("answer_suffix,limit", [("", 16), ("x" * 140, 192)])
def test_answer_limit_applies_to_assembly_and_final_validation(answer_suffix, limit):
    document, candidates, decisions = example_document(answer_suffix=answer_suffix)
    config = dict(QA_CONFIG, max_answer_chars=limit)
    trajectory = assemble_document(document, candidates, decisions, config)["trajectory"]
    validate_trajectory(trajectory, config)
    maximum = max(len(qa["answer"]) for qa in trajectory["qas"])
    if answer_suffix:
        assert maximum > 128
    smaller = dict(config, max_answer_chars=maximum - 1)
    with pytest.raises(ValueError, match=f"answer exceeds {maximum - 1} characters"):
        assemble_document(document, candidates, decisions, smaller)
    with pytest.raises(ValueError, match=f"answer exceeds {maximum - 1} characters"):
        validate_trajectory(trajectory, smaller)


def test_config_and_segment_contract_is_fixed():
    document, candidates, decisions = example_document()
    document["segments"][3]["char_span"][0] += 1
    with pytest.raises(ValueError, match="consecutively cover"):
        assemble_document(document, candidates, decisions, QA_CONFIG)


@pytest.mark.parametrize("segment_count", range(6, 11))
@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_variable_length_roles_and_all_prefix_evaluation(segment_count, split):
    document, candidates, decisions = example_document(segment_count=segment_count, split=split)
    trajectory = assemble_document(document, candidates, decisions, dict(QA_CONFIG, role_seed=17))[
        "trajectory"
    ]
    validate_trajectory(trajectory, dict(QA_CONFIG, role_seed=17))
    assert len(trajectory["qas"]) == 8 * segment_count
    role = "train" if split == "train" else "evaluation"
    assert sum(q["role"] == role for q in trajectory["qas"]) == 4 * segment_count
    assert sum(q["role"] == "gate" for q in trajectory["qas"]) == 4 * segment_count
    by_id = {q["qa_id"]: q for q in trajectory["qas"]}
    for i, step in enumerate(trajectory["usage"]):
        assert len(step["gate_qa_ids"]) == (8 if i else 0)
        assert len(step["task_new_qa_ids"]) == 4
        assert len(step["task_old_qa_ids"]) == ((4 if i else 0) if split == "train" else i * 4)
        assert all(
            int(by_id[q]["segment_id"][3:]) < i
            for q in step["gate_qa_ids"] + step["task_old_qa_ids"]
        )
    if split != "train":
        last = trajectory["usage"][-1]
        assert len(last["task_new_qa_ids"] + last["task_old_qa_ids"]) == 4 * segment_count
