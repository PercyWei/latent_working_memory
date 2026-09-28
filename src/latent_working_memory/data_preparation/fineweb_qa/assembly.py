"""将已核验的 FineWeb QA 组装为固定八段轨迹。"""

from collections import Counter
from hashlib import sha256
import json
import re


def _candidate_order(qa: dict) -> tuple:
    return qa["segment_id"], tuple(int(part) for part in re.findall(r"\d+", qa["qa_id"]))


def _selection_key(seed: int, trajectory_id: str, segment_id: int, qa_id: str) -> str:
    value = json.dumps([seed, trajectory_id, segment_id, qa_id], ensure_ascii=False)
    return sha256(value.encode("utf-8")).hexdigest()


def _review_candidates(accepted_qas: list[dict], review: dict) -> tuple[list[dict], list[dict]]:
    by_id = {qa["qa_id"]: qa for qa in accepted_qas}
    if len(by_id) != len(accepted_qas):
        raise ValueError("duplicate candidate qa_id")
    decisions = review["decisions"]
    decision_ids = [item["qa_id"] for item in decisions]
    if len(set(decision_ids)) != len(decision_ids) or set(decision_ids) != set(by_id):
        raise ValueError("review decisions must contain every candidate qa_id exactly once")
    if any(type(item["accepted"]) is not bool for item in decisions):
        raise ValueError("review accepted values must be booleans")

    groups = []
    grouped = set()
    for group in review["same_fact_groups"]:
        if len(group) < 2 or len(set(group)) != len(group):
            raise ValueError("same-fact groups must contain at least two distinct qa_ids")
        if not set(group) <= set(by_id):
            raise ValueError("same-fact group contains an unknown qa_id")
        if grouped.intersection(group):
            raise ValueError("same-fact groups must be disjoint")
        groups.append(group)
        grouped.update(group)
    groups.extend([qa_id] for qa_id in by_id if qa_id not in grouped)

    decision_by_id = {item["qa_id"]: item for item in decisions}
    removals = [
        {
            "qa_id": item["qa_id"],
            "stage": "document_review",
            "reason": item["reason"],
            "representative_qa_id": None,
        }
        for item in decisions
        if not item["accepted"]
    ]
    representatives = []
    for group in groups:
        eligible = sorted(
            (by_id[qa_id] for qa_id in group if decision_by_id[qa_id]["accepted"]),
            key=_candidate_order,
        )
        if not eligible:
            continue
        representative = eligible[0]
        representatives.append({**representative, "fact_group": representative["qa_id"]})
        for duplicate in eligible[1:]:
            removals.append(
                {
                    "qa_id": duplicate["qa_id"],
                    "stage": "same_fact",
                    "reason": "same factual claim as the retained representative",
                    "representative_qa_id": representative["qa_id"],
                }
            )
    representatives.sort(key=_candidate_order)
    removals.sort(key=lambda item: _candidate_order(by_id[item["qa_id"]]))
    return representatives, removals


def _round_robin(
    pools: dict[int, list[str]],
    cursors: dict[int, int],
    segment_ids: list[int],
    count: int,
    start: int,
) -> tuple[list[str], int]:
    selected = []
    for offset in range(count):
        segment_id = segment_ids[(start + offset) % len(segment_ids)]
        pool = pools[segment_id]
        selected.append(pool[cursors[segment_id] % len(pool)])
        cursors[segment_id] += 1
    return selected, (start + count) % len(segment_ids)


def _build_schedule(qas: list[dict]) -> list[dict]:
    task_pools = {segment_id: [] for segment_id in range(1, 9)}
    gate_pools = {segment_id: [] for segment_id in range(1, 9)}
    for qa in qas:
        pools = gate_pools if qa["role"] == "gate" else task_pools
        pools[qa["segment_id"]].append(qa["qa_id"])
    task_cursors = dict.fromkeys(task_pools, 0)
    gate_cursors = dict.fromkeys(gate_pools, 0)
    task_start = gate_start = 0
    schedule = []
    for segment_id in range(1, 9):
        old_ids = []
        gate_ids = []
        if segment_id == 2:
            old_ids, _ = _round_robin(task_pools, task_cursors, [1], 4, 0)
            gate_ids, _ = _round_robin(gate_pools, gate_cursors, [1], 8, 0)
        elif segment_id > 2:
            previous = [segment_id - 1]
            earlier = list(range(1, segment_id - 1))
            old_ids, _ = _round_robin(task_pools, task_cursors, previous, 2, 0)
            older_ids, task_start = _round_robin(task_pools, task_cursors, earlier, 2, task_start)
            old_ids.extend(older_ids)
            gate_ids, _ = _round_robin(gate_pools, gate_cursors, previous, 4, 0)
            earlier_ids, gate_start = _round_robin(gate_pools, gate_cursors, earlier, 4, gate_start)
            gate_ids.extend(earlier_ids)
        schedule.append(
            {
                "segment_id": segment_id,
                "new_qa_ids": list(task_pools[segment_id]),
                "old_qa_ids": old_ids,
                "gate_qa_ids": gate_ids,
            }
        )
    return schedule


def assemble_trajectory(
    trajectory: dict, accepted_qas: list[dict], review: dict, config: dict
) -> dict:
    """返回事实代表、配额诊断及完整轨迹；缺额时不输出正式轨迹。"""
    representatives, removals = _review_candidates(accepted_qas, review)
    final_counts = config["annotation"]["final_counts"]
    gate_counts = config["annotation"]["gate_counts"]
    seed = config["source"]["seed"]
    task_role = "train" if trajectory["split"] == "train" else "evaluation"
    selected = []
    per_segment = []
    for segment in trajectory["segments"]:
        segment_id = segment["segment_id"]
        available = sorted(
            (qa for qa in representatives if qa["segment_id"] == segment_id),
            key=lambda qa: _selection_key(
                seed, trajectory["trajectory_id"], segment_id, qa["qa_id"]
            ),
        )
        required = final_counts[segment_id - 1]
        chosen = available[:required]
        for index, qa in enumerate(chosen):
            final_qa = {key: value for key, value in qa.items() if key != "evidence_quote"}
            final_qa["role"] = "gate" if index < gate_counts[segment_id - 1] else task_role
            selected.append(final_qa)
        per_segment.append(
            {
                "segment_id": segment_id,
                "accepted_count": len(available),
                "required_count": required,
                "selected_count": len(chosen),
                "shortfall": max(0, required - len(available)),
            }
        )
    complete = not any(item["shortfall"] for item in per_segment)
    record = None
    if complete:
        record = {
            **trajectory,
            "qas": selected,
            "schedule": _build_schedule(selected),
            "evaluation_qa_ids": [qa["qa_id"] for qa in selected if qa["role"] == "evaluation"],
        }
        validate_trajectory(record, config)
    return {
        "complete": complete,
        "trajectory": record,
        "accepted_qas": representatives,
        "selected_qas": selected,
        "per_segment": per_segment,
        "removals": removals,
    }


def validate_trajectory(trajectory: dict, config: dict) -> None:
    """检查最终 JSONL 记录的文本、证据、事实分池与更新时间契约。"""
    text_config = config["text"]
    final_counts = config["annotation"]["final_counts"]
    gate_counts = config["annotation"]["gate_counts"]
    if (
        text_config["segments"] != 8
        or gate_counts != [8, 4, 4, 4, 4, 4, 4, 0]
        or final_counts != [12, 8, 8, 8, 8, 8, 8, 4]
    ):
        raise ValueError("configuration must specify the agreed eight-segment QA quotas")
    if trajectory["split"] not in {"train", "dev", "test"}:
        raise ValueError("trajectory split must be train, dev or test")
    for key in ("trajectory_id", "document_id", "dedup_cluster", "source"):
        if key not in trajectory:
            raise ValueError(f"trajectory is missing {key}")
    text = trajectory["text"]
    if not text_config["min_trajectory_chars"] <= len(text) <= text_config["max_trajectory_chars"]:
        raise ValueError("trajectory character length is outside the configured range")
    segments = trajectory["segments"]
    if [segment["segment_id"] for segment in segments] != list(range(1, 9)):
        raise ValueError("segments must be ordered from 1 to 8")
    previous_end = 0
    for segment in segments:
        start, end = segment["char_start"], segment["char_end"]
        if type(start) is not int or type(end) is not int or start != previous_end:
            raise ValueError("segment character spans must be contiguous integer boundaries")
        if not text_config["min_segment_chars"] <= end - start <= text_config["max_segment_chars"]:
            raise ValueError("segment character length is outside the configured range")
        previous_end = end
    if previous_end != len(text):
        raise ValueError("segments must cover the complete trajectory text")

    qas = trajectory["qas"]
    by_id = {qa["qa_id"]: qa for qa in qas}
    if len(by_id) != len(qas):
        raise ValueError("duplicate final qa_id")
    if len({qa["fact_group"] for qa in qas}) != len(qas):
        raise ValueError("a fact group may contribute only one final QA")
    task_role = "train" if trajectory["split"] == "train" else "evaluation"
    counts = Counter()
    for qa in qas:
        segment_id = qa["segment_id"]
        if type(segment_id) is not int or not 1 <= segment_id <= 8:
            raise ValueError("QA segment_id must identify an existing segment")
        if qa["fact_group"] != qa["qa_id"]:
            raise ValueError("fact_group must be the retained representative qa_id")
        if qa["role"] not in {task_role, "gate"}:
            raise ValueError("QA role is incompatible with the trajectory split")
        if "evidence_quote" in qa or "token_ids" in qa:
            raise ValueError(
                "final QA must store character spans without evidence text or token IDs"
            )
        if any(
            not isinstance(qa[key], str) or not qa[key].strip()
            for key in ("fact_statement", "question", "answer")
        ):
            raise ValueError("fact statement, question and answer must be nonempty strings")
        if len(qa["answer"]) > config["annotation"]["max_answer_chars"]:
            raise ValueError("answer exceeds the character limit")
        evidence_start, evidence_end = qa["evidence_span"]
        answer_start, answer_end = qa["answer_span"]
        if any(
            type(value) is not int
            for value in (evidence_start, evidence_end, answer_start, answer_end)
        ):
            raise ValueError("evidence and answer spans must use integer character offsets")
        segment = segments[segment_id - 1]
        if not (
            segment["char_start"]
            <= evidence_start
            <= answer_start
            < answer_end
            <= evidence_end
            <= segment["char_end"]
        ):
            raise ValueError("QA evidence and answer must be contained in the assigned segment")
        if text[answer_start:answer_end] != qa["answer"]:
            raise ValueError("answer does not match its original text span")
        counts[segment_id, qa["role"]] += 1
    for segment_id in range(1, 9):
        if (
            counts[segment_id, task_role] != 4
            or counts[segment_id, "gate"] != gate_counts[segment_id - 1]
        ):
            raise ValueError("final QA counts do not satisfy the per-segment role quotas")

    schedule = trajectory["schedule"]
    if [step["segment_id"] for step in schedule] != list(range(1, 9)):
        raise ValueError("schedule must contain the initial write and all seven updates")
    for step in schedule:
        current = step["segment_id"]
        new_ids, old_ids, gate_ids = (
            step[key] for key in ("new_qa_ids", "old_qa_ids", "gate_qa_ids")
        )
        expected_lengths = (4, 0, 0) if current == 1 else (4, 4, 8)
        if tuple(map(len, (new_ids, old_ids, gate_ids))) != expected_lengths:
            raise ValueError("schedule does not satisfy the per-step QA counts")
        all_ids = new_ids + old_ids + gate_ids
        if len(set(all_ids)) != len(all_ids) or not set(all_ids) <= set(by_id):
            raise ValueError("schedule contains duplicate or unknown QA IDs")
        for qa_id in new_ids:
            qa = by_id[qa_id]
            if qa["role"] != task_role or qa["segment_id"] != current:
                raise ValueError("new-task QA must belong to the current segment")
        for qa_id in old_ids + gate_ids:
            if by_id[qa_id]["segment_id"] >= current:
                raise ValueError("history QA contains current or future evidence")
        if any(by_id[qa_id]["role"] != task_role for qa_id in old_ids):
            raise ValueError("old-task QA must come from the task pool")
        if any(by_id[qa_id]["role"] != "gate" for qa_id in gate_ids):
            raise ValueError("gate QA must come from the gate pool")
        if current > 2:
            if sum(by_id[qa_id]["segment_id"] == current - 1 for qa_id in old_ids) != 2:
                raise ValueError("old-task QA must include two questions from the previous segment")
            if sum(by_id[qa_id]["segment_id"] == current - 1 for qa_id in gate_ids) != 4:
                raise ValueError("gate QA must include four questions from the previous segment")
    expected_evaluation = [qa["qa_id"] for qa in qas if qa["role"] == "evaluation"]
    if trajectory["evaluation_qa_ids"] != expected_evaluation:
        raise ValueError("evaluation QA list must contain all and only the evaluation pool")
