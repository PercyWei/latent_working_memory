"""Apply document review and assemble variable-length FineWeb FactQA trajectories."""

from __future__ import annotations

import hashlib
import json

from latent_working_memory.data_preparation.segmentation import (
    TOKEN_ESTIMATION_RULE,
    _nonempty,
    _nonnegative_integer,
    _span,
    validate_text_layout,
)


QA_FIELDS = (
    "qa_id",
    "segment_id",
    "fact_group_id",
    "fact_statement",
    "question",
    "answer",
    "evidence_char_span",
    "answer_char_span",
)


def qa_quotas(segment_count: int) -> tuple[list[int], list[int], list[int]]:
    """Candidate, task and gate counts: 10N, 4N and 4N in total."""
    if type(segment_count) is not int or segment_count < 2:
        raise ValueError("a trajectory requires at least two segments")
    return (
        [15] + [10] * (segment_count - 2) + [5],
        [4] * segment_count,
        [8] + [4] * (segment_count - 2) + [0],
    )


def _candidate_qa(
    candidate: dict, text: str, layout: dict[str, tuple[int, int]], max_answer_chars: int
) -> dict:
    qa_id = _nonempty(candidate["qa_id"], "qa_id")
    segment_id = candidate["segment_id"]
    if segment_id not in layout:
        raise ValueError(f"unknown segment for {qa_id}: {segment_id}")
    fact_statement = _nonempty(candidate["fact_statement"], f"{qa_id}.fact_statement")
    question = _nonempty(candidate["question"], f"{qa_id}.question")
    answer = _nonempty(candidate["answer"], f"{qa_id}.answer")
    if len(answer) > max_answer_chars:
        raise ValueError(f"answer exceeds {max_answer_chars} characters for {qa_id}")
    evidence_start, evidence_end = _span(
        candidate["evidence_char_span"], f"{qa_id}.evidence_char_span", len(text)
    )
    answer_start, answer_end = _span(
        candidate["answer_char_span"], f"{qa_id}.answer_char_span", len(text)
    )
    segment_start, segment_end = layout[segment_id]
    if not segment_start <= evidence_start < evidence_end <= segment_end:
        raise ValueError(f"evidence is outside {segment_id} for {qa_id}")
    if not evidence_start <= answer_start < answer_end <= evidence_end:
        raise ValueError(f"answer is outside evidence for {qa_id}")
    if text[answer_start:answer_end] != answer:
        raise ValueError(f"answer span does not match answer for {qa_id}")
    evidence = text[evidence_start:evidence_end]
    segment_text = text[segment_start:segment_end]
    quote_start = segment_text.find(evidence)
    if (
        quote_start != evidence_start - segment_start
        or segment_text.find(evidence, quote_start + 1) >= 0
    ):
        raise ValueError(f"evidence quote is not unique within {segment_id} for {qa_id}")
    return {
        "qa_id": qa_id,
        "segment_id": segment_id,
        "fact_statement": fact_statement,
        "question": question,
        "answer": answer,
        "evidence_char_span": [evidence_start, evidence_end],
        "answer_char_span": [answer_start, answer_end],
    }


def _role_rank(seed: int, document_id: str, segment_id: str, qa_id: str) -> bytes:
    identity = json.dumps(
        [seed, document_id, segment_id, qa_id], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(identity.encode()).digest()


def _older_ids(pools: list[list[str]], step: int, count: int, state: dict) -> list[str]:
    """Rotate source segments, then rotate each source's fixed QA order."""
    selected = []
    for _ in range(count):
        source_index = state["source_cursor"] % (step - 1)
        state["source_cursor"] += 1
        source = pools[source_index]
        position = state["item_cursors"][source_index] % len(source)
        state["item_cursors"][source_index] += 1
        selected.append(source[position])
    return selected


def _usage(train_ids: list[list[str]], gate_ids: list[list[str]], split: str) -> list[dict]:
    train_state = {"source_cursor": 0, "item_cursors": [0] * len(train_ids)}
    gate_state = {"source_cursor": 0, "item_cursors": [0] * len(train_ids)}
    usage = []
    for step in range(len(train_ids)):
        segment_id = f"seg{step}"
        if step == 0:
            old, gates = [], []
        elif step == 1:
            old, gates = train_ids[0].copy(), gate_ids[0].copy()
        else:
            old = train_ids[step - 1][:2] + _older_ids(train_ids, step, 2, train_state)
            gates = gate_ids[step - 1].copy() + _older_ids(gate_ids, step, 4, gate_state)
        if split != "train":
            old = [qa_id for pool in train_ids[:step] for qa_id in pool]
        usage.append(
            {
                "segment_id": segment_id,
                "task_new_qa_ids": train_ids[step].copy(),
                "task_old_qa_ids": old,
                "gate_qa_ids": gates,
            }
        )
    return usage


def assemble_document(
    document: dict, candidates: list[dict], review_decisions: list[dict], qa_config: dict
) -> dict:
    """Choose one QA per reviewed fact and freeze a complete trajectory.

    A segment shortfall returns all eligible distinct facts for analysis, but no
    partial trajectory can enter the final JSONL dataset.
    """
    seed = _nonnegative_integer(qa_config["role_seed"], "role_seed")
    document_id, layout = validate_text_layout(document)
    segment_ids = tuple(layout)
    _, task_counts, gate_counts = qa_quotas(len(segment_ids))
    task_role = "train" if document["split"] == "train" else "evaluation"
    text = document["text"]
    canonical = [
        _candidate_qa(qa, text, layout, qa_config["max_answer_chars"]) for qa in candidates
    ]
    qa_ids = [qa["qa_id"] for qa in canonical]
    if len(set(qa_ids)) != len(qa_ids):
        raise ValueError("candidate QA IDs must be unique")
    decisions = {}
    for decision in review_decisions:
        qa_id = _nonempty(decision["qa_id"], "review qa_id")
        if qa_id in decisions:
            raise ValueError(f"duplicate document-review decision for {qa_id}")
        if not isinstance(decision["accepted"], bool):
            raise ValueError(f"review accepted must be boolean for {qa_id}")
        if not isinstance(decision["reason"], str):
            raise ValueError(f"review reason must be a string for {qa_id}")
        if decision["accepted"]:
            _nonempty(decision["fact_group_id"], f"{qa_id}.fact_group_id")
        decisions[qa_id] = decision
    if set(decisions) != set(qa_ids):
        raise ValueError("document-review decisions must cover every candidate exactly once")

    representatives = {}
    accepted_count = 0
    for order, qa in enumerate(canonical):
        decision = decisions[qa["qa_id"]]
        if not decision["accepted"]:
            continue
        accepted_count += 1
        group_id = decision["fact_group_id"]
        key = (int(qa["segment_id"][3:]), order)
        if group_id not in representatives or key < representatives[group_id][0]:
            representatives[group_id] = (key, dict(qa, fact_group_id=group_id))

    by_segment = {segment_id: [] for segment_id in segment_ids}
    for _, qa in representatives.values():
        by_segment[qa["segment_id"]].append(qa)
    for segment_id in segment_ids:
        by_segment[segment_id].sort(
            key=lambda qa: (_role_rank(seed, document_id, segment_id, qa["qa_id"]), qa["qa_id"])
        )
    eligible_qas = [qa for segment_id in segment_ids for qa in by_segment[segment_id]]
    shortfalls = []
    for index, segment_id in enumerate(segment_ids):
        available = len(by_segment[segment_id])
        train_needed = task_counts[index]
        gate_needed = gate_counts[index]
        missing = max(0, train_needed + gate_needed - available)
        if missing:
            shortfalls.append(
                {
                    "segment_id": segment_id,
                    "available": available,
                    "required_task": train_needed,
                    "required_gate": gate_needed,
                    "task_missing": max(0, train_needed - available),
                    "gate_missing": max(0, gate_needed - max(0, available - train_needed)),
                    "missing_total": missing,
                }
            )
    statistics = {
        "candidate_count": len(canonical),
        "review_accepted_count": accepted_count,
        "review_rejected_count": len(canonical) - accepted_count,
        "duplicate_fact_count": accepted_count - len(representatives),
        "eligible_fact_count": len(representatives),
        "eligible_by_segment": {
            segment_id: len(by_segment[segment_id]) for segment_id in segment_ids
        },
    }
    if shortfalls:
        return {
            "ok": False,
            "trajectory": None,
            "eligible_qas": eligible_qas,
            "shortfalls": shortfalls,
            "statistics": statistics,
        }

    train_ids, gate_ids, qas = [], [], []
    for index, segment_id in enumerate(segment_ids):
        ordered = by_segment[segment_id]
        train = ordered[: task_counts[index]]
        gate = ordered[task_counts[index] : task_counts[index] + gate_counts[index]]
        train_ids.append([qa["qa_id"] for qa in train])
        gate_ids.append([qa["qa_id"] for qa in gate])
        qas.extend(dict(qa, role=task_role) for qa in train)
        qas.extend(dict(qa, role="gate") for qa in gate)
    trajectory = {
        "trajectory_id": document["trajectory_id"],
        "document_id": document_id,
        "dedup_cluster": document["dedup_cluster"],
        "split": document["split"],
        "source": {
            "file": document["source"]["file"],
            "row_group": document["source"]["row_group"],
            "row_index": document["source"]["row_index"],
        },
        "window_char_span": list(document["window_char_span"]),
        "text": text,
        "segments": [
            {"segment_id": segment["segment_id"], "char_span": list(segment["char_span"])}
            for segment in document["segments"]
        ],
        "qas": qas,
        "usage": _usage(train_ids, gate_ids, document["split"]),
        "text_char_length": len(text),
        "estimated_tokens": len(text) / 4,
        "estimated_tokens_rule": TOKEN_ESTIMATION_RULE,
    }
    validate_trajectory(trajectory, qa_config)
    return {
        "ok": True,
        "trajectory": trajectory,
        "eligible_qas": eligible_qas,
        "shortfalls": [],
        "statistics": statistics,
    }


def validate_trajectory(trajectory: dict, qa_config: dict) -> None:
    """Reject a trajectory that violates offsets, fact isolation or the fixed schedule."""
    seed = _nonnegative_integer(qa_config["role_seed"], "role_seed")
    document_id, layout = validate_text_layout(trajectory)
    segment_ids = tuple(layout)
    _, task_counts, gate_counts = qa_quotas(len(segment_ids))
    task_role = "train" if trajectory["split"] == "train" else "evaluation"
    text = trajectory["text"]
    if trajectory["text_char_length"] != len(text):
        raise ValueError("text_char_length does not match text")
    if trajectory["estimated_tokens"] != len(text) / 4:
        raise ValueError("estimated_tokens must use the character estimate")
    if trajectory["estimated_tokens_rule"] != TOKEN_ESTIMATION_RULE:
        raise ValueError("estimated_tokens_rule is not the canonical rule")
    qas = trajectory["qas"]
    if not isinstance(qas, list) or len(qas) != sum(task_counts) + sum(gate_counts):
        raise ValueError("trajectory must contain 8N QAs")
    seen_ids, seen_facts = set(), set()
    train_ids, gate_ids = [], []
    expected_order = []
    for index, segment_id in enumerate(segment_ids):
        segment_qas = [qa for qa in qas if qa["segment_id"] == segment_id]
        if len(segment_qas) != task_counts[index] + gate_counts[index]:
            raise ValueError(f"QA quota is wrong for {segment_id}")
        sorted_qas = sorted(
            segment_qas,
            key=lambda qa: (_role_rank(seed, document_id, segment_id, qa["qa_id"]), qa["qa_id"]),
        )
        if segment_qas != sorted_qas:
            raise ValueError(f"QA role order differs from the fixed seed for {segment_id}")
        expected_order.extend(qa["qa_id"] for qa in segment_qas)
        train_ids.append([qa["qa_id"] for qa in segment_qas if qa["role"] == task_role])
        gate_ids.append([qa["qa_id"] for qa in segment_qas if qa["role"] == "gate"])
        if len(train_ids[-1]) != task_counts[index]:
            raise ValueError(f"task QA quota is wrong for {segment_id}")
        if len(gate_ids[-1]) != gate_counts[index]:
            raise ValueError(f"gate QA quota is wrong for {segment_id}")
        if [qa["role"] for qa in segment_qas] != [task_role] * len(train_ids[-1]) + ["gate"] * len(
            gate_ids[-1]
        ):
            raise ValueError(f"QA roles are interleaved for {segment_id}")
        for qa in segment_qas:
            if set(qa) != set(QA_FIELDS) | {"role"}:
                raise ValueError(f"QA fields are not canonical for {qa['qa_id']}")
            checked = _candidate_qa(qa, text, layout, qa_config["max_answer_chars"])
            if any(qa[key] != checked[key] for key in checked):
                raise ValueError(f"QA differs from its canonical offsets: {qa['qa_id']}")
            _nonempty(qa["fact_group_id"], f"{qa['qa_id']}.fact_group_id")
            if qa["qa_id"] in seen_ids:
                raise ValueError(f"QA ID is repeated: {qa['qa_id']}")
            if qa["fact_group_id"] in seen_facts:
                raise ValueError(f"fact group appears in more than one QA: {qa['fact_group_id']}")
            seen_ids.add(qa["qa_id"])
            seen_facts.add(qa["fact_group_id"])
    if [qa["qa_id"] for qa in qas] != expected_order:
        raise ValueError("QAs are not ordered by source segment")
    if trajectory["usage"] != _usage(train_ids, gate_ids, trajectory["split"]):
        raise ValueError("usage schedule differs from the fixed update and gate schedule")
