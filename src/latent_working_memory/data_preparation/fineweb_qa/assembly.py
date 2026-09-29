"""Apply document review and assemble one fixed FineWeb FactQA trajectory."""

from __future__ import annotations

import hashlib
import json


SEGMENT_IDS = tuple(f"seg{i}" for i in range(8))
TRAIN_PER_SEGMENT = (4,) * 8
GATE_PER_SEGMENT = (8, 4, 4, 4, 4, 4, 4, 0)
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
TOKEN_ESTIMATION_RULE = "len(text) / 4"


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _span(value: object, name: str, length: int) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item character span")
    start = _nonnegative_integer(value[0], f"{name} start")
    end = _nonnegative_integer(value[1], f"{name} end")
    if not start < end <= length:
        raise ValueError(f"{name} is outside the text")
    return start, end


def _qa_config(qa_config: dict) -> int:
    if not isinstance(qa_config, dict):
        raise ValueError("QA configuration must be a mapping")
    seed = _nonnegative_integer(qa_config["role_seed"], "role_seed")
    for key, expected in (
        ("train_per_segment", TRAIN_PER_SEGMENT),
        ("gate_per_segment", GATE_PER_SEGMENT),
    ):
        actual = qa_config[key]
        if (
            not isinstance(actual, (list, tuple))
            or len(actual) != len(expected)
            or any(isinstance(n, bool) or not isinstance(n, int) for n in actual)
            or tuple(actual) != expected
        ):
            raise ValueError(f"{key} must be {list(expected)} for this pilot")
    return seed


def _document_layout(document: dict) -> tuple[str, dict[str, tuple[int, int]]]:
    _nonempty(document["trajectory_id"], "trajectory_id")
    document_id = _nonempty(document["document_id"], "document_id")
    _nonempty(document["dedup_cluster"], "dedup_cluster")
    if document["split"] != "train":
        raise ValueError("this pilot assembles train trajectories only")
    source = document["source"]
    _nonempty(source["file"], "source.file")
    _nonnegative_integer(source["row_group"], "source.row_group")
    _nonnegative_integer(source["row_index"], "source.row_index")
    text = document["text"]
    if not isinstance(text, str) or not text:
        raise ValueError("trajectory text must be nonempty")
    window_start, window_end = _span(document["window_char_span"], "window_char_span", 2**63)
    if window_end - window_start != len(text):
        raise ValueError("window_char_span does not match trajectory text length")
    segments = document["segments"]
    if not isinstance(segments, list) or len(segments) != 8:
        raise ValueError("trajectory must have eight segments")
    layout = {}
    next_start = 0
    for expected_id, segment in zip(SEGMENT_IDS, segments, strict=True):
        if segment["segment_id"] != expected_id:
            raise ValueError(f"expected segment {expected_id}")
        start, end = _span(segment["char_span"], f"{expected_id}.char_span", len(text))
        if start != next_start:
            raise ValueError("segment spans must consecutively cover the trajectory text")
        layout[expected_id] = (start, end)
        next_start = end
    if next_start != len(text):
        raise ValueError("segment spans do not cover the trajectory text")
    return document_id, layout


def _candidate_qa(candidate: dict, text: str, layout: dict[str, tuple[int, int]]) -> dict:
    qa_id = _nonempty(candidate["qa_id"], "qa_id")
    segment_id = candidate["segment_id"]
    if segment_id not in layout:
        raise ValueError(f"unknown segment for {qa_id}: {segment_id}")
    fact_statement = _nonempty(candidate["fact_statement"], f"{qa_id}.fact_statement")
    question = _nonempty(candidate["question"], f"{qa_id}.question")
    answer = _nonempty(candidate["answer"], f"{qa_id}.answer")
    if len(answer) > 128:
        raise ValueError(f"answer exceeds 128 characters for {qa_id}")
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


def _usage(train_ids: list[list[str]], gate_ids: list[list[str]]) -> list[dict]:
    train_state = {"source_cursor": 0, "item_cursors": [0] * 8}
    gate_state = {"source_cursor": 0, "item_cursors": [0] * 8}
    usage = []
    for step, segment_id in enumerate(SEGMENT_IDS):
        if step == 0:
            old, gates = [], []
        elif step == 1:
            old, gates = train_ids[0].copy(), gate_ids[0].copy()
        else:
            old = train_ids[step - 1][:2] + _older_ids(train_ids, step, 2, train_state)
            gates = gate_ids[step - 1].copy() + _older_ids(gate_ids, step, 4, gate_state)
        usage.append(
            {
                "segment_id": segment_id,
                "train_new_qa_ids": train_ids[step].copy(),
                "train_old_qa_ids": old,
                "gate_qa_ids": gates,
            }
        )
    return usage


def assemble_document(
    document: dict, candidates: list[dict], review_decisions: list[dict], qa_config: dict
) -> dict:
    """Choose one QA per reviewed fact and freeze a complete train trajectory.

    A segment shortfall returns all eligible distinct facts for analysis, but no
    partial trajectory can enter the final JSONL dataset.
    """
    seed = _qa_config(qa_config)
    document_id, layout = _document_layout(document)
    text = document["text"]
    canonical = [_candidate_qa(qa, text, layout) for qa in candidates]
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

    by_segment = {segment_id: [] for segment_id in SEGMENT_IDS}
    for _, qa in representatives.values():
        by_segment[qa["segment_id"]].append(qa)
    for segment_id in SEGMENT_IDS:
        by_segment[segment_id].sort(
            key=lambda qa: (_role_rank(seed, document_id, segment_id, qa["qa_id"]), qa["qa_id"])
        )
    eligible_qas = [qa for segment_id in SEGMENT_IDS for qa in by_segment[segment_id]]
    shortfalls = []
    for index, segment_id in enumerate(SEGMENT_IDS):
        available = len(by_segment[segment_id])
        train_needed = TRAIN_PER_SEGMENT[index]
        gate_needed = GATE_PER_SEGMENT[index]
        missing = max(0, train_needed + gate_needed - available)
        if missing:
            shortfalls.append(
                {
                    "segment_id": segment_id,
                    "available": available,
                    "required_train": train_needed,
                    "required_gate": gate_needed,
                    "train_missing": max(0, train_needed - available),
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
            segment_id: len(by_segment[segment_id]) for segment_id in SEGMENT_IDS
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
    for index, segment_id in enumerate(SEGMENT_IDS):
        ordered = by_segment[segment_id]
        train = ordered[: TRAIN_PER_SEGMENT[index]]
        gate = ordered[
            TRAIN_PER_SEGMENT[index] : TRAIN_PER_SEGMENT[index] + GATE_PER_SEGMENT[index]
        ]
        train_ids.append([qa["qa_id"] for qa in train])
        gate_ids.append([qa["qa_id"] for qa in gate])
        qas.extend(dict(qa, role="train") for qa in train)
        qas.extend(dict(qa, role="gate") for qa in gate)
    trajectory = {
        "trajectory_id": document["trajectory_id"],
        "document_id": document_id,
        "dedup_cluster": document["dedup_cluster"],
        "split": "train",
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
        "usage": _usage(train_ids, gate_ids),
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
    seed = _qa_config(qa_config)
    document_id, layout = _document_layout(trajectory)
    text = trajectory["text"]
    if trajectory["text_char_length"] != len(text):
        raise ValueError("text_char_length does not match text")
    if trajectory["estimated_tokens"] != len(text) / 4:
        raise ValueError("estimated_tokens must use the character estimate")
    if trajectory["estimated_tokens_rule"] != TOKEN_ESTIMATION_RULE:
        raise ValueError("estimated_tokens_rule is not the canonical rule")
    qas = trajectory["qas"]
    if not isinstance(qas, list) or len(qas) != sum(TRAIN_PER_SEGMENT) + sum(GATE_PER_SEGMENT):
        raise ValueError("trajectory must contain 64 QAs")
    seen_ids, seen_facts = set(), set()
    train_ids, gate_ids = [], []
    expected_order = []
    for index, segment_id in enumerate(SEGMENT_IDS):
        segment_qas = [qa for qa in qas if qa["segment_id"] == segment_id]
        if len(segment_qas) != TRAIN_PER_SEGMENT[index] + GATE_PER_SEGMENT[index]:
            raise ValueError(f"QA quota is wrong for {segment_id}")
        sorted_qas = sorted(
            segment_qas,
            key=lambda qa: (_role_rank(seed, document_id, segment_id, qa["qa_id"]), qa["qa_id"]),
        )
        if segment_qas != sorted_qas:
            raise ValueError(f"QA role order differs from the fixed seed for {segment_id}")
        expected_order.extend(qa["qa_id"] for qa in segment_qas)
        train_ids.append([qa["qa_id"] for qa in segment_qas if qa["role"] == "train"])
        gate_ids.append([qa["qa_id"] for qa in segment_qas if qa["role"] == "gate"])
        if len(train_ids[-1]) != TRAIN_PER_SEGMENT[index]:
            raise ValueError(f"train QA quota is wrong for {segment_id}")
        if len(gate_ids[-1]) != GATE_PER_SEGMENT[index]:
            raise ValueError(f"gate QA quota is wrong for {segment_id}")
        if [qa["role"] for qa in segment_qas] != ["train"] * len(train_ids[-1]) + ["gate"] * len(
            gate_ids[-1]
        ):
            raise ValueError(f"QA roles are interleaved for {segment_id}")
        for qa in segment_qas:
            if set(qa) != set(QA_FIELDS) | {"role"}:
                raise ValueError(f"QA fields are not canonical for {qa['qa_id']}")
            checked = _candidate_qa(qa, text, layout)
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
    if trajectory["usage"] != _usage(train_ids, gate_ids):
        raise ValueError("usage schedule differs from the fixed update and gate schedule")
