"""Read canonical FactQA trajectories and index complete per-segment QA pools."""

from dataclasses import dataclass
import json
from pathlib import Path

from latent_working_memory.data_preparation.fineweb_factqa.assembly import validate_trajectory
from latent_working_memory.data_preparation.fineweb_source import SourceWindowTracker


@dataclass(frozen=True, slots=True)
class Segment:
    segment_id: str
    char_span: tuple[int, int]
    input_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class QA:
    qa_id: str
    segment_id: str
    fact_group_id: str
    fact_statement: str
    question: str
    answer: str
    role: str
    evidence_char_span: tuple[int, int]
    answer_char_span: tuple[int, int]
    question_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FactQATrajectory:
    trajectory_id: str
    document_id: str
    dedup_cluster: str
    split: str
    text: str
    source: dict
    window_char_span: tuple[int, int]
    segments: tuple[Segment, ...]
    qas: dict[str, QA]
    task_qa_ids: tuple[tuple[str, ...], ...]
    gate_qa_ids: tuple[tuple[str, ...], ...]
    full_input_ids: tuple[int, ...]

    def prefix_ids(self, step: int, tokenizer) -> tuple[int, ...]:
        """Tokenize the original prefix, which need not equal concatenated segments."""
        if type(step) is not int or not 0 <= step < len(self.segments):
            raise IndexError("prefix step must be a zero-based segment index")
        if step == len(self.segments) - 1:
            return self.full_input_ids
        return _encode_texts(tokenizer, [self.text[: self.segments[step].char_span[1]]])[0]


def _exact_fields(value: dict, fields: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} requires exactly {sorted(fields)}")


def _encode_texts(tokenizer, texts: list[str]) -> tuple[tuple[int, ...], ...]:
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    if any(not ids for ids in encoded):
        raise ValueError("FactQA text, questions and answers must tokenize to nonempty sequences")
    return tuple(tuple(ids) for ids in encoded)


def tokenize_trajectory(
    record: dict, tokenizer, qa_config: dict, expected_split: str
) -> FactQATrajectory:
    """Validate one published record, then tokenize source and QA fields separately."""
    _exact_fields(
        record,
        {
            "trajectory_id",
            "document_id",
            "dedup_cluster",
            "split",
            "source",
            "window_char_span",
            "text",
            "segments",
            "qas",
            "text_char_length",
            "estimated_tokens",
            "estimated_tokens_rule",
        },
        "FactQA trajectory",
    )
    if expected_split not in ("train", "dev", "test") or record["split"] != expected_split:
        raise ValueError("trajectory split differs from the requested dataset split")
    # Reuse the writer's contract: fact isolation, source spans and role quotas.
    validate_trajectory(record, qa_config)
    for segment in record["segments"]:
        _exact_fields(segment, {"segment_id", "char_span"}, "segment")

    texts = [record["text"]]
    texts.extend(record["text"][slice(*segment["char_span"])] for segment in record["segments"])
    texts.extend(qa["question"] for qa in record["qas"])
    texts.extend(qa["answer"] for qa in record["qas"])
    encoded = _encode_texts(tokenizer, texts)
    segment_count, qa_count = len(record["segments"]), len(record["qas"])
    question_offset = 1 + segment_count
    answer_offset = question_offset + qa_count
    segments = tuple(
        Segment(segment["segment_id"], tuple(segment["char_span"]), encoded[1 + index])
        for index, segment in enumerate(record["segments"])
    )
    qas = {
        qa["qa_id"]: QA(
            qa_id=qa["qa_id"],
            segment_id=qa["segment_id"],
            fact_group_id=qa["fact_group_id"],
            fact_statement=qa["fact_statement"],
            question=qa["question"],
            answer=qa["answer"],
            role=qa["role"],
            evidence_char_span=tuple(qa["evidence_char_span"]),
            answer_char_span=tuple(qa["answer_char_span"]),
            question_ids=encoded[question_offset + index],
            answer_ids=encoded[answer_offset + index],
        )
        for index, qa in enumerate(record["qas"])
    }
    task_qa_ids = tuple(
        tuple(
            qa.qa_id
            for qa in qas.values()
            if qa.segment_id == segment.segment_id and qa.role == "task"
        )
        for segment in segments
    )
    gate_qa_ids = tuple(
        tuple(
            qa.qa_id
            for qa in qas.values()
            if qa.segment_id == segment.segment_id and qa.role == "gate"
        )
        for segment in segments
    )
    return FactQATrajectory(
        trajectory_id=record["trajectory_id"],
        document_id=record["document_id"],
        dedup_cluster=record["dedup_cluster"],
        split=record["split"],
        text=record["text"],
        source=dict(record["source"]),
        window_char_span=tuple(record["window_char_span"]),
        segments=segments,
        qas=qas,
        task_qa_ids=task_qa_ids,
        gate_qa_ids=gate_qa_ids,
        full_input_ids=encoded[0],
    )


def load_factqa(dataset_dir: str | Path, tokenizer) -> dict[str, tuple[FactQATrajectory, ...]]:
    """Load published JSONL using its saved character spans, complete QAs and splits."""
    root = Path(dataset_dir)
    with (root / "preparation.json").open(encoding="utf-8") as stream:
        preparation = json.load(stream)
    qa_config = preparation["qa"]
    if type(qa_config["max_answer_chars"]) is not int or qa_config["max_answer_chars"] < 1:
        raise ValueError("qa.max_answer_chars must be a positive integer")
    seen_trajectories, seen_questions = set(), set()
    source_windows = SourceWindowTracker()
    result = {}
    for split in ("train", "dev", "test"):
        trajectories = []
        path = root / f"{split}.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    trajectory = tokenize_trajectory(record, tokenizer, qa_config, split)
                    if trajectory.trajectory_id in seen_trajectories:
                        raise ValueError("duplicate trajectory ID in FactQA dataset")
                    source_windows.add(record)
                    if seen_questions.intersection(trajectory.qas):
                        raise ValueError("duplicate QA ID in FactQA dataset")
                    seen_trajectories.add(trajectory.trajectory_id)
                    seen_questions.update(trajectory.qas)
                    trajectories.append(trajectory)
                except (ValueError, KeyError, TypeError) as error:
                    error.add_note(f"{path}:{line_number}")
                    raise
        result[split] = tuple(trajectories)
    return result


def filter_factqa(
    splits: dict[str, tuple[FactQATrajectory, ...]],
    max_input_tokens: int | None,
) -> tuple[dict[str, tuple[FactQATrajectory, ...]], dict]:
    """按指定正文长度上限筛选 train；None 跳过上限筛选，dev/test 保留完整轨迹。"""
    filtered, statistics = {}, {}
    for split in ("train", "dev", "test"):
        rows = splits[split]
        kept = (
            tuple(row for row in rows if len(row.full_input_ids) <= max_input_tokens)
            if split == "train" and max_input_tokens is not None
            else rows
        )
        filtered[split] = kept
        statistics[split] = {
            "read": len(rows),
            "kept": len(kept),
            "filtered_too_long": len(rows) - len(kept),
            "trajectories": len(kept),
            "source_tokens": sum(len(row.full_input_ids) for row in kept),
            "questions": sum(len(row.qas) for row in kept),
        }
    return filtered, {
        "kind": "factqa",
        "input_token_interval": [1, max_input_tokens],
        "upper_limit_split": "train",
        "splits": statistics,
    }
