"""Read canonical FactQA trajectories without changing their text or QA schedule."""

from dataclasses import dataclass
import json
from pathlib import Path

from latent_working_memory.data_preparation.fineweb_factqa.assembly import validate_trajectory
from latent_working_memory.data_preparation.pretrain.fineweb import document_split
from latent_working_memory.data_preparation.fineweb_source import split_fractions
from latent_working_memory.data_preparation.segmentation import SegmentationConfig


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
class StepUsage:
    segment_id: str
    new_qa_ids: tuple[str, ...]
    old_qa_ids: tuple[str, ...]
    gate_qa_ids: tuple[str, ...]


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
    usage: tuple[StepUsage, ...]
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
            "usage",
            "text_char_length",
            "estimated_tokens",
            "estimated_tokens_rule",
        },
        "FactQA trajectory",
    )
    if expected_split not in ("train", "dev", "test") or record["split"] != expected_split:
        raise ValueError("trajectory split differs from the requested dataset split")
    # Reuse the writer's contract: fact isolation, source spans, role quotas and
    # the exact fixed usage schedule, including all historical dev/test tasks.
    validate_trajectory(record, qa_config)
    _exact_fields(record["source"], {"file", "row_group", "row_index"}, "source")
    for segment in record["segments"]:
        _exact_fields(segment, {"segment_id", "char_span"}, "segment")
    for step in record["usage"]:
        _exact_fields(
            step,
            {"segment_id", "task_new_qa_ids", "task_old_qa_ids", "gate_qa_ids"},
            "usage step",
        )

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
    usage = tuple(
        StepUsage(
            step["segment_id"],
            tuple(step["task_new_qa_ids"]),
            tuple(step["task_old_qa_ids"]),
            tuple(step["gate_qa_ids"]),
        )
        for step in record["usage"]
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
        usage=usage,
        full_input_ids=encoded[0],
    )


def load_factqa(dataset_dir: str | Path, tokenizer) -> dict[str, tuple[FactQATrajectory, ...]]:
    """Load published train/dev/test JSONL and enforce their frozen source splits."""
    root = Path(dataset_dir)
    with (root / "preparation.json").open(encoding="utf-8") as stream:
        preparation = json.load(stream)
    qa_config = preparation["qa"]
    if type(qa_config["max_answer_chars"]) is not int or qa_config["max_answer_chars"] < 1:
        raise ValueError("qa.max_answer_chars must be a positive integer")
    pool_config = preparation["source_pool_config"]
    if "source" in pool_config:
        # The published train1000 dataset records its original source protocol.
        source_seed = pool_config["source"]["data_seed"]
        fractions = tuple(pool_config["source"]["split_fractions"])
    else:
        source_seed = pool_config["source_seed"]
        fractions = split_fractions(pool_config["split_counts"])
    window_config = pool_config["window"]
    segmentation = None
    if set(window_config) == {
        "min_segments",
        "max_segments",
        "min_segment_chars",
        "max_segment_chars",
    }:
        # Published 20260930 FactQA sampled arbitrary integer character lengths.
        min_segments, max_segments = window_config["min_segments"], window_config["max_segments"]
        min_chars, max_chars = (
            window_config["min_segment_chars"],
            window_config["max_segment_chars"],
        )
    else:
        segmentation = SegmentationConfig(**window_config)
        if segmentation.continuation_tokens != 0:
            raise ValueError("FactQA window requires continuation_tokens=0")
        min_segments, max_segments = segmentation.min_segments, segmentation.max_segments
    seen_documents, seen_trajectories, seen_questions, seen_sources = set(), set(), set(), set()
    cluster_splits = {}
    result = {}
    for split in ("train", "dev", "test"):
        trajectories = []
        path = root / f"{split}.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    trajectory = tokenize_trajectory(record, tokenizer, qa_config, split)
                    if not min_segments <= len(trajectory.segments) <= max_segments:
                        raise ValueError(
                            "segment count differs from the frozen window specification"
                        )
                    lengths = [
                        segment.char_span[1] - segment.char_span[0]
                        for segment in trajectory.segments
                    ]
                    valid_lengths = (
                        all(min_chars <= length <= max_chars for length in lengths)
                        if segmentation is None
                        else all(segmentation.is_valid_segment_length(length) for length in lengths)
                    )
                    if not valid_lengths:
                        raise ValueError(
                            "segment length differs from the frozen window specification"
                        )
                    cluster = trajectory.dedup_cluster
                    if cluster in cluster_splits and cluster_splits[cluster] != split:
                        raise ValueError("a dedup cluster occurs in multiple dataset splits")
                    if (
                        document_split(
                            cluster,
                            source_seed,
                            fractions,
                        )
                        != split
                    ):
                        raise ValueError(
                            "trajectory split differs from its frozen source-cluster split"
                        )
                    source = trajectory.source
                    location = (source["file"], source["row_group"], source["row_index"])
                    if trajectory.document_id in seen_documents:
                        raise ValueError("duplicate source document in FactQA dataset")
                    if trajectory.trajectory_id in seen_trajectories:
                        raise ValueError("duplicate trajectory ID in FactQA dataset")
                    if location in seen_sources:
                        raise ValueError("duplicate source row in FactQA dataset")
                    if seen_questions.intersection(trajectory.qas):
                        raise ValueError("duplicate QA ID in FactQA dataset")
                    seen_documents.add(trajectory.document_id)
                    seen_trajectories.add(trajectory.trajectory_id)
                    seen_sources.add(location)
                    seen_questions.update(trajectory.qas)
                    cluster_splits[cluster] = split
                    trajectories.append(trajectory)
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
        result[split] = tuple(trajectories)
    return result
