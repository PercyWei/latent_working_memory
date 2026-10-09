"""加载完整 AE／LM 正文，仅对训练集应用正文 token 长度上限。"""

from dataclasses import dataclass
import json
from pathlib import Path
import random

from latent_working_memory.data_preparation.fineweb_multisegment.records import MultisegmentSample
from latent_working_memory.data_preparation.fineweb_source import SourceWindowTracker
from latent_working_memory.data_preparation.segmentation import SegmentationConfig


@dataclass(frozen=True, slots=True)
class PretrainExample:
    sample_id: str
    document_id: str
    dedup_cluster: str
    task: str
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


def _limits(config, model_window):
    if type(model_window) is not int or model_window < 1:
        raise ValueError("model_window must be a positive integer")
    requested = config.training.max_input_tokens
    limit = model_window if requested is None else min(requested, model_window)
    minimum = config.training.min_input_tokens
    if config.objective.method == "icae_multi":
        minimum = max(minimum, config.objective.icae_max_segments)
    elif config.objective.method == "autocompressors":
        minimum = max(minimum, 2 * config.objective.ac_num_segments)
    return minimum, limit


def _counts():
    return {
        "read": 0,
        "kept": 0,
        "filtered_too_short": 0,
        "filtered_too_long": 0,
        "read_by_task": {"ae": 0, "continuation": 0},
        "kept_by_task": {"ae": 0, "continuation": 0},
    }


def _length_statistics(values):
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "total": sum(values),
    }


def _keep(example, minimum, limit, counts):
    if len(example.input_ids) < minimum:
        counts["filtered_too_short"] += 1
        return False
    if limit is not None and len(example.input_ids) > limit:
        counts["filtered_too_long"] += 1
        return False
    counts["kept"] += 1
    counts["kept_by_task"][example.task] += 1
    return True


def _summarize(examples, counts):
    for name, values in (
        ("input_tokens", [len(row.input_ids) for row in examples]),
        ("target_tokens", [len(row.target_ids) for row in examples]),
    ):
        counts[name] = _length_statistics(values)


def load_pretraining(config, tokenizer, model_window):
    """完整正文只构造一个 AE 或 LM 样本；LM 续文来自正文后的 continuation。"""
    minimum, limit = _limits(config, model_window)
    training = config.training
    root = Path(training.dataset_dir)
    metadata = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    window = SegmentationConfig(**metadata["config"]["window"])
    if window.continuation_tokens < 1:
        raise ValueError("continuation_tokens must be a positive integer")
    lm_only = config.objective.method == "autocompressors"
    seen_samples = set()
    source_windows = SourceWindowTracker()
    splits, statistics, token_pool = {}, {}, {}
    for split in ("train", "dev", "test"):
        split_limit = limit if split == "train" else None
        examples, segment_counts = [], []
        counts = {
            **_counts(),
            "short_continuation_sources": 0,
            "lm_to_ae_sources": 0,
        }
        path = root / f"{split}.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    sample = MultisegmentSample(**record)
                    sample.validate_plan(window)
                    if sample.split != split:
                        raise ValueError("record split differs from its JSONL split")
                    if sample.trajectory_id in seen_samples:
                        raise ValueError("duplicate pretraining trajectory_id")
                    source_windows.add(record, len(sample.continuation))
                    seen_samples.add(sample.trajectory_id)
                    counts["read"] += 1
                    rng = random.Random(f"{training.seed}:pretraining:{sample.trajectory_id}")
                    task = "continuation" if lm_only or rng.random() < training.lm_ratio else "ae"
                    counts["read_by_task"][task] += 1
                    input_ids = tuple(
                        token_pool.setdefault(token, token)
                        for token in tokenizer.encode(sample.text, add_special_tokens=False)
                    )
                    target_ids = input_ids
                    if task == "continuation":
                        continuation = tokenizer.encode(
                            sample.continuation, add_special_tokens=False
                        )
                        if len(continuation) < training.lm_target_tokens:
                            counts["short_continuation_sources"] += 1
                            if not lm_only:
                                task = "ae"
                                counts["lm_to_ae_sources"] += 1
                        if task == "continuation":
                            target_ids = tuple(
                                token_pool.setdefault(token, token)
                                for token in continuation[: training.lm_target_tokens]
                            )
                    example = PretrainExample(
                        f"{sample.trajectory_id}:{task}",
                        sample.document_id,
                        sample.dedup_cluster,
                        task,
                        input_ids,
                        target_ids,
                    )
                    if _keep(example, minimum, split_limit, counts):
                        examples.append(example)
                        segment_counts.append(len(sample.segments))
                except (ValueError, KeyError, TypeError) as error:
                    error.add_note(f"{path}:{line_number}")
                    raise
        _summarize(examples, counts)
        counts["source_segments"] = _length_statistics(segment_counts)
        splits[split], statistics[split] = tuple(examples), counts
    return splits, {
        "kind": "multisegment_text",
        "input_token_interval": [minimum, limit],
        "upper_limit_split": "train",
        "sampling_seed": training.seed,
        "lm_ratio": 1.0 if lm_only else training.lm_ratio,
        "lm_target_tokens": training.lm_target_tokens,
        "splits": statistics,
    }
