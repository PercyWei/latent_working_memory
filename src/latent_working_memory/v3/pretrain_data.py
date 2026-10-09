"""加载完整 AE／LM 正文，按正文 token 长度过滤整条样本。"""

from dataclasses import dataclass
import json
from pathlib import Path
import random

from latent_working_memory.data_preparation.fineweb_multisegment.records import MultisegmentSample
from latent_working_memory.data_preparation.fineweb_source import SourceWindowTracker
from latent_working_memory.data_preparation.pretrain.text_samples import TextSample
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
    if len(example.input_ids) > limit:
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
    """已有 TextSample 也保留完整输入与目标，使用同一正文长度上限。"""
    minimum, limit = _limits(config, model_window)
    root = Path(config.training.dataset_dir)
    seen_samples, document_sources, cluster_splits = set(), {}, {}
    splits, statistics = {}, {}
    for split in ("train", "dev", "test"):
        examples, counts = [], _counts()
        path = root / f"{split}.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    sample = TextSample(**json.loads(line))
                    if sample.sample_id in seen_samples:
                        raise ValueError("duplicate pretraining sample_id")
                    identity = (split, sample.dedup_cluster)
                    if (
                        sample.document_id in document_sources
                        and document_sources[sample.document_id] != identity
                    ):
                        raise ValueError("source document changes split or dedup cluster")
                    if (
                        sample.dedup_cluster in cluster_splits
                        and cluster_splits[sample.dedup_cluster] != split
                    ):
                        raise ValueError("dedup cluster occurs in multiple pretraining splits")
                    seen_samples.add(sample.sample_id)
                    document_sources[sample.document_id] = identity
                    cluster_splits[sample.dedup_cluster] = split
                    counts["read"] += 1
                    counts["read_by_task"][sample.task] += 1
                    input_ids = tuple(tokenizer.encode(sample.text, add_special_tokens=False))
                    target_ids = (
                        input_ids
                        if sample.task == "ae"
                        else tuple(tokenizer.encode(sample.continuation, add_special_tokens=False))
                    )
                    if not target_ids:
                        raise ValueError("continuation tokenized to an empty target")
                    example = PretrainExample(
                        sample.sample_id,
                        sample.document_id,
                        sample.dedup_cluster,
                        sample.task,
                        input_ids,
                        target_ids,
                    )
                    if _keep(example, minimum, limit, counts):
                        examples.append(example)
                except (ValueError, KeyError, TypeError) as error:
                    error.add_note(f"{path}:{line_number}")
                    raise
        _summarize(examples, counts)
        splits[split], statistics[split] = tuple(examples), counts
    return splits, {
        "kind": "text_samples",
        "view": "text_samples",
        "input_token_interval": [minimum, limit],
        "splits": statistics,
    }


def load_multisegment_pretraining(config, tokenizer, model_window):
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
                    if _keep(example, minimum, limit, counts):
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
        "view": training.pretrain_data_view,
        "input_token_interval": [minimum, limit],
        "sampling_seed": training.seed,
        "lm_ratio": 1.0 if lm_only else training.lm_ratio,
        "lm_target_tokens": training.lm_target_tokens,
        "splits": statistics,
    }
