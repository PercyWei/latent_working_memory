"""Tokenize canonical FineWeb AE/continuation samples for v3 pretraining."""

from dataclasses import dataclass
import json
from pathlib import Path
import random

from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig
from latent_working_memory.data_preparation.fineweb_multisegment.records import MultisegmentSample
from latent_working_memory.data_preparation.pretrain.text_samples import TextSample


@dataclass(frozen=True, slots=True)
class PretrainExample:
    sample_id: str
    document_id: str
    dedup_cluster: str
    task: str
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


def load_pretraining(
    dataset_dir: str | Path, tokenizer, min_input_tokens: int, max_input_tokens: int
) -> tuple[dict[str, tuple[PretrainExample, ...]], dict]:
    """Filter whole inputs by current-tokenizer length; never crop source or targets."""
    if (
        type(min_input_tokens) is not int
        or type(max_input_tokens) is not int
        or not 1 <= min_input_tokens <= max_input_tokens
    ):
        raise ValueError("input token interval requires positive integers with min <= max")
    root = Path(dataset_dir)
    seen_samples, document_sources, cluster_splits = set(), {}, {}
    splits, statistics = {}, {}
    for split in ("train", "dev", "test"):
        examples = []
        counts = {
            "read": 0,
            "kept": 0,
            "filtered_too_short": 0,
            "filtered_too_long": 0,
            "read_by_task": {"ae": 0, "continuation": 0},
            "kept_by_task": {"ae": 0, "continuation": 0},
        }
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
                    input_ids = tuple(
                        tokenizer.encode(sample.text, add_special_tokens=False, truncation=False)
                    )
                    if len(input_ids) < min_input_tokens:
                        counts["filtered_too_short"] += 1
                        continue
                    if len(input_ids) > max_input_tokens:
                        counts["filtered_too_long"] += 1
                        continue
                    target_ids = (
                        input_ids
                        if sample.task == "ae"
                        else tuple(
                            tokenizer.encode(
                                sample.continuation, add_special_tokens=False, truncation=False
                            )
                        )
                    )
                    if not target_ids:
                        raise ValueError("continuation tokenized to an empty target")
                    examples.append(
                        PretrainExample(
                            sample.sample_id,
                            sample.document_id,
                            sample.dedup_cluster,
                            sample.task,
                            input_ids,
                            target_ids,
                        )
                    )
                    counts["kept"] += 1
                    counts["kept_by_task"][sample.task] += 1
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
        for name, field in (("input_tokens", "input_ids"), ("target_tokens", "target_ids")):
            lengths = [len(getattr(example, field)) for example in examples]
            counts[name] = {
                "min": min(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
                "total": sum(lengths),
            }
        splits[split] = tuple(examples)
        statistics[split] = counts
    return splits, {
        "input_token_interval": [min_input_tokens, max_input_tokens],
        "splits": statistics,
    }


def load_multisegment_pretraining(
    dataset_dir: str | Path,
    tokenizer,
    min_input_tokens: int,
    max_input_tokens: int,
    view: str,
    lm_only: bool = False,
    seed: int = 20261004,
    lm_ratio: float = 0.5,
) -> tuple[dict[str, tuple[PretrainExample, ...]], dict]:
    """按来源独立抽取连续前缀与一个任务，加载后固定供各 epoch 复用。"""
    if (
        type(min_input_tokens) is not int
        or type(max_input_tokens) is not int
        or not 1 <= min_input_tokens <= max_input_tokens
    ):
        raise ValueError("input token interval requires positive integers with min <= max")
    if view != "multisegment_random_prefix":
        raise ValueError(f"unknown multisegment data view: {view}")
    root = Path(dataset_dir)
    metadata = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    preparation_config = DataPreparationConfig.from_mapping(metadata["config"]).window
    continuation_tokens = preparation_config.continuation_tokens
    seen_samples, document_sources, cluster_splits = set(), {}, {}
    splits, statistics = {}, {}
    # Share token integers across samples without duplicating each source for AE/LM.
    token_pool = {}
    for split in ("train", "dev", "test"):
        path = root / f"{split}.jsonl"
        examples, original_lengths, actual_lengths = [], [], []
        prefix_segments, available_segments = [], []
        counts = {
            "source_samples": 0,
            "kept_source_samples": 0,
            "cropped_source_samples": 0,
            "cropped_source_tokens": 0,
            "read": 0,
            "kept": 0,
            "filtered_too_short": 0,
            "short_continuation_sources": 0,
            "lm_to_ae_sources": 0,
            "read_by_task": {"ae": 0, "continuation": 0},
            "kept_by_task": {"ae": 0, "continuation": 0},
        }
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    sample = MultisegmentSample(**json.loads(line))
                    sample.validate_plan(preparation_config)
                    if sample.split != split:
                        raise ValueError("record split differs from its JSONL split")
                    if sample.trajectory_id in seen_samples:
                        raise ValueError("duplicate pretraining trajectory_id")
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
                    seen_samples.add(sample.trajectory_id)
                    document_sources[sample.document_id] = identity
                    cluster_splits[sample.dedup_cluster] = split
                    counts["source_samples"] += 1
                    counts["read"] += 1
                    segment_ids = [
                        tokenizer.encode(
                            sample.text[slice(*segment["char_span"])],
                            add_special_tokens=False,
                            truncation=False,
                        )
                        for segment in sample.segments
                    ]
                    body_ids = [token for segment in segment_ids for token in segment]
                    cuts, total = [], 0
                    for segment in segment_ids:
                        total += len(segment)
                        if total > max_input_tokens:
                            break
                        cuts.append(total)
                    rng = random.Random(f"{seed}:pretraining:{sample.trajectory_id}")
                    count = rng.randint(1, len(cuts)) if cuts else 1
                    original_cut = cuts[count - 1] if cuts else len(segment_ids[0])
                    cut = min(original_cut, max_input_tokens)
                    task = "continuation" if lm_only or rng.random() < lm_ratio else "ae"
                    counts["read_by_task"][task] += 1
                    if cut < min_input_tokens:
                        counts["filtered_too_short"] += 1
                        continue
                    input_ids = tuple(
                        token_pool.setdefault(token, token) for token in body_ids[:cut]
                    )
                    target_ids = input_ids
                    if task == "continuation":
                        remaining = body_ids[cut:] + tokenizer.encode(
                            sample.continuation, add_special_tokens=False, truncation=False
                        )
                        if len(remaining) < continuation_tokens:
                            counts["short_continuation_sources"] += 1
                            if not lm_only:
                                task = "ae"
                                counts["lm_to_ae_sources"] += 1
                        if task == "continuation":
                            target_ids = tuple(
                                token_pool.setdefault(token, token)
                                for token in remaining[:continuation_tokens]
                            )
                    examples.append(
                        PretrainExample(
                            f"{sample.trajectory_id}:{task}",
                            sample.document_id,
                            sample.dedup_cluster,
                            task,
                            input_ids,
                            target_ids,
                        )
                    )
                    original_lengths.append(original_cut)
                    actual_lengths.append(cut)
                    prefix_segments.append(count)
                    available_segments.append(len(cuts))
                    counts["kept_source_samples"] += 1
                    counts["cropped_source_samples"] += int(cut < original_cut)
                    counts["cropped_source_tokens"] += original_cut - cut
                    counts["kept"] += 1
                    counts["kept_by_task"][task] += 1
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
        # Each retained source contributes one prefix and one training objective.
        for name, lengths in (
            ("original_source_prefix_tokens", original_lengths),
            ("actual_source_prefix_tokens", actual_lengths),
            ("prefix_segments", prefix_segments),
            ("available_prefix_segments", available_segments),
            ("input_tokens", [len(example.input_ids) for example in examples]),
            ("target_tokens", [len(example.target_ids) for example in examples]),
        ):
            counts[name] = {
                "min": min(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
                "total": sum(lengths),
            }
        splits[split] = tuple(examples)
        statistics[split] = counts
    return splits, {
        "kind": "multisegment_text",
        "view": view,
        "continuation_tokens": continuation_tokens,
        "sampling_seed": seed,
        "lm_ratio": 1.0 if lm_only else lm_ratio,
        "input_token_interval": [min_input_tokens, max_input_tokens],
        "splits": statistics,
    }
