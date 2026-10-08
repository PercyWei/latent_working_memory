"""Tokenize canonical FineWeb AE/continuation samples for v3 pretraining."""

from dataclasses import dataclass
import json
from pathlib import Path

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
) -> tuple[dict[str, tuple[PretrainExample, ...]], dict]:
    """Tokenize fixed character segments and pair AE/LM on the selected prefix.

    The maximum crops the selected prefix on the right. Continuation starts at
    that actual endpoint in the concatenated segment and continuation token stream.
    Source locations are provenance; loading never reads the original Parquet.
    """
    if (
        type(min_input_tokens) is not int
        or type(max_input_tokens) is not int
        or not 1 <= min_input_tokens <= max_input_tokens
    ):
        raise ValueError("input token interval requires positive integers with min <= max")
    if view not in {"multisegment_full", "multisegment_first_write"}:
        raise ValueError(f"unknown multisegment data view: {view}")
    root = Path(dataset_dir)
    metadata = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    preparation_config = DataPreparationConfig.from_mapping(metadata["config"]).window
    continuation_tokens = preparation_config.continuation_tokens
    tasks = ("continuation",) if lm_only else ("ae", "continuation")
    seen_samples, document_sources, cluster_splits = set(), {}, {}
    splits, statistics = {}, {}
    # Share token integers across samples and each input tuple between its AE/LM pair.
    token_pool = {}
    for split in ("train", "dev", "test"):
        path = root / f"{split}.jsonl"
        examples, original_lengths, actual_lengths = [], [], []
        counts = {
            "source_samples": 0,
            "kept_source_samples": 0,
            "cropped_source_samples": 0,
            "cropped_source_tokens": 0,
            "read": 0,
            "kept": 0,
            "filtered_too_short": 0,
            "filtered_continuation_length": 0,
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
                    counts["read"] += len(tasks)
                    for task in tasks:
                        counts["read_by_task"][task] += 1
                    segment_ids = [
                        tokenizer.encode(
                            sample.text[slice(*segment["char_span"])],
                            add_special_tokens=False,
                            truncation=False,
                        )
                        for segment in sample.segments
                    ]
                    body_ids = [token for segment in segment_ids for token in segment]
                    original_cut = (
                        len(body_ids) if view == "multisegment_full" else len(segment_ids[0])
                    )
                    cut = min(original_cut, max_input_tokens)
                    if cut < min_input_tokens:
                        counts["filtered_too_short"] += len(tasks)
                        continue
                    ids = body_ids + tokenizer.encode(
                        sample.continuation, add_special_tokens=False, truncation=False
                    )
                    if len(ids) < cut + continuation_tokens:
                        counts["filtered_continuation_length"] += len(tasks)
                        continue
                    input_ids = tuple(token_pool.setdefault(token, token) for token in ids[:cut])
                    continuation = tuple(
                        token_pool.setdefault(token, token)
                        for token in ids[cut : cut + continuation_tokens]
                    )
                    examples.extend(
                        PretrainExample(
                            f"{sample.trajectory_id}:{task}",
                            sample.document_id,
                            sample.dedup_cluster,
                            task,
                            input_ids,
                            input_ids if task == "ae" else continuation,
                        )
                        for task in tasks
                    )
                    original_lengths.append(original_cut)
                    actual_lengths.append(cut)
                    counts["kept_source_samples"] += 1
                    counts["cropped_source_samples"] += int(cut < original_cut)
                    counts["cropped_source_tokens"] += original_cut - cut
                    counts["kept"] += len(tasks)
                    for task in tasks:
                        counts["kept_by_task"][task] += 1
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
        # Prefix statistics count retained sources once; example statistics count AE/LM.
        for name, lengths in (
            ("original_source_prefix_tokens", original_lengths),
            ("actual_source_prefix_tokens", actual_lengths),
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
        "input_token_interval": [min_input_tokens, max_input_tokens],
        "splits": statistics,
    }
