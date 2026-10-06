"""Tokenize canonical FineWeb AE/continuation samples for v3 pretraining."""

from dataclasses import dataclass
import json
from pathlib import Path

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
