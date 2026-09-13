from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from latent_working_memory.data_preparation.pretrain.sources import load_sources
from latent_working_memory.data_preparation.pretrain.text_samples import TextSample
from latent_working_memory.data_preparation.pretrain.dedup import source_key
from latent_working_memory.data_preparation.pretrain.config import DataConfig, PreparationConfig
from latent_working_memory.data_preparation.pretrain.segmentation import sentence_spans


def length_statistics(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"samples": 0, "tokens": 0}
    ordered = sorted(values)
    result = {
        "samples": len(values),
        "tokens": sum(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(values) / len(values),
    }
    for percentile in (50, 90, 95, 99):
        position = (len(ordered) - 1) * percentile / 100
        low, high = math.floor(position), math.ceil(position)
        result[f"p{percentile}"] = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return result


def audit_preparation(
    directory: Path,
    tokenizer: PreTrainedTokenizerBase,
    config: DataConfig,
    preparation: PreparationConfig,
    root: Path,
    sources: list[dict] | None = None,
) -> dict[str, Any]:
    if sources is None:
        sources = load_sources(root / "source-pool.json")
    registry = {row["record"]["id"]: row for row in sources if row["status"] == "eligible"}
    boundaries = {}
    seen_ids, cluster_splits, source_splits = set(), {}, {}
    lengths, counts = defaultdict(list), Counter()
    for split in ("train", "dev", "test"):
        with (directory / f"{split}.jsonl").open() as handle:
            for line in handle:
                sample = TextSample(**json.loads(line))
                original = registry[sample.document_id]
                text = original["record"]["text"]
                expected_method = (
                    "pysbd_conservative" if directory.name == "semantic" else "random_token"
                )
                if sample.boundary_method != expected_method or sample.source_id != source_key(
                    original["record"]["url"]
                ):
                    raise ValueError("sample boundary method or source identity mismatch")
                if sample.dedup_cluster != original["cluster"]:
                    raise ValueError("sample cluster differs from its registered source")
                start, end = sample.x_char_span
                if not 0 <= start < end <= len(text):
                    raise ValueError("invalid input character span")
                task = sample.task
                target_length = len(
                    tokenizer.encode(
                        sample.continuation if task == "continuation" else sample.text,
                        add_special_tokens=False,
                    )
                )
                size = len(tokenizer.encode(sample.text, add_special_tokens=False))
                if (size, target_length) != (
                    sample.reference_input_tokens,
                    sample.reference_target_tokens,
                ):
                    raise ValueError("reference tokens differ from construction tokenizer")
                if sample.sample_id in seen_ids:
                    raise ValueError("duplicate prepared sample ID")
                seen_ids.add(sample.sample_id)
                if original["split"] != split or text[start:end] != sample.text:
                    raise ValueError("input source text or split mismatch")
                final_end = end
                if task == "continuation":
                    y_start, final_end = sample.y_char_span
                    if not end == y_start < final_end <= len(text) or (
                        text[y_start:final_end] != sample.continuation
                    ):
                        raise ValueError("LM target must be the contiguous original continuation")
                    lengths[f"{split}/continuation/target"].append(target_length)
                elif sample.y_char_span is not None:
                    raise ValueError("AE views must have no continuation span")
                if directory.name == "semantic":
                    if sample.document_id not in boundaries:
                        spans = sentence_spans(text)
                        boundaries[sample.document_id] = (
                            {s.start for s in spans},
                            {s.end for s in spans},
                        )
                    starts, ends = boundaries[sample.document_id]
                    if start not in starts or end not in ends or final_end not in ends:
                        raise ValueError(
                            "semantic endpoints differ from the construction boundary rules"
                        )
                for key, mapping in (
                    (original["cluster"], cluster_splits),
                    (sample.source_id, source_splits),
                ):
                    if key in mapping and mapping[key] != split:
                        raise ValueError("source or duplicate cluster crosses data splits")
                    mapping[key] = split
                if not preparation.accepts_lengths(
                    size, target_length if task == "continuation" else None
                ):
                    raise ValueError(
                        "sample lengths or LM prefix fraction violate the preparation recipe"
                    )
                bucket = next(b for b in preparation.length_bounds if size <= b)
                lengths[f"{split}/input"].append(size)
                lengths[f"{split}/{task}/input"].append(size)
                counts[f"{split}/{task}/length_up_to/{bucket}"] += 1
                counts[f"{split}/{task}"] += 1
    report = {
        "checks": {
            "original_text_continuity": True,
            "shared_source_assignments": True,
            "source_and_cluster_split_isolation": True,
            "sample_lengths_and_lm_fraction": True,
            **({"semantic_rule_endpoints": True} if directory.name == "semantic" else {}),
        },
        "statistics": dict(counts),
        "lengths": {key: length_statistics(values) for key, values in sorted(lengths.items())},
    }
    return report


def compare_preparations(root: Path, random_metadata: dict) -> dict:
    metadata = {
        "semantic": json.loads((root / "semantic/preparation.json").read_text()),
        "random": random_metadata,
    }
    semantic, random_data = metadata["semantic"], metadata["random"]
    if (
        semantic["source_pool_id"] != random_data["source_pool_id"]
        or semantic["tokenizer"] != random_data["tokenizer"]
        or semantic["recipe"]["length_bounds"] != random_data["recipe"]["length_bounds"]
        or semantic["recipe"]["samples_per_task"] != random_data["recipe"]["samples_per_task"]
    ):
        raise ValueError("datasets must share source assignments, task quotas and length intervals")
    # Compare recorded counts; original text was checked before completing each variant.
    recipes = {
        v: PreparationConfig(
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in m["recipe"].items()
            }
        )
        for v, m in metadata.items()
    }
    if any(
        getattr(recipes["semantic"], key) != getattr(recipes["random"], key)
        for key in ("min_sample_tokens", "max_sample_tokens", "lm_prefix_fraction")
    ):
        raise ValueError("datasets must share sample length and LM fraction constraints")
    targets = recipes["semantic"].balanced_histogram()
    groups = {}
    for split, quota in zip(
        ("train", "dev", "test"), semantic["recipe"]["samples_per_task"], strict=True
    ):
        for task in ("ae", "continuation"):
            for variant in metadata:
                if metadata[variant]["statistics"].get(f"{split}/{task}", 0) != quota:
                    raise ValueError("retained AE/LM sample quotas differ")
            a, b = (
                [
                    metadata[v]["input_histogram"][split][task][str(bound)]
                    for bound in semantic["recipe"]["length_bounds"]
                ]
                for v in ("semantic", "random")
            )
            expected = [
                targets[split][task][str(bound)] for bound in semantic["recipe"]["length_bounds"]
            ]
            if a != b or a != expected:
                raise ValueError("input length interval counts differ from balanced quotas")
            groups[f"{split}/{task}"] = {
                "semantic_counts": a,
                "random_counts": b,
                "semantic_proportions": [n / quota for n in a],
                "random_proportions": [n / quota for n in b],
                "target_lengths": {
                    v: metadata[v]["lengths"].get(f"{split}/{task}/target") for v in metadata
                },
            }
    return {
        "checks": {
            "shared_source_split_isolation": True,
            "equal_task_quotas": True,
            "equal_input_length_interval_counts": True,
            "balanced_input_length_intervals": True,
        },
        "length_bounds": semantic["recipe"]["length_bounds"],
        "groups": groups,
        "protocol": "independent sources and samples; compare split/task/input-length intervals",
    }
