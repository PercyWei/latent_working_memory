from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.data import Episode
from latent_working_memory.v1.sampling import capacity_weights, read_tokens


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
    config: ExperimentConfig,
    length_bounds: tuple[int, ...],
    root: Path,
) -> dict[str, Any]:
    registry = {}
    with (root / "sources.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["status"] == "eligible":
                registry[row["record"]["id"]] = row
    originals = {}
    with (directory / "documents.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            document_id = row["record"]["id"]
            registered = registry[document_id]
            if any(row[k] != registered[k] for k in ("record", "split", "cluster")):
                raise ValueError("document differs from the shared source registry")
            originals[document_id] = row
    seen_ids, cluster_splits, source_splits = set(), {}, {}
    lengths, counts = defaultdict(list), Counter()
    for split in ("train", "dev", "test"):
        with (directory / f"{split}.jsonl").open() as handle:
            for line in handle:
                episode = Episode.from_record(json.loads(line))
                source = episode.sources[0]
                original = originals[source.document_id]
                text, provenance = original["record"]["text"], source.provenance
                if provenance["boundary_variant"] != directory.name:
                    raise ValueError("sample belongs to another boundary variant")
                if provenance["dedup_cluster"] != original["cluster"]:
                    raise ValueError("sample cluster differs from its registered source")
                start, end = provenance["x_char_span"]
                if not 0 <= start < end <= len(text):
                    raise ValueError("invalid input character span")
                ae, lm = read_tokens(episode, tokenizer)
                if episode.episode_id in seen_ids:
                    raise ValueError("duplicate prepared episode ID")
                seen_ids.add(episode.episode_id)
                if (
                    original["split"] != split
                    or tuple(tokenizer.encode(text[start:end], add_special_tokens=False))
                    != episode.input_ids
                    or (ae is not None and text[start:end] != episode.reads[0].references[0].text)
                ):
                    raise ValueError("input source text or split mismatch")
                key = hashlib.blake2b(" ".join(text[start:end].split()).encode()).hexdigest()
                if provenance["input_text_key"] != key:
                    raise ValueError("input text key differs from original text")
                if provenance["quality_review"]["decision"] != "keep":
                    raise ValueError("retained sample lacks a keep quality decision")
                final_end = end
                if lm is not None:
                    y_start, final_end = provenance["y_char_span"]
                    if not end == y_start < final_end <= len(text) or (
                        text[y_start:final_end] != episode.reads[0].references[0].text
                    ):
                        raise ValueError("LM target must be the contiguous original continuation")
                    lengths[f"{split}/continuation/target"].append(len(lm.target_ids) - 1)
                elif provenance["y_char_span"] is not None:
                    raise ValueError("AE views must have no continuation span")
                if provenance["parent_char_span"] != [start, final_end]:
                    raise ValueError("AE/LM text must equal its parent sample span")
                for key, mapping in (
                    (original["cluster"], cluster_splits),
                    (source.source_id, source_splits),
                ):
                    if key in mapping and mapping[key] != split:
                        raise ValueError("source or duplicate cluster crosses data splits")
                    mapping[key] = split
                if not capacity_weights(config, len(episode.input_ids), ae, lm, 0):
                    raise ValueError("sample has no legal memory capacity")
                task, size = episode.reads[0].task, len(episode.input_ids)
                bucket = next(b for b in length_bounds if size <= b)
                lengths[f"{split}/input"].append(size)
                lengths[f"{split}/{task}/input"].append(size)
                lengths[f"{split}/granularity/{provenance['granularity']}"].append(size)
                counts[f"{split}/{task}/length_up_to/{bucket}"] += 1
                counts[f"{split}/{task}"] += 1
                if directory.name == "random":
                    counts[f"{split}/{task}/input_boundary_cut"] += not (
                        provenance["input_starts_at_sentence"]
                        and provenance["input_ends_at_sentence"]
                    )
    report = {
        "checks": {
            "original_text_continuity": True,
            "shared_source_assignments": True,
            "source_and_cluster_split_isolation": True,
            "legal_capacities": True,
            "model_keep_decisions": True,
        },
        "statistics": dict(counts),
        "lengths": {key: length_statistics(values) for key, values in sorted(lengths.items())},
    }
    (directory / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def compare_preparations(
    root: Path, tokenizer: PreTrainedTokenizerBase, config: ExperimentConfig, random_metadata: dict
) -> dict:
    metadata = {
        "semantic": json.loads((root / "semantic/preparation.json").read_text()),
        "random": random_metadata,
    }
    semantic, random_data = metadata["semantic"], metadata["random"]
    if (
        semantic["source_pool_id"] != random_data["source_pool_id"]
        or semantic["length_bounds"] != random_data["length_bounds"]
        or semantic["samples_per_task"] != random_data["samples_per_task"]
    ):
        raise ValueError("datasets must share source assignments, task quotas and length intervals")
    # Audit each persisted leaf against the shared registry, then compare aggregate distributions.
    audits = {
        v: audit_preparation(root / v, tokenizer, config, tuple(m["length_bounds"]), root)
        for v, m in metadata.items()
    }
    groups = {}
    for split, quota in zip(("train", "dev", "test"), semantic["samples_per_task"], strict=True):
        for task in ("ae", "continuation"):
            for variant in metadata:
                if audits[variant]["statistics"].get(f"{split}/{task}", 0) != quota:
                    raise ValueError("retained AE/LM sample quotas differ")
            a, b = (
                [
                    audits[v]["statistics"].get(f"{split}/{task}/length_up_to/{bound}", 0)
                    for bound in semantic["length_bounds"]
                ]
                for v in ("semantic", "random")
            )
            if a != b:
                raise ValueError("input length interval counts differ")
            groups[f"{split}/{task}"] = {
                "semantic_counts": a,
                "random_counts": b,
                "semantic_proportions": [n / quota for n in a],
                "random_proportions": [n / quota for n in b],
                "target_lengths": {
                    v: audits[v]["lengths"].get(f"{split}/{task}/target") for v in audits
                },
            }
    return {
        "checks": {
            "shared_source_split_isolation": True,
            "equal_task_quotas": True,
            "equal_input_length_interval_counts": True,
        },
        "length_bounds": semantic["length_bounds"],
        "groups": groups,
        "protocol": "independent sources and samples; compare post-review split/task/input-length intervals",
    }
