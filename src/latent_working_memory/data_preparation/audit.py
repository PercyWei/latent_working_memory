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
from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.data_preparation.segmentation import sentence_spans


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
    preparation: PreparationConfig,
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
    boundaries = {}
    if directory.name == "semantic":
        for key, original in originals.items():
            spans = sentence_spans(original["record"]["text"])
            boundaries[key] = ({s.start for s in spans}, {s.end for s in spans})
    seen_ids, cluster_splits, source_splits = set(), {}, {}
    lengths, counts = defaultdict(list), Counter()
    composition = defaultdict(lambda: defaultdict(Counter))
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
                task = episode.reads[0].task
                if (
                    provenance["tokenizer_name_or_path"] != config.model_name_or_path
                    or provenance["tokenizer_revision"] != config.model_revision
                    or provenance["dataset"] != config.pretrain_dataset
                    or provenance["subset"] != config.pretrain_subset
                    or episode.reads[0].prompt
                    != (config.ae_prompt if task == "ae" else config.lm_prompt)
                ):
                    raise ValueError(
                        "sample tokenizer, dataset or task prompt differs from its contract"
                    )
                target_length = len(
                    tokenizer.encode(episode.reads[0].references[0].text, add_special_tokens=False)
                )
                if episode.episode_id in seen_ids:
                    raise ValueError("duplicate prepared episode ID")
                seen_ids.add(episode.episode_id)
                if (
                    original["split"] != split
                    or tuple(tokenizer.encode(text[start:end], add_special_tokens=False))
                    != episode.input_ids
                    or (task == "ae" and text[start:end] != episode.reads[0].references[0].text)
                ):
                    raise ValueError("input source text or split mismatch")
                key = hashlib.blake2b(" ".join(text[start:end].split()).encode()).hexdigest()
                if provenance["input_text_key"] != key:
                    raise ValueError("input text key differs from original text")
                final_end = end
                if task == "continuation":
                    y_start, final_end = provenance["y_char_span"]
                    if not end == y_start < final_end <= len(text) or (
                        text[y_start:final_end] != episode.reads[0].references[0].text
                    ):
                        raise ValueError("LM target must be the contiguous original continuation")
                    lengths[f"{split}/continuation/target"].append(target_length)
                elif provenance["y_char_span"] is not None:
                    raise ValueError("AE views must have no continuation span")
                if directory.name == "semantic":
                    starts, ends = boundaries[source.document_id]
                    if start not in starts or end not in ends or final_end not in ends:
                        raise ValueError(
                            "semantic endpoints differ from the construction boundary rules"
                        )
                if provenance["parent_char_span"] != [start, final_end]:
                    raise ValueError("AE/LM text must equal its parent sample span")
                for key, mapping in (
                    (original["cluster"], cluster_splits),
                    (source.source_id, source_splits),
                ):
                    if key in mapping and mapping[key] != split:
                        raise ValueError("source or duplicate cluster crosses data splits")
                    mapping[key] = split
                if not preparation.accepts_lengths(
                    len(episode.input_ids), target_length if task == "continuation" else None
                ):
                    raise ValueError(
                        "sample lengths or LM prefix fraction violate the preparation recipe"
                    )
                task, size = episode.reads[0].task, len(episode.input_ids)
                bucket = next(b for b in preparation.length_bounds if size <= b)
                lengths[f"{split}/input"].append(size)
                lengths[f"{split}/{task}/input"].append(size)
                lengths[f"{split}/granularity/{provenance['granularity']}"].append(size)
                counts[f"{split}/{task}/length_up_to/{bucket}"] += 1
                counts[f"{split}/{task}"] += 1
                for group in (f"{split}/{task}", f"{split}/{task}/length_up_to/{bucket}"):
                    cell = composition[group][provenance["granularity"]]
                    cell["samples"] += 1
                    cell["input_tokens"] += size
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
            "sample_lengths_and_lm_fraction": True,
            **({"semantic_rule_endpoints": True} if directory.name == "semantic" else {}),
        },
        "statistics": dict(counts),
        "composition": {
            group: {
                granularity: dict(cell)
                | {
                    "sample_fraction": cell["samples"] / sum(c["samples"] for c in cells.values()),
                    "input_token_fraction": cell["input_tokens"]
                    / sum(c["input_tokens"] for c in cells.values()),
                }
                for granularity, cell in sorted(cells.items())
            }
            for group, cells in sorted(composition.items())
        },
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
        or semantic["contract"] != random_data["contract"]
        or semantic["length_bounds"] != random_data["length_bounds"]
        or semantic["samples_per_task"] != random_data["samples_per_task"]
    ):
        raise ValueError("datasets must share source assignments, task quotas and length intervals")
    # Audit each persisted leaf against the shared registry, then compare aggregate distributions.
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
    audits = {v: audit_preparation(root / v, tokenizer, config, recipes[v], root) for v in metadata}
    targets = recipes["semantic"].balanced_histogram()
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
            expected = [targets[split][task][str(bound)] for bound in semantic["length_bounds"]]
            if a != b or a != expected:
                raise ValueError("input length interval counts differ from balanced quotas")
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
            "balanced_input_length_intervals": True,
        },
        "length_bounds": semantic["length_bounds"],
        "groups": groups,
        "protocol": "independent sources and samples; compare split/task/input-length intervals",
    }
