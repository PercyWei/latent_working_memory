from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.data import Episode, EpisodeIndex
from latent_working_memory.v1.sampling import PretrainSampler, capacity_weights, read_tokens


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
) -> dict[str, Any]:
    originals = {}
    with (directory / "documents.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            originals[row["record"]["id"]] = row
    seen_ids, cluster_splits, source_splits = set(), {}, {}
    lengths = defaultdict(list)
    counts = Counter()
    cells = defaultdict(list)
    for split in ("train", "dev", "test"):
        with (directory / f"{split}.jsonl").open() as handle:
            for line in handle:
                episode = Episode.from_record(json.loads(line))
                source = episode.sources[0]
                original = originals[source.document_id]
                text = original["record"]["text"]
                provenance = source.provenance
                if provenance["dedup_cluster"] != original["cluster"]:
                    raise ValueError("view duplicate cluster differs from its source record")
                start, end = provenance["x_char_span"]
                ae, lm = read_tokens(episode, tokenizer)
                if episode.episode_id in seen_ids:
                    raise ValueError("duplicate prepared episode ID")
                seen_ids.add(episode.episode_id)
                if (
                    original["split"] != split
                    or text[start:end] != episode.reads[0].references[0].text
                ):
                    raise ValueError("AE source text or split mismatch")
                final_end = end
                if lm is not None:
                    y_start, final_end = provenance["y_char_span"]
                    if (
                        y_start != end
                        or text[y_start:final_end] != episode.reads[1].references[0].text
                    ):
                        raise ValueError("LM target must be the contiguous original continuation")
                    lengths[f"{split}/continuation"].append(len(lm.target_ids) - 1)
                    counts[f"{split}/lm_views"] += 1
                elif provenance["y_char_span"] is not None:
                    raise ValueError("AE-only views must have no continuation span")
                if any(start < b and a < final_end for a, b in original["excluded_spans"]):
                    raise ValueError("prepared view overlaps an excluded original span")
                for key, mapping in (
                    (original["cluster"], cluster_splits),
                    (source.source_id, source_splits),
                ):
                    if key in mapping and mapping[key] != split:
                        raise ValueError("source or duplicate cluster crosses data splits")
                    mapping[key] = split
                if not capacity_weights(config, len(episode.input_ids), ae, lm, 0):
                    raise ValueError("prepared view has no legal memory capacity")
                size = len(episode.input_ids)
                bucket = next((b for b in config.input_length_bounds if size <= b), size)
                granularity = provenance["granularity"]
                lengths[f"{split}/input"].append(size)
                lengths[f"{split}/granularity/{granularity}"].append(size)
                counts[f"{split}/length_up_to/{bucket}/views"] += 1
                counts[f"{split}/length_up_to/{bucket}/input_tokens"] += size
                counts[f"{split}/ae_views"] += 1
                cells[(split, granularity, bucket)].append(
                    {
                        "kind": "view",
                        "split": split,
                        "document_id": source.document_id,
                        "episode_id": episode.episode_id,
                        "granularity": granularity,
                        "input_tokens": size,
                        "input": text[start:end],
                        "continuation": text[end:final_end] if lm else None,
                        "x_char_span": [start, end],
                        "y_char_span": provenance["y_char_span"],
                    }
                )
    rng = random.Random(config.data_seed + 17)
    for rows in cells.values():
        rng.shuffle(rows)
    keys = sorted(cells)
    rng.shuffle(keys)
    panel, seen_documents = [], set()
    while keys and len(panel) < preparation.audit_examples:
        active = []
        for key in keys:
            rows = cells[key]
            while rows and rows[-1]["document_id"] in seen_documents:
                rows.pop()
            if rows:
                row = rows.pop()
                panel.append(row)
                seen_documents.add(row["document_id"])
                active.append(key)
            if len(panel) == preparation.audit_examples:
                break
        keys = active
    with (directory / "audit-stratified-views.jsonl").open("w") as handle:
        for row in panel:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    decisions = [
        json.loads(line)
        for line in (directory / "document-decisions.jsonl").read_text().splitlines()
    ]
    populations = defaultdict(list)
    for i, row in enumerate(decisions):
        if row["status"] in {"accepted", "quality_rejected"}:
            populations[row["status"]].append(i)
    selected = set()
    for rows in populations.values():
        selected.update(rng.sample(rows, min(len(rows), preparation.audit_examples)))
    with (
        (directory / "candidates.jsonl").open() as source,
        (directory / "audit-random-documents.jsonl").open("w") as target,
    ):
        for i, line in enumerate(source):
            if i in selected:
                target.write(
                    json.dumps(
                        decisions[i] | {"record": json.loads(line), "reviewer": None},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    report = {
        "checks": {
            "original_text_continuity": True,
            "excluded_spans_absent": True,
            "source_and_cluster_split_isolation": True,
            "legal_capacities": True,
        },
        "statistics": dict(counts),
        "lengths": {key: length_statistics(values) for key, values in sorted(lengths.items())},
        "quality_audit": {
            "random_populations": {key: len(value) for key, value in populations.items()},
            "random_documents": len(selected),
            "stratified_views": len(panel),
            "status": "pending_review",
            "protocol": "random accepted/rejected documents estimate error rates after review; stratified views diagnose issues",
        },
    }
    if counts["train/ae_views"]:
        index = EpisodeIndex(directory / "train.jsonl")
        sampler = PretrainSampler(index, tokenizer, config)
        exposures = defaultdict(Counter)
        seen_by_length = defaultdict(set)
        for visit in range(1000):
            example = sampler.sample(
                visit // (config.batch_size * config.gradient_accumulation_steps)
            )
            bucket = next(
                (b for b in config.input_length_bounds if example.input_length <= b),
                example.input_length,
            )
            row = exposures[str(bucket)]
            row["samples"] += 1
            seen_by_length[str(bucket)].add(example.episode.sources[0].document_id)
            row["input_tokens"] += example.input_length
            row["ae_target_tokens"] += len(example.ae.target_ids) - 1
            row["lm_target_tokens"] += len(example.lm.target_ids) - 1 if example.lm else 0
            row[f"capacity/{example.capacity}"] += 1
            row["effective_ratio_sum"] += example.input_length / example.capacity
            row[f"granularity/{example.episode.sources[0].provenance['granularity']}"] += 1
        for bucket, row in exposures.items():
            row["distinct_documents"] = len(seen_by_length[bucket])
            row["effective_ratio_mean"] = row.pop("effective_ratio_sum") / row["samples"]
        report["sampling_preview"] = {
            "visits": 1000,
            "length_groups": dict(exposures),
            "protocol": "deterministic sampler dry run; no model training",
        }
    (directory / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
