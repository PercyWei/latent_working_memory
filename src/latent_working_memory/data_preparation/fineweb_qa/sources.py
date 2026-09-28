from __future__ import annotations

import hashlib
import itertools
import json
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

from latent_working_memory.data_preparation.pretrain.config import PreparationConfig
from latent_working_memory.data_preparation.pretrain.dedup import cluster_documents
from latent_working_memory.data_preparation.pretrain.fineweb import document_split
from latent_working_memory.data_preparation.pretrain.quality import document_rejection_reason
from latent_working_memory.data_preparation.pretrain.segmentation import sentence_spans
from latent_working_memory.data_preparation.pretrain.sources import parquet_records


def _order_key(seed: int, *parts: object) -> bytes:
    value = json.dumps([seed, *parts], ensure_ascii=False, separators=(",", ":"))
    return hashlib.blake2b(value.encode(), digest_size=16).digest()


def _compact_record(record: dict[str, Any]) -> dict[str, str]:
    return {name: record[name] for name in ("id", "url", "text")}


def select_window(text: str, document_id: str, seed: int, settings: dict) -> dict | None:
    """Choose one unchanged, contiguous window, using paragraph/sentence boundaries."""
    minimum = settings["min_segment_chars"]
    target = settings["target_segment_chars"]
    maximum = settings["max_segment_chars"]
    count = settings["segments"]
    first = len(text) - len(text.lstrip())
    final = len(text.rstrip())
    if final - first < settings["min_trajectory_chars"]:
        return None

    # FineWeb uses line breaks for paragraphs. Keep intervening whitespace in the slices.
    starts = {first}
    for match in re.finditer(r"\n[ \t\r\n]*", text):
        if match.end() < final:
            starts.add(match.end())
    paragraph_ends = sorted((starts - {first}) | {final})
    sentence_ends: list[int] | None = None
    windows = []
    for start in sorted(starts):
        if final - start < settings["min_trajectory_chars"]:
            break
        boundaries = [start]
        for _ in range(count):
            position = boundaries[-1]
            lower, upper = position + minimum, position + maximum
            left = bisect_left(paragraph_ends, lower)
            right = bisect_right(paragraph_ends, upper)
            choices = paragraph_ends[left:right]
            if not choices:
                if sentence_ends is None:
                    sentence_ends = []
                    for sentence in sentence_spans(text):
                        end = sentence.end
                        while end < final and text[end].isspace():
                            end += 1
                        sentence_ends.append(end)
                    sentence_ends = sorted(set(sentence_ends))
                left = bisect_left(sentence_ends, lower)
                right = bisect_right(sentence_ends, upper)
                choices = sentence_ends[left:right]
            if not choices:
                break
            boundaries.append(min(choices, key=lambda end: (abs(end - position - target), end)))
        if len(boundaries) != count + 1:
            continue
        size = boundaries[-1] - start
        if settings["min_trajectory_chars"] <= size <= settings["max_trajectory_chars"]:
            windows.append(boundaries)
    if not windows:
        return None
    boundaries = min(
        windows,
        key=lambda values: _order_key(seed, document_id, values[0], values[-1]),
    )
    start, end = boundaries[0], boundaries[-1]
    return {
        "char_start": start,
        "char_end": end,
        "text": text[start:end],
        "segments": [
            {"segment_id": i + 1, "char_start": a - start, "char_end": b - start}
            for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True))
        ],
    }


def _validate_settings(source: dict, text: dict) -> None:
    for name in (
        "scan_document_limit",
        "candidate_document_limit",
        "trajectory_limit",
        "dedup_batch_documents",
    ):
        if type(source[name]) is not int or source[name] <= 0:
            raise ValueError(f"source.{name} must be a positive integer")
    if type(source["seed"]) is not int or source["seed"] < 0:
        raise ValueError("source.seed must be a non-negative integer")
    if source["split"] not in ("train", "dev", "test"):
        raise ValueError("source.split must be train, dev or test")
    for name in (
        "segments",
        "min_segment_chars",
        "target_segment_chars",
        "max_segment_chars",
        "min_trajectory_chars",
        "max_trajectory_chars",
    ):
        if type(text[name]) is not int or text[name] <= 0:
            raise ValueError(f"text.{name} must be a positive integer")
    if not text["min_segment_chars"] <= text["target_segment_chars"] <= text["max_segment_chars"]:
        raise ValueError("segment character bounds must contain the target")
    if not text["min_trajectory_chars"] <= text["max_trajectory_chars"]:
        raise ValueError("invalid trajectory character bounds")


def prepare_sources(config: dict) -> dict:
    """Exclude the old candidate pool and freeze a bounded local QA document sample.

    Two streaming reads avoid retaining the old pool in memory: first collect the
    bounded continuation, then compare its clusters against batches of old records.
    Every old record is excluded, so old-to-old cluster edges need not be materialized.
    """
    source, text_settings = config["source"], config["text"]
    _validate_settings(source, text_settings)
    report_path = Path(source["source_report"])
    report = json.loads(report_path.read_text())
    source_seed = report["source_seed"]
    old_count = report["source_pool_candidates"]
    files = [Path(source["raw_dir"]) / Path(path).name for path in report["source_files"]]
    recipe = PreparationConfig()
    split_fractions = (0.9, 0.05, 0.05)

    located = []
    with closing(parquet_records(files, source_seed)) as records:
        skipped = sum(1 for _ in itertools.islice(records, old_count))
        if skipped != old_count:
            raise ValueError("source ends before the complete old candidate pool")
        for record, location in itertools.islice(records, source["scan_document_limit"]):
            located.append((_compact_record(record), location))
    candidates = [record for record, _ in located]
    print(
        f"Collected {len(candidates)} new source documents after excluding the old {old_count}",
        flush=True,
    )
    reasons = [
        document_rejection_reason(record, recipe.min_document_chars) for record in candidates
    ]
    clusters = cluster_documents(candidates, recipe)

    excluded_clusters = set()
    if candidates:
        with closing(parquet_records(files, source_seed)) as records:
            remaining = old_count
            while remaining:
                batch = [
                    _compact_record(record)
                    for record, _ in itertools.islice(
                        records, min(remaining, source["dedup_batch_documents"])
                    )
                ]
                if not batch:
                    raise ValueError("source ends before the complete old candidate pool")
                combined = cluster_documents([*candidates, *batch], recipe)
                old_clusters = set(combined[len(candidates) :])
                excluded_clusters.update(
                    cluster
                    for cluster, joint in zip(clusters, combined[: len(candidates)], strict=True)
                    if joint in old_clusters
                )
                remaining -= len(batch)
                print(
                    f"Compared old source pool: {old_count - remaining}/{old_count}; "
                    f"excluded new clusters: {len(excluded_clusters)}",
                    flush=True,
                )

    # Choose a representative after exclusions and basic checks, not by QA outcomes.
    representatives = {}
    for i in sorted(
        range(len(candidates)),
        key=lambda i: _order_key(source["seed"], candidates[i]["id"]),
    ):
        if reasons[i] is None and clusters[i] not in excluded_clusters:
            representatives.setdefault(clusters[i], i)
    split_counts = Counter(
        document_split(cluster, source_seed, split_fractions) for cluster in representatives
    )
    selected = [
        i
        for cluster, i in representatives.items()
        if document_split(cluster, source_seed, split_fractions) == source["split"]
    ][: source["candidate_document_limit"]]

    trajectories = []
    window_failures: Counter[str] = Counter()
    checked = 0
    for i in selected:
        checked += 1
        record, location = located[i]
        window = select_window(record["text"], record["id"], source["seed"], text_settings)
        if window is None:
            reason = (
                "too_short"
                if len(record["text"].strip()) < text_settings["min_trajectory_chars"]
                else "no_legal_window"
            )
            window_failures[reason] += 1
            continue
        identity = _order_key(
            source["seed"], record["id"], window["char_start"], window["char_end"]
        ).hex()
        trajectories.append(
            {
                "trajectory_id": f"fineweb-qa-{identity}",
                "document_id": record["id"],
                "dedup_cluster": clusters[i],
                "split": source["split"],
                "source": {
                    **location,
                    "url": record["url"],
                    "window_char_start": window["char_start"],
                    "window_char_end": window["char_end"],
                },
                "text": window["text"],
                "segments": window["segments"],
            }
        )
        if len(trajectories) == source["trajectory_limit"]:
            break

    return {
        "trajectories": trajectories,
        "statistics": {
            "old_pool_documents_excluded": old_count,
            "new_documents_scanned": len(candidates),
            "new_clusters": len(set(clusters)),
            "new_clusters_matching_old_pool": len(excluded_clusters),
            "new_documents_matching_old_pool": sum(c in excluded_clusters for c in clusters),
            "basic_rejections": dict(Counter(reason for reason in reasons if reason is not None)),
            "eligible_clusters_by_split": dict(split_counts),
            "candidate_documents_selected": len(selected),
            "window_documents_checked": checked,
            "window_failures": dict(window_failures),
            "frozen_trajectories": len(trajectories),
            "trajectory_shortfall": source["trajectory_limit"] - len(trajectories),
        },
        "source_provenance": {
            "source_report": str(report_path),
            "source_files": [str(path) for path in files],
            "source_seed": source_seed,
            "split_fractions": list(split_fractions),
            "selection_seed": source["seed"],
            "old_candidate_pool_size": old_count,
            "exclusion_scope": "entire_old_candidate_pool_and_matching_new_clusters",
            "exclusion_reason": "actual_prepared_document_ids_unavailable_locally",
            "scan_document_limit": source["scan_document_limit"],
            "candidate_document_limit": source["candidate_document_limit"],
            "dedup_batch_documents": source["dedup_batch_documents"],
            "near_duplicate_threshold": recipe.near_duplicate_threshold,
            "near_duplicate_min_words": recipe.near_duplicate_min_words,
            "boundary_method": "paragraph_then_pysbd_conservative",
            "length_unit": "python_characters",
        },
    }
