"""Freeze source clusters, splits and variable-length windows before allocating batches."""

from __future__ import annotations

import copy
import hashlib
import itertools
import math
import random
import uuid
from datetime import datetime

import pyarrow.parquet as pq
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from latent_working_memory.data_preparation.fineweb_qa.storage import save_json, load_json
from latent_working_memory.data_preparation.pretrain.config import PreparationConfig
from latent_working_memory.data_preparation.pretrain.dedup import (
    cluster_documents,
    matching_clusters,
)
from latent_working_memory.data_preparation.pretrain.fineweb import document_split
from latent_working_memory.data_preparation.pretrain.quality import document_rejection_reason
from latent_working_memory.data_preparation.pretrain.sources import (
    parquet_records,
    referenced_records,
)


def _rank(*parts: object) -> bytes:
    return hashlib.blake2b("\0".join(map(str, parts)).encode(), digest_size=16).digest()


def _old_pool_matches(
    files: list[Path],
    data_seed: int,
    old_pool_documents: int,
    new_records: list[dict],
    clusters: list[str],
    recipe: PreparationConfig,
) -> set[str]:
    with closing(parquet_records(files, data_seed)) as stream:

        def references():
            for _ in range(old_pool_documents):
                pair = next(stream, None)
                if pair is None:
                    raise ValueError("FineWeb ended before the old source pool was reconstructed")
                yield pair[0]

        return matching_clusters(references(), new_records, clusters, recipe)


def pool_after_exclusions(
    pool: dict, excluded_sources: list[dict], previous_datasets: list[str]
) -> dict:
    """Freeze a new run's unused candidates without rescanning an unchanged source recipe."""
    ids = {d["document_id"] for d in excluded_sources}
    clusters = {d["dedup_cluster"] for d in excluded_sources}
    result = copy.deepcopy(pool)
    documents = [
        d
        for d in result["documents"]
        if d["document_id"] not in ids and d["dedup_cluster"] not in clusters
    ]
    for index, document in enumerate(documents):
        document["pool_index"] = index
    counts = Counter(result["statistics"])
    counts["previously_used_documents"] += len(pool["documents"]) - len(documents)
    counts["frozen_documents"] = len(documents)
    result.update(
        pool_id=str(uuid.uuid4()),
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        documents=documents,
        excluded_sources=copy.deepcopy(excluded_sources),
        previous_datasets=list(previous_datasets),
        statistics=dict(counts),
        split_counts={
            split: sum(d["split"] == split for d in documents) for split in ("train", "dev", "test")
        },
        segment_counts=dict(Counter(str(len(d["segments"])) for d in documents)),
    )
    return result


def _window_cuts(text: str, document_id: str, seed: int, config: dict) -> list[int]:
    """Sample bounded character lengths and an arbitrary start without rejection."""
    rng = random.Random(_rank(seed, document_id))
    minimum, maximum = config["min_segment_chars"], config["max_segment_chars"]
    segment_count = rng.randint(
        config["min_segments"], min(config["max_segments"], len(text) // minimum)
    )
    remaining = min(len(text), segment_count * maximum)
    lengths = []
    for count in range(segment_count, 0, -1):
        length = rng.randint(minimum, min(maximum, remaining - (count - 1) * minimum))
        lengths.append(length)
        remaining -= length
    rng.shuffle(lengths)
    start = rng.randint(0, len(text) - sum(lengths))
    cuts = [start]
    for length in lengths:
        cuts.append(cuts[-1] + length)
    return cuts


def _validate_pool_config(config: dict) -> None:
    source, window = config["source"], config["window"]
    for key in ("file_count", "old_pool_documents", "scan_documents"):
        if type(source[key]) is not int or source[key] <= 0:
            raise ValueError(f"source.{key} must be a positive integer")
    if set(window) != {"min_segments", "max_segments", "min_segment_chars", "max_segment_chars"}:
        raise ValueError("window requires segment-count and character bounds")
    if any(type(v) is not int or v <= 0 for v in window.values()):
        raise ValueError("window bounds must be positive integers")
    if not 2 <= window["min_segments"] <= window["max_segments"]:
        raise ValueError("segment counts must be ordered and at least two")
    if window["min_segment_chars"] > window["max_segment_chars"]:
        raise ValueError("segment character bounds must be ordered")
    fractions = source["split_fractions"]
    if (
        len(fractions) != 3
        or any(type(v) not in (int, float) or v <= 0 for v in fractions)
        or not math.isclose(sum(fractions), 1.0)
    ):
        raise ValueError(
            "source.split_fractions must contain three positive fractions summing to one"
        )
    if set(config["batch_counts"]) != {"train", "dev", "test"} or any(
        type(v) is not int or v <= 0 for v in config["batch_counts"].values()
    ):
        raise ValueError("batch_counts requires positive train/dev/test counts")


def prepare_pool(config: dict, excluded_sources=(), previous_datasets=()) -> dict:
    """Build one immutable source pool; a different scan requires a different pool."""
    _validate_pool_config(config)
    path = Path(config["pool_dir"]) / "source-pool.json"
    contract = {key: config[key] for key in ("source", "window", "batch_counts")}
    if path.exists():
        pool = load_json(path)
        if (
            pool["config"] != contract
            or pool["excluded_sources"] != list(excluded_sources)
            or pool["previous_datasets"] != list(previous_datasets)
        ):
            raise ValueError("source pool configuration changed; create a new pool")
        return pool
    source, window = config["source"], config["window"]
    recipe = PreparationConfig(
        near_duplicate_threshold=source["near_duplicate_threshold"],
        near_duplicate_min_words=source["near_duplicate_min_words"],
    )
    files = [
        Path(source["raw_dir"]) / f"{i:03d}_00000.parquet" for i in range(source["file_count"])
    ]
    with closing(parquet_records(files, source["data_seed"])) as records:
        for _ in range(source["old_pool_documents"]):
            if next(records, None) is None:
                raise ValueError("source ended before the old pool")
        scanned = list(itertools.islice(records, source["scan_documents"]))
    raw = [record for record, _ in scanned]
    clusters = cluster_documents(raw, recipe)
    excluded = _old_pool_matches(
        files, source["data_seed"], source["old_pool_documents"], raw, clusters, recipe
    )
    used_clusters = (
        matching_clusters(referenced_records(excluded_sources), raw, clusters, recipe)
        if excluded_sources
        else set()
    )
    counts = Counter(
        scanned_documents=len(scanned), old_pool_documents=source["old_pool_documents"]
    )
    candidates, seen = [], set()
    for (record, location), cluster in zip(scanned, clusters, strict=True):
        if document_rejection_reason(record, 64) is not None:
            counts["basic_rejected_documents"] += 1
        elif cluster in excluded:
            counts["old_pool_duplicate_documents"] += 1
        elif cluster in used_clusters:
            counts["previously_used_documents"] += 1
        elif cluster in seen:
            counts["within_scan_duplicate_documents"] += 1
        else:
            seen.add(cluster)
            candidates.append((record, location, cluster))
    candidates.sort(
        key=lambda item: (_rank(source["selection_seed"], item[0]["id"]), item[0]["id"])
    )
    documents = []
    by_split = Counter({split: 0 for split in ("train", "dev", "test")})
    for record, location, cluster in candidates:
        text = record["text"]
        if len(text) < window["min_segments"] * window["min_segment_chars"]:
            counts["length_failed"] += 1
            continue
        split = document_split(cluster, source["data_seed"], tuple(source["split_fractions"]))
        cuts = _window_cuts(text, record["id"], source["selection_seed"], window)
        start, end = cuts[0], cuts[-1]
        documents.append(
            {
                "pool_index": len(documents),
                "trajectory_id": f"{record['id']}:{start}:{end}",
                "document_id": record["id"],
                "dedup_cluster": cluster,
                "split": split,
                "source": {
                    "file": location["source_file"],
                    "row_group": location["row_group"],
                    "row_index": location["row_index"],
                },
                "window_char_span": [start, end],
                "segments": [
                    {"segment_id": f"seg{i}", "char_span": [left - start, right - start]}
                    for i, (left, right) in enumerate(zip(cuts, cuts[1:]))
                ],
            }
        )
        by_split[split] += 1
    counts["frozen_documents"] = len(documents)
    pool = {
        "pool_id": str(uuid.uuid4()),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": contract,
        "excluded_sources": list(excluded_sources),
        "previous_datasets": list(previous_datasets),
        "statistics": dict(counts),
        "split_counts": dict(by_split),
        "segment_counts": dict(Counter(str(len(d["segments"])) for d in documents)),
        "documents": documents,
    }
    save_json(path, pool)
    return pool


def batch_ranges(pool: dict, offsets: dict, index: int) -> dict:
    """Locate a run-local batch in the immutable per-split source lists."""
    if type(index) is not int or index < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    if set(offsets) != {"train", "dev", "test"}:
        raise ValueError("source_offsets must contain train, dev and test")
    ranges = {}
    for split in ("train", "dev", "test"):
        offset, available = offsets[split], pool["split_counts"][split]
        if type(offset) is not int or not 0 <= offset <= available:
            raise ValueError(f"source_offsets.{split} must be within the frozen source pool")
        count = pool["config"]["batch_counts"][split]
        start = min(offset + index * count, available)
        stop = min(offset + (index + 1) * count, available)
        ranges[split] = {"start": start, "stop": stop, "requested": count, "selected": stop - start}
    return ranges


def prepare_selection(config: dict) -> dict:
    """Allocate disjoint split slices by batch index; never recluster during annotation."""
    pool = load_json(Path(config["source_pool_dir"]) / "source-pool.json")
    index = config["batch_index"]
    ranges = batch_ranges(pool, config["source_offsets"], index)
    selected = []
    for split, bounds in ranges.items():
        candidates = [d for d in pool["documents"] if d["split"] == split]
        selected.extend(dict(d) for d in candidates[bounds["start"] : bounds["stop"]])
    if not selected:
        raise ValueError(f"source pool exhausted in batch {index}")
    grouped = defaultdict(list)
    for doc in selected:
        grouped[(doc["source"]["file"], doc["source"]["row_group"])].append(doc)
    for (filename, group), docs in grouped.items():
        with pq.ParquetFile(filename) as parquet:
            table = parquet.read_row_group(group, columns=["id", "text"])
        for doc in docs:
            row = table.slice(doc["source"]["row_index"], 1).to_pylist()[0]
            if row["id"] != doc["document_id"]:
                raise ValueError("source document differs from frozen pool")
            doc["text"] = row["text"][slice(*doc["window_char_span"])]
            if len(doc["text"]) != doc["window_char_span"][1] - doc["window_char_span"][0]:
                raise ValueError("source text is shorter than its frozen window")
    return {
        "source_pool_id": pool["pool_id"],
        "source_pool_config": pool["config"],
        "batch_index": index,
        "ranges": ranges,
        "documents": selected,
        "statistics": {
            "frozen_documents": len(selected),
            "selected_by_split": {key: value["selected"] for key, value in ranges.items()},
            "available_by_split": {
                split: count - config["source_offsets"][split]
                for split, count in pool["split_counts"].items()
            },
        },
    }
