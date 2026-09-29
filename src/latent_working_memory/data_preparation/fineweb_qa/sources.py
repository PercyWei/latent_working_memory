"""Freeze source clusters, splits and variable-length windows before allocating batches."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import random
import uuid
from datetime import datetime

import pyarrow.parquet as pq
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from latent_working_memory.data_preparation.fineweb_qa.storage import save_json, load_json
from latent_working_memory.data_preparation.pretrain.config import PreparationConfig
from latent_working_memory.data_preparation.pretrain.dedup import cluster_documents, source_key
from latent_working_memory.data_preparation.pretrain.fineweb import document_split
from latent_working_memory.data_preparation.pretrain.quality import document_rejection_reason
from latent_working_memory.data_preparation.pretrain.sources import parquet_records


_WORDS = re.compile(r"\w+")
_OLD_BATCH_SIZE = 512


def _rank(*parts: object) -> bytes:
    return hashlib.blake2b("\0".join(map(str, parts)).encode(), digest_size=16).digest()


def _normalized_text(record: Mapping[str, Any]) -> str:
    return " ".join(record["text"].split())


def _shingles(text: str, min_words: int) -> set[bytes]:
    words = _WORDS.findall(text.casefold())
    if len(words) < min_words:
        return set()
    return {
        hashlib.blake2b(" ".join(words[i : i + 5]).encode(), digest_size=8).digest()
        for i in range(len(words) - 4)
    }


def _prefix(values: set[bytes], frequency: Counter[bytes], threshold: float) -> list[bytes]:
    ordered = sorted(values, key=lambda token: (frequency[token], token))
    return ordered[: len(values) - math.ceil(threshold * len(values)) + 1]


def _old_pool_matches(
    files: list[Path],
    data_seed: int,
    old_pool_documents: int,
    new_records: list[dict[str, Any]],
    clusters: list[str],
    recipe: PreparationConfig,
) -> set[str]:
    """Match old documents in bounded batches, then exclude entire new clusters.

    The prefix index uses the same normalized word 5-grams and exact Jaccard
    threshold as ``cluster_documents``. All new documents are clustered first,
    so an old match also excludes new documents joined transitively to it.
    """
    if not new_records:
        return set()

    ids: dict[str, set[int]] = defaultdict(set)
    urls: dict[str, set[int]] = defaultdict(set)
    texts: dict[str, set[int]] = defaultdict(set)
    shingle_sets = []
    for index, record in enumerate(new_records):
        normalized = _normalized_text(record)
        ids[record["id"]].add(index)
        urls[source_key(record["url"])].add(index)
        texts[normalized].add(index)
        shingle_sets.append(_shingles(normalized, recipe.near_duplicate_min_words))

    frequency = Counter(token for values in shingle_sets for token in values)
    postings: dict[bytes, list[int]] = defaultdict(list)
    for index, values in enumerate(shingle_sets):
        if values:
            for token in _prefix(values, frequency, recipe.near_duplicate_threshold):
                postings[token].append(index)

    excluded: set[str] = set()
    with closing(parquet_records(files, data_seed)) as old_records:
        remaining = old_pool_documents
        while remaining:
            batch = list(itertools.islice(old_records, min(_OLD_BATCH_SIZE, remaining)))
            if not batch:
                raise ValueError("FineWeb ended before the old source pool was reconstructed")
            remaining -= len(batch)
            for old, _ in batch:
                normalized = _normalized_text(old)
                exact = set(ids.get(old["id"], ()))
                exact.update(urls.get(source_key(old["url"]), ()))
                exact.update(texts.get(normalized, ()))
                excluded.update(clusters[index] for index in exact)

                old_values = _shingles(normalized, recipe.near_duplicate_min_words)
                if not old_values:
                    continue
                possible: set[int] = set()
                for token in _prefix(old_values, frequency, recipe.near_duplicate_threshold):
                    possible.update(postings.get(token, ()))
                for index in possible:
                    values = shingle_sets[index]
                    if min(len(values), len(old_values)) < recipe.near_duplicate_threshold * max(
                        len(values), len(old_values)
                    ):
                        continue
                    overlap = len(values & old_values)
                    if overlap >= recipe.near_duplicate_threshold * (
                        len(values) + len(old_values) - overlap
                    ):
                        excluded.add(clusters[index])
    return excluded


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


def prepare_pool(config: dict) -> dict:
    """Build one immutable source pool; a different scan requires a different pool."""
    _validate_pool_config(config)
    path = Path(config["pool_dir"]) / "source-pool.json"
    contract = {key: config[key] for key in ("source", "window", "batch_counts")}
    if path.exists():
        pool = load_json(path)
        if pool["config"] != contract:
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
    counts = Counter(
        scanned_documents=len(scanned), old_pool_documents=source["old_pool_documents"]
    )
    candidates, seen = [], set()
    for (record, location), cluster in zip(scanned, clusters, strict=True):
        if document_rejection_reason(record, 64) is not None:
            counts["basic_rejected_documents"] += 1
        elif cluster in excluded:
            counts["old_pool_duplicate_documents"] += 1
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
        "statistics": dict(counts),
        "split_counts": dict(by_split),
        "segment_counts": dict(Counter(str(len(d["segments"])) for d in documents)),
        "documents": documents,
    }
    save_json(path, pool)
    return pool


def prepare_selection(config: dict) -> dict:
    """Allocate disjoint split slices by batch index; never recluster during annotation."""
    pool = load_json(Path(config["source_pool_dir"]) / "source-pool.json")
    index = config["batch_index"]
    if type(index) is not int or index < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    selected, ranges = [], {}
    for split in ("train", "dev", "test"):
        candidates = [d for d in pool["documents"] if d["split"] == split]
        count = pool["config"]["batch_counts"][split]
        start, stop = min(index * count, len(candidates)), min((index + 1) * count, len(candidates))
        ranges[split] = {"start": start, "stop": stop, "requested": count, "selected": stop - start}
        selected.extend(dict(d) for d in candidates[start:stop])
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
            "available_by_split": pool["split_counts"],
        },
    }
