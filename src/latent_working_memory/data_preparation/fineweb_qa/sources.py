"""Freeze FineWeb source documents and contiguous eight-segment QA windows."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import random
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

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
    remaining = min(len(text), config["segments"] * maximum)
    lengths = []
    for count in range(config["segments"], 0, -1):
        length = rng.randint(minimum, min(maximum, remaining - (count - 1) * minimum))
        lengths.append(length)
        remaining -= length
    rng.shuffle(lengths)
    start = rng.randint(0, len(text) - sum(lengths))
    cuts = [start]
    for length in lengths:
        cuts.append(cuts[-1] + length)
    return cuts


def _validate_config(source: dict, window: dict) -> None:
    for key in (
        "file_count",
        "old_pool_documents",
        "scan_documents",
        "max_train_inspections",
    ):
        if type(source[key]) is not int or source[key] <= 0:
            raise ValueError(f"source.{key} must be a positive integer")
    for key in (
        "count",
        "segments",
        "min_segment_chars",
        "max_segment_chars",
    ):
        if type(window[key]) is not int or window[key] <= 0:
            raise ValueError(f"window.{key} must be a positive integer")
    if set(window) != {"count", "segments", "min_segment_chars", "max_segment_chars"}:
        raise ValueError("window requires count, segments and minimum/maximum segment characters")
    if window["segments"] != 8 or window["min_segment_chars"] > window["max_segment_chars"]:
        raise ValueError("window requires eight segments and ordered character bounds")
    fractions = source["split_fractions"]
    if (
        len(fractions) != 3
        or any(type(value) not in (int, float) or value <= 0 for value in fractions)
        or not math.isclose(sum(fractions), 1.0)
    ):
        raise ValueError(
            "source.split_fractions must contain three positive fractions summing to one"
        )


def prepare_selection(config: dict) -> dict:
    """Freeze text-only train windows without a tokenizer or annotation requests."""
    source, window = config["source"], config["window"]
    _validate_config(source, window)
    recipe = PreparationConfig(
        near_duplicate_threshold=source["near_duplicate_threshold"],
        near_duplicate_min_words=source["near_duplicate_min_words"],
    )
    raw_dir = Path(source["raw_dir"])
    files = [raw_dir / f"{index:03d}_00000.parquet" for index in range(source["file_count"])]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"FineWeb source file is missing: {missing[0]}")

    with closing(parquet_records(files, source["data_seed"])) as records:
        for _ in range(source["old_pool_documents"]):
            if next(records, None) is None:
                raise ValueError("FineWeb ended before the old source pool was reconstructed")
        scanned = list(itertools.islice(records, source["scan_documents"]))

    new_records = [record for record, _ in scanned]
    reasons = [document_rejection_reason(record, 64) for record in new_records]
    clusters = cluster_documents(new_records, recipe)
    excluded_clusters = _old_pool_matches(
        files,
        source["data_seed"],
        source["old_pool_documents"],
        new_records,
        clusters,
        recipe,
    )
    statistics = {
        "old_pool_documents": source["old_pool_documents"],
        "scanned_documents": len(scanned),
        "old_pool_duplicate_documents": 0,
        "basic_rejected_documents": 0,
        "within_scan_duplicate_documents": 0,
        "non_train_documents": 0,
        "train_candidates": 0,
        "train_inspected": 0,
        "length_failed": 0,
        "frozen_documents": 0,
    }
    seen: set[str] = set()
    train_candidates = []
    for (record, location), cluster, reason in zip(scanned, clusters, reasons, strict=True):
        if reason is not None:
            statistics["basic_rejected_documents"] += 1
        elif cluster in excluded_clusters:
            statistics["old_pool_duplicate_documents"] += 1
        elif cluster in seen:
            statistics["within_scan_duplicate_documents"] += 1
        else:
            seen.add(cluster)
            if (
                document_split(cluster, source["data_seed"], tuple(source["split_fractions"]))
                != "train"
            ):
                statistics["non_train_documents"] += 1
                continue
            train_candidates.append((record, location, cluster))
    train_candidates.sort(
        key=lambda item: (_rank(source["selection_seed"], item[0]["id"]), item[0]["id"])
    )
    statistics["train_candidates"] = len(train_candidates)

    documents = []
    for record, location, cluster in train_candidates[: source["max_train_inspections"]]:
        if len(documents) == window["count"]:
            break
        statistics["train_inspected"] += 1
        text = record["text"]
        if len(text) < window["segments"] * window["min_segment_chars"]:
            statistics["length_failed"] += 1
            continue
        cuts = _window_cuts(text, record["id"], source["selection_seed"], window)
        start, end = cuts[0], cuts[-1]
        trajectory_id = f"{record['id']}:{start}:{end}"
        documents.append(
            {
                "trajectory_id": trajectory_id,
                "document_id": record["id"],
                "dedup_cluster": cluster,
                "split": "train",
                "source": {
                    "file": location["source_file"],
                    "row_group": location["row_group"],
                    "row_index": location["row_index"],
                },
                "window_char_span": [start, end],
                "text": text[start:end],
                "segments": [
                    {
                        "segment_id": f"seg{index}",
                        "char_span": [cuts[index] - start, cuts[index + 1] - start],
                    }
                    for index in range(window["segments"])
                ],
            }
        )
    statistics["train_uninspected"] = len(train_candidates) - statistics["train_inspected"]
    statistics["frozen_documents"] = len(documents)
    return {"documents": documents, "statistics": statistics}
