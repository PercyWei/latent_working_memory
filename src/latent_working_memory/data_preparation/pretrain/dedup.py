from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from latent_working_memory.data_preparation.pretrain.config import PreparationConfig


_WORDS = re.compile(r"\w+")


def source_key(url: str) -> str:
    parsed = urlsplit(url)
    if not parsed.netloc:
        raise ValueError("FineWeb URL must contain a hostname")
    return urlunsplit(("", parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query, ""))


def cluster_documents(records: Sequence[Mapping[str, Any]], config: PreparationConfig) -> list[str]:
    """Exact keys plus an exact Jaccard join on normalized word 5-grams.

    Prefix filtering under one global frequency order limits candidate comparisons;
    full set overlap verifies each candidate before union. Cluster IDs are source URLs.
    """
    parent = list(range(len(records)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        parent[root(b)] = root(a)

    exact: dict[tuple[str, str], int] = {}
    shingles = []
    for i, record in enumerate(records):
        text = _normalized_text(record)
        for key in (
            ("id", record["id"]),
            ("url", source_key(record["url"])),
            ("text", text),
        ):
            if key in exact:
                union(i, exact[key])
            else:
                exact[key] = i
        shingles.append(_shingles(text, config.near_duplicate_min_words))
    frequencies = Counter(token for values in shingles for token in values)
    postings = defaultdict(list)
    threshold = config.near_duplicate_threshold
    for i, values in enumerate(shingles):
        if not values:
            continue
        prefix = _prefix(values, frequencies, threshold)
        candidates = {j for token in prefix for j in postings[token]}
        for j in candidates:
            other = shingles[j]
            if min(len(values), len(other)) < threshold * max(len(values), len(other)):
                continue
            overlap = len(values & other)
            if overlap >= threshold * (len(values) + len(other) - overlap):
                union(i, j)
        for token in prefix:
            postings[token].append(i)
    names = {}
    for i, record in enumerate(records):
        key = root(i)
        name = source_key(record["url"])
        names[key] = min(names[key], name) if key in names else name
    return [names[root(i)] for i in range(len(records))]


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


def matching_clusters(
    reference_records: Iterable[Mapping[str, Any]],
    new_records: Sequence[Mapping[str, Any]],
    clusters: Sequence[str],
    recipe: PreparationConfig,
) -> set[str]:
    """Exclude exact and near-duplicate clusters matched by any reference document."""
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
    for old in reference_records:
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
