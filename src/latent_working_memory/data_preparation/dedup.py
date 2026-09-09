from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from latent_working_memory.data_preparation.config import PreparationConfig


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
        text = " ".join(record["text"].split())
        for key in (
            ("id", record["id"]),
            ("url", source_key(record["url"])),
            ("text", text),
        ):
            if key in exact:
                union(i, exact[key])
            else:
                exact[key] = i
        words = re.findall(r"\w+", text.casefold())
        values = set()
        if len(words) >= config.near_duplicate_min_words:
            values = {
                hashlib.blake2b(" ".join(words[j : j + 5]).encode(), digest_size=8).digest()
                for j in range(len(words) - 4)
            }
        shingles.append(values)
    frequencies = Counter(token for values in shingles for token in values)
    postings = defaultdict(list)
    threshold = config.near_duplicate_threshold
    for i, values in enumerate(shingles):
        if not values:
            continue
        ordered = sorted(values, key=lambda token: (frequencies[token], token))
        prefix = ordered[: len(values) - math.ceil(threshold * len(values)) + 1]
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
