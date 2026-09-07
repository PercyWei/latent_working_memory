from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from latent_working_memory.v1.state import MemoryState


@dataclass(frozen=True, slots=True)
class NllSummary:
    total_nll: float
    target_tokens: int

    def __post_init__(self) -> None:
        if self.total_nll < 0:
            raise ValueError("total_nll must be non-negative")
        if type(self.target_tokens) is not int or self.target_tokens <= 0:
            raise ValueError("target_tokens must be a positive integer")

    @property
    def mean_nll(self) -> float:
        return self.total_nll / self.target_tokens

    @property
    def perplexity(self) -> float:
        return math.exp(self.mean_nll)


@dataclass(frozen=True, slots=True)
class MemoryFootprintPoint:
    prefix_end: int
    persistent_bytes: int

    def __post_init__(self) -> None:
        if type(self.prefix_end) is not int or self.prefix_end < 0:
            raise ValueError("prefix_end must be a non-negative integer")
        if type(self.persistent_bytes) is not int or self.persistent_bytes < 0:
            raise ValueError("persistent_bytes must be a non-negative integer")


def normalize_exact_match_text(text: str) -> str:
    return " ".join(text.strip().split())


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_exact_match_text(prediction) == normalize_exact_match_text(reference)


def aggregate_nll(items: Iterable[NllSummary]) -> NllSummary:
    summaries = tuple(items)
    if not summaries:
        raise ValueError("at least one NLL summary is required")
    return NllSummary(
        total_nll=sum(summary.total_nll for summary in summaries),
        target_tokens=sum(summary.target_tokens for summary in summaries),
    )


def persistent_memory_bytes(state: MemoryState, metadata_bytes: int = 0) -> int:
    if type(metadata_bytes) is not int or metadata_bytes < 0:
        raise ValueError("metadata_bytes must be a non-negative integer")
    return state.values.numel() * state.values.element_size() + metadata_bytes


def byte_token_area(points: Iterable[MemoryFootprintPoint], stream_end: int) -> int:
    samples = tuple(points)
    if not samples:
        raise ValueError("at least one footprint point is required")
    if type(stream_end) is not int or stream_end < 0:
        raise ValueError("stream_end must be a non-negative integer")
    if samples[0].prefix_end != 0:
        raise ValueError("the first footprint point must start at prefix 0")
    if samples[-1].prefix_end > stream_end:
        raise ValueError("footprint points must not extend past stream_end")
    if any(first.prefix_end >= second.prefix_end for first, second in zip(samples, samples[1:])):
        raise ValueError("footprint points must have strictly increasing prefixes")

    area = 0
    for index, point in enumerate(samples):
        next_prefix = samples[index + 1].prefix_end if index + 1 < len(samples) else stream_end
        area += point.persistent_bytes * (next_prefix - point.prefix_end)
    return area
