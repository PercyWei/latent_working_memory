from __future__ import annotations

import pytest
import torch

from latent_working_memory.v1.evaluation import (
    MemoryFootprintPoint,
    NllSummary,
    aggregate_nll,
    byte_token_area,
    exact_match,
    persistent_memory_bytes,
)
from latent_working_memory.v1.state import MemoryState


def test_exact_match_only_normalizes_whitespace() -> None:
    assert exact_match("  Alpha   Beta\n", "Alpha Beta")
    assert not exact_match("alpha beta", "Alpha Beta")


def test_nll_aggregation_is_token_weighted() -> None:
    summary = aggregate_nll((NllSummary(2.0, 1), NllSummary(2.0, 3)))
    assert summary.mean_nll == 1.0
    assert summary.perplexity == pytest.approx(2.718281828459045)


def test_memory_bytes_and_byte_token_area() -> None:
    state = MemoryState(torch.zeros(4, 8, dtype=torch.bfloat16), seen_tokens=10)
    assert persistent_memory_bytes(state, metadata_bytes=8) == 72
    points = (
        MemoryFootprintPoint(0, 32),
        MemoryFootprintPoint(10, 64),
    )
    assert byte_token_area(points, stream_end=20) == 32 * 10 + 64 * 10
