from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SupportOrder(str, Enum):
    """Deterministic order used when latent supports are concatenated."""

    SCORE_DESC = "score_desc"
    MEMORY_ORDER = "memory_order"


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Paper retrieval defaults plus explicit choices for missing details."""

    threshold: float = 0.8
    decay: float = 0.05
    support_order: SupportOrder = SupportOrder.SCORE_DESC
    max_retrieved: int | None = None

    def __post_init__(self) -> None:
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be within the cosine-similarity range [-1, 1]")
        if self.decay < 0.0:
            raise ValueError("decay must be non-negative")
        if self.max_retrieved is not None and self.max_retrieved < 1:
            raise ValueError("max_retrieved must be positive when provided")
