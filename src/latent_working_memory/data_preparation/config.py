from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    max_documents: int = 18000
    samples_per_task: tuple[int, int, int] = (100000, 2000, 2000)
    min_document_chars: int = 64
    near_duplicate_threshold: float = 0.9
    near_duplicate_min_words: int = 64
    dedup_input_min_tokens: int = 32
    min_sample_tokens: int = 32
    max_sample_tokens: int = 4096
    lm_prefix_fraction: tuple[float, float] = (0.3, 0.7)
    length_bounds: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096)
    candidates_per_document: int = 64
    review_base_url: str = "http://127.0.0.1:8000/v1"
    review_model: str = "Qwen/Qwen3.8-27B"
    review_max_new_tokens: int = 256
    scoring_batch_size: int = 8
    review_timeout_seconds: int = 120

    def __post_init__(self) -> None:
        for name in (
            "max_documents",
            "min_document_chars",
            "near_duplicate_min_words",
            "dedup_input_min_tokens",
            "min_sample_tokens",
            "max_sample_tokens",
            "candidates_per_document",
            "review_max_new_tokens",
            "scoring_batch_size",
            "review_timeout_seconds",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.samples_per_task) != 3 or any(
            type(n) is not int or n <= 0 for n in self.samples_per_task
        ):
            raise ValueError("samples_per_task must contain positive train/dev/test quotas")
        if (
            not self.length_bounds
            or any(type(n) is not int or n <= 0 for n in self.length_bounds)
            or tuple(sorted(set(self.length_bounds))) != self.length_bounds
        ):
            raise ValueError("length_bounds must be strictly increasing positive integers")
        if not self.min_sample_tokens <= self.length_bounds[0] <= self.max_sample_tokens or (
            self.length_bounds[-1] != self.max_sample_tokens
        ):
            raise ValueError("length_bounds must cover the sample length range exactly")
        if (
            len(self.lm_prefix_fraction) != 2
            or any(
                type(n) not in (int, float) or not math.isfinite(n) for n in self.lm_prefix_fraction
            )
            or not 0 < self.lm_prefix_fraction[0] <= self.lm_prefix_fraction[1] < 1
        ):
            raise ValueError("lm_prefix_fraction must satisfy 0 < lower <= upper < 1")
        value = self.near_duplicate_threshold
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("near_duplicate_threshold must lie in (0, 1]")
        if not isinstance(self.review_model, str) or not self.review_model.strip():
            raise ValueError("review_model must be a non-empty served model name")
        url = urlsplit(self.review_base_url)
        if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment:
            raise ValueError("review_base_url must be an HTTP(S) API base URL")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def length_intervals(self) -> list[tuple[int, int]]:
        """Inclusive token intervals; each boundary belongs to the shorter interval."""
        return list(
            zip(
                (self.min_sample_tokens, *(n + 1 for n in self.length_bounds[:-1])),
                self.length_bounds,
                strict=True,
            )
        )

    def target_length_range(self, input_length: int) -> tuple[int, int]:
        # Exact fractions avoid off-by-one errors at cuts such as X=70, Y=30.
        low, high = (Fraction(str(n)) for n in self.lm_prefix_fraction)
        return (
            max(self.min_sample_tokens, math.ceil(input_length * (1 - high) / high)),
            min(self.max_sample_tokens, math.floor(input_length * (1 - low) / low)),
        )

    def accepts_lengths(self, input_length: int, target_length: int | None) -> bool:
        if not self.min_sample_tokens <= input_length <= self.max_sample_tokens:
            return False
        if target_length is None:
            return True
        lower, upper = self.target_length_range(input_length)
        return lower <= target_length <= upper

    def balanced_histogram(self) -> dict[str, dict[str, dict[str, int]]]:
        targets = {}
        for split, count in zip(("train", "dev", "test"), self.samples_per_task, strict=True):
            base, extra = divmod(count, len(self.length_bounds))
            targets[split] = {
                task: {str(b): base + (i < extra) for i, b in enumerate(self.length_bounds)}
                for task in ("ae", "continuation")
            }
        return targets

    @classmethod
    def load(cls, path: Path) -> PreparationConfig:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(cls)}:
            raise ValueError("invalid preparation configuration fields")
        for name in ("samples_per_task", "length_bounds", "lm_prefix_fraction"):
            if name in raw:
                raw[name] = tuple(raw[name])
        return cls(**raw)
