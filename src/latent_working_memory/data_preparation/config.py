from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
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
    length_bounds: tuple[int, ...] = (64, 128, 256, 512, 1024)
    random_candidates_per_document: int = 64
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
            "random_candidates_per_document",
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

    @classmethod
    def load(cls, path: Path) -> PreparationConfig:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(cls)}:
            raise ValueError("invalid preparation configuration fields")
        for name in ("samples_per_task", "length_bounds"):
            if name in raw:
                raw[name] = tuple(raw[name])
        return cls(**raw)
