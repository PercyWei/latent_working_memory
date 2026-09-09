from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    max_documents: int = 18000
    split_document_limits: tuple[int, int, int] | None = None
    near_duplicate_threshold: float = 0.9
    near_duplicate_min_words: int = 64
    score_block_tokens: int = 512
    scoring_batch_size: int = 8
    review_max_new_tokens: int = 256
    fluency_model_name_or_path: str | None = None
    review_model_name_or_path: str | None = None
    audit_examples: int = 200

    def __post_init__(self) -> None:
        for name in (
            "max_documents",
            "near_duplicate_min_words",
            "score_block_tokens",
            "scoring_batch_size",
            "review_max_new_tokens",
            "audit_examples",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        value = self.near_duplicate_threshold
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("near_duplicate_threshold must lie in (0, 1]")
        limits = self.split_document_limits
        if limits is not None and (
            len(limits) != 3 or any(type(n) is not int or n <= 0 for n in limits)
        ):
            raise ValueError("split_document_limits must contain three positive integers")
        for name in ("fluency_model_name_or_path", "review_model_name_or_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty model path or null")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: Path) -> PreparationConfig:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(cls)}:
            raise ValueError("invalid preparation configuration fields")
        if raw.get("split_document_limits") is not None:
            raw["split_document_limits"] = tuple(raw["split_document_limits"])
        return cls(**raw)
