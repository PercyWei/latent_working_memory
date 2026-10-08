"""多段采样参数与由实际参数生成的数据集名称。"""

from dataclasses import dataclass, field
from fractions import Fraction
import math
import re


@dataclass(frozen=True)
class DataPreparationConfig:
    source_glob: str
    capacity: int = 512
    min_segment_ratio: float = 1
    max_segment_ratio: float = 3
    min_segments: int = 3
    max_segments: int = 5
    continuation_tokens: int = 512
    content_reserve_ratio: float = 1.5
    source_batch_size: int = 100000
    source_seed: int = 20260907
    seed: int = 20260916
    split_fractions: tuple = (0.9, 0.05, 0.05)
    counts: dict = field(default_factory=lambda: {"train": 32000, "dev": 128, "test": 128})

    def __post_init__(self):
        object.__setattr__(self, "split_fractions", tuple(self.split_fractions))
        if not isinstance(self.source_glob, str) or not self.source_glob.strip():
            raise ValueError("source_glob is required")
        for name in (
            "capacity",
            "min_segments",
            "max_segments",
            "continuation_tokens",
            "source_batch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("min_segment_ratio", "max_segment_ratio", "content_reserve_ratio"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_segment_ratio > self.max_segment_ratio:
            raise ValueError("segment ratios must be ordered")
        if self.min_segment_tokens > self.max_segment_tokens:
            raise ValueError("segment ratio interval contains no integer token length")
        if not 2 <= self.min_segments <= self.max_segments:
            raise ValueError("segment counts must be ordered and at least two")
        if self.content_reserve_ratio < 1:
            raise ValueError("content_reserve_ratio must be at least one")
        for name in ("source_seed", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            len(self.split_fractions) != 3
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in self.split_fractions
            )
            or not math.isclose(sum(self.split_fractions), 1)
        ):
            raise ValueError("split_fractions must contain three positive fractions summing to one")
        if (
            set(self.counts) != {"train", "dev", "test"}
            or any(type(v) is not int or v < 0 for v in self.counts.values())
            or self.counts["train"] == 0
        ):
            raise ValueError("counts require positive train and nonnegative dev/test quotas")

    @property
    def min_segment_tokens(self):
        return math.ceil(self.capacity * Fraction(str(self.min_segment_ratio)))

    @property
    def max_segment_tokens(self):
        return math.floor(self.capacity * Fraction(str(self.max_segment_ratio)))

    def candidate_chars(self, content_tokens):
        return math.ceil(
            4
            * (content_tokens + self.continuation_tokens)
            * Fraction(str(self.content_reserve_ratio))
        )

    def available_content_tokens(self, characters):
        return (
            math.floor(characters / (4 * Fraction(str(self.content_reserve_ratio))))
            - self.continuation_tokens
        )

    def dataset_name(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
            raise ValueError(
                "run_id must start with a letter or digit and contain only letters, digits, '.', '_' or '-'"
            )
        # 名称中的规模是简洁近似值；精确数量和分段规则写入 preparation.json。
        train = (
            f"{self.counts['train'] / 1000:.1f}".rstrip("0").rstrip(".") + "k"
            if self.counts["train"] >= 1000
            else str(self.counts["train"])
        )
        return (
            f"fineweb-multisegment-k{self.capacity}"
            f"-seg{self.min_segment_ratio:g}to{self.max_segment_ratio:g}x"
            f"_train{train}_{run_id}"
        )
