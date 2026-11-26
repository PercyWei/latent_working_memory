"""Dataset-independent inputs. All character spans are half-open in ``text``."""

from dataclasses import dataclass

Span = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Question:
    qa_id: str
    evidence_spans: tuple[Span, ...]
    evidence_granularity: str
    exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    dataset: str
    split: str | None
    text: str
    unit_spans: tuple[Span, ...]
    qas: tuple[Question, ...]
    source_ref: dict
    unit_kind: str


@dataclass(frozen=True, slots=True)
class MetricConfig:
    blocks: tuple[int, ...] = (4, 8, 16)
    min_block_tokens: tuple[int, ...] = (32, 128, 256)
    max_block_tokens: tuple[int | None, ...] = (512, 1024, 2048, None)
    window_tokens: int = 128
    recent_tokens: tuple[int, ...] = (0, 128, 256, 512)

    def __post_init__(self):
        for name in ("blocks", "min_block_tokens", "max_block_tokens", "recent_tokens"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple; use (value,) for one value")
            if not values:
                raise ValueError(f"{name} must be nonempty")
            minimum = 0 if name == "recent_tokens" else 1
            for value in values:
                if name == "max_block_tokens" and value is None:
                    continue
                if type(value) is not int:
                    raise TypeError(
                        f"{name} must contain integers"
                        + (" or None" if name == "max_block_tokens" else "")
                    )
                if value < minimum:
                    requirement = "nonnegative" if minimum == 0 else "positive"
                    raise ValueError(f"{name} must contain {requirement} values")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicate values")
        if type(self.window_tokens) is not int:
            raise TypeError("window_tokens must be an integer")
        if self.window_tokens < 1:
            raise ValueError("window_tokens must be positive")
        if not self.segmentations:
            raise ValueError(
                "no valid segmentation combinations: max_block_tokens < min_block_tokens"
            )

    @property
    def segmentations(self) -> tuple[tuple[int, int, int | None], ...]:
        """Valid combinations, retaining the supplied order on each axis."""
        return tuple(
            (blocks, minimum, maximum)
            for blocks in self.blocks
            for minimum in self.min_block_tokens
            for maximum in self.max_block_tokens
            if maximum is None or minimum <= maximum
        )

    @property
    def skipped_segmentations(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (blocks, minimum, maximum)
            for blocks in self.blocks
            for minimum in self.min_block_tokens
            for maximum in self.max_block_tokens
            if maximum is not None and minimum > maximum
        )
