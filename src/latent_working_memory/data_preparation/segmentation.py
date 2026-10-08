"""共用名义段长采样、逐段字符余量及文本轨迹布局。"""

from dataclasses import dataclass
from fractions import Fraction
import math
import random


TOKEN_ESTIMATION_RULE = "len(text) / 4"


@dataclass(frozen=True)
class SegmentationConfig:
    capacity: int = 512
    min_segment_ratio: float = 1
    max_segment_ratio: float = 3
    min_segments: int = 3
    max_segments: int = 5
    continuation_tokens: int = 0
    content_reserve_ratio: float = 1

    def __post_init__(self):
        for name in ("capacity", "min_segments", "max_segments"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.continuation_tokens) is not int or self.continuation_tokens < 0:
            raise ValueError("continuation_tokens must be a nonnegative integer")
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

    @property
    def min_segment_tokens(self):
        return math.ceil(self.capacity * Fraction(str(self.min_segment_ratio)))

    @property
    def max_segment_tokens(self):
        return math.floor(self.capacity * Fraction(str(self.max_segment_ratio)))

    def reserved_chars(self, tokens):
        """将余量前的名义 token 长度换算为含余量的完整字符窗口。"""
        return math.ceil(4 * tokens * Fraction(str(self.content_reserve_ratio)))

    @property
    def min_segment_chars(self):
        return self.reserved_chars(self.min_segment_tokens)

    @property
    def max_segment_chars(self):
        return self.reserved_chars(self.max_segment_tokens)

    @property
    def continuation_chars(self):
        return self.reserved_chars(self.continuation_tokens)

    @property
    def minimum_window_chars(self):
        return self.min_segments * self.min_segment_chars + self.continuation_chars

    def candidate_chars(self, parts):
        return sum(self.reserved_chars(part) for part in parts) + self.continuation_chars

    def is_valid_segment_length(self, characters):
        nominal_tokens = math.floor(characters / (4 * Fraction(str(self.content_reserve_ratio))))
        return (
            self.min_segment_tokens <= nominal_tokens <= self.max_segment_tokens
            and self.reserved_chars(nominal_tokens) == characters
        )


def segment_lengths(count, available_chars, config, rng):
    """按字符预算抽取各段余量前的名义 token 长度，逐段预留最小字符数。"""
    remaining, parts = available_chars, []
    for slots in range(count, 0, -1):
        available = remaining - config.min_segment_chars * (slots - 1)
        high = min(
            config.max_segment_tokens,
            math.floor(available / (4 * Fraction(str(config.content_reserve_ratio)))),
        )
        part = rng.randint(config.min_segment_tokens, high)
        parts.append(part)
        remaining -= config.reserved_chars(part)
    rng.shuffle(parts)
    return parts


def sample_windows(text_length, document_id, seed, config):
    """顺序抽取不重叠窗口；每个窗口包含逐段余量及 continuation。"""
    rng = random.Random(f"{seed}:segmentation:{document_id}")
    start = 0
    while text_length - start >= config.minimum_window_chars:
        available_chars = text_length - start - config.continuation_chars
        max_count = min(config.max_segments, available_chars // config.min_segment_chars)
        count = rng.randint(config.min_segments, max_count)
        parts = segment_lengths(count, available_chars, config, rng)
        candidate_end = start + config.candidate_chars(parts)
        yield start, parts, candidate_end
        start = candidate_end


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _span(value: object, name: str, length: int) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item character span")
    start = _nonnegative_integer(value[0], f"{name} start")
    end = _nonnegative_integer(value[1], f"{name} end")
    if not start < end <= length:
        raise ValueError(f"{name} is outside the text")
    return start, end


def validate_text_layout(document: dict) -> tuple[str, dict[str, tuple[int, int]]]:
    _nonempty(document["trajectory_id"], "trajectory_id")
    document_id = _nonempty(document["document_id"], "document_id")
    _nonempty(document["dedup_cluster"], "dedup_cluster")
    if document["split"] not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev or test")
    source = document["source"]
    if not isinstance(source, dict) or set(source) != {"file", "row_group", "row_index"}:
        raise ValueError("source requires file, row_group and row_index")
    _nonempty(source["file"], "source.file")
    _nonnegative_integer(source["row_group"], "source.row_group")
    _nonnegative_integer(source["row_index"], "source.row_index")
    text = document["text"]
    if not isinstance(text, str) or not text:
        raise ValueError("trajectory text must be nonempty")
    window_start, window_end = _span(document["window_char_span"], "window_char_span", 2**63)
    if window_end - window_start != len(text):
        raise ValueError("window_char_span does not match trajectory text length")
    segments = document["segments"]
    if not isinstance(segments, list) or len(segments) < 2:
        raise ValueError("trajectory must have at least two segments")
    layout = {}
    next_start = 0
    for index, segment in enumerate(segments):
        expected_id = f"seg{index}"
        if segment["segment_id"] != expected_id:
            raise ValueError(f"expected segment {expected_id}")
        start, end = _span(segment["char_span"], f"{expected_id}.char_span", len(text))
        if start != next_start:
            raise ValueError("segment spans must consecutively cover the trajectory text")
        layout[expected_id] = (start, end)
        next_start = end
    if next_start != len(text):
        raise ValueError("segment spans do not cover the trajectory text")
    return document_id, layout
