"""多段文本持久化契约；原始文件位置只用于来源追溯。"""

from dataclasses import dataclass

from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig


@dataclass(frozen=True)
class MultisegmentSample:
    sample_id: str
    document_id: str
    dedup_cluster: str
    text: str
    write_token_ends: tuple[int, ...]
    source: dict

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.sample_id,
                self.document_id,
                self.dedup_cluster,
                self.text,
            )
        ):
            raise ValueError("sample identifiers and text must be nonempty strings")
        ends = self.write_token_ends
        if (
            not isinstance(ends, (list, tuple))
            or not ends
            or any(type(v) is not int or v < 1 for v in ends)
            or list(ends) != sorted(set(ends))
        ):
            raise ValueError("write_token_ends must contain increasing positive integers")
        object.__setattr__(self, "write_token_ends", tuple(ends))
        if not isinstance(self.source, dict) or set(self.source) != {
            "file",
            "row_group",
            "row_index",
            "char_span",
        }:
            raise ValueError("source requires file, row_group, row_index and char_span")
        if not isinstance(self.source["file"], str) or not self.source["file"]:
            raise ValueError("source.file must be a nonempty path")
        if any(
            type(self.source[k]) is not int or self.source[k] < 0
            for k in ("row_group", "row_index")
        ):
            raise ValueError("source row locations must be nonnegative integers")
        span = self.source["char_span"]
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or any(type(v) is not int for v in span)
            or not 0 <= span[0] < span[1]
            or span[1] - span[0] != len(self.text)
        ):
            raise ValueError("source char_span must match the complete candidate text")

    def validate_plan(self, config: DataPreparationConfig):
        ends = self.write_token_ends
        parts = [end - start for start, end in zip((0,) + ends[:-1], ends, strict=True)]
        if not config.min_segments <= len(parts) <= config.max_segments or any(
            not config.min_segment_tokens <= n <= config.max_segment_tokens for n in parts
        ):
            raise ValueError("token plan differs from the configured segment bounds")
