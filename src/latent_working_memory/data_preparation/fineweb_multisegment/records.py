"""固定字符分段的多段文本契约，与 FactQA 共用正文布局。"""

from dataclasses import asdict, dataclass

from latent_working_memory.data_preparation.segmentation import (
    SegmentationConfig,
    TOKEN_ESTIMATION_RULE,
    validate_text_layout,
)


@dataclass(frozen=True)
class MultisegmentSample:
    trajectory_id: str
    document_id: str
    dedup_cluster: str
    split: str
    source: dict
    window_char_span: list[int]
    text: str
    segments: list[dict]
    continuation: str
    text_char_length: int
    estimated_tokens: float
    estimated_tokens_rule: str

    def __post_init__(self):
        validate_text_layout(asdict(self))
        if not isinstance(self.continuation, str) or not self.continuation:
            raise ValueError("continuation must be a nonempty string")
        if type(self.text_char_length) is not int or self.text_char_length != len(self.text):
            raise ValueError("text_char_length must match text")
        if (
            type(self.estimated_tokens) not in (int, float)
            or self.estimated_tokens != len(self.text) / 4
            or self.estimated_tokens_rule != TOKEN_ESTIMATION_RULE
        ):
            raise ValueError("estimated_tokens must follow len(text) / 4")

    def validate_plan(self, config: SegmentationConfig):
        lengths = [segment["char_span"][1] - segment["char_span"][0] for segment in self.segments]
        if not config.min_segments <= len(lengths) <= config.max_segments or any(
            not config.is_valid_segment_length(length) for length in lengths
        ):
            raise ValueError("character segments differ from the configured segment bounds")
        if len(self.continuation) != config.continuation_chars:
            raise ValueError("continuation length differs from its configured reserved window")
