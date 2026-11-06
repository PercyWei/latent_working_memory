"""完整文本预训练样本的唯一持久化格式。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedPretrainSample:
    trajectory_id: str
    document_id: str
    dedup_cluster: str
    split: str
    source: dict
    window_char_span: list[int]
    text: str
    estimated_tokens: float

    def __post_init__(self):
        for name in ("trajectory_id", "document_id", "dedup_cluster", "text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.split not in ("train", "dev"):
            raise ValueError("split must be train or dev")
        span = self.window_char_span
        if (not isinstance(span, list) or len(span) != 2
            or any(type(position) is not int for position in span)):
            raise ValueError("window_char_span must contain two integer character positions")
        start, end = span
        if not 0 <= start < end or end - start != len(self.text):
            raise ValueError("window_char_span must be absolute and match the text length")
        if (type(self.estimated_tokens) not in (int, float)
            or self.estimated_tokens != len(self.text) / 4):
            raise ValueError("estimated_tokens must equal len(text) / 4")
