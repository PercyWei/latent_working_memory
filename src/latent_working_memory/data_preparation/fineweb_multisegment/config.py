"""多段采样参数与由实际参数生成的数据集名称。"""

from dataclasses import asdict, dataclass, field
import re

from latent_working_memory.data_preparation.fineweb_source import split_fractions
from latent_working_memory.data_preparation.segmentation import SegmentationConfig


@dataclass(frozen=True)
class DataPreparationConfig:
    source_dir: str
    window: SegmentationConfig = field(
        default_factory=lambda: SegmentationConfig(
            continuation_tokens=512, content_reserve_ratio=1.5
        )
    )
    source_batch_size: int = 100000
    source_seed: int = 20260907
    selection_seed: int = 20260916
    split_counts: dict = field(default_factory=lambda: {"train": 32000, "dev": 128, "test": 128})

    def __post_init__(self):
        if not isinstance(self.source_dir, str) or not self.source_dir.strip():
            raise ValueError("source_dir is required")
        if not isinstance(self.window, SegmentationConfig):
            raise ValueError("window must be a SegmentationConfig")
        if self.window.continuation_tokens < 1:
            raise ValueError("continuation_tokens must be a positive integer")
        if type(self.source_batch_size) is not int or self.source_batch_size < 1:
            raise ValueError("source_batch_size must be a positive integer")
        for name in ("source_seed", "selection_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        split_fractions(self.split_counts)

    @classmethod
    def from_mapping(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError("multisegment configuration must be a JSON object")
        raw = dict(raw)
        raw["window"] = SegmentationConfig(**raw["window"])
        return cls(**raw)

    def to_dict(self):
        return asdict(self)

    def dataset_name(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
            raise ValueError(
                "run_id must start with a letter or digit and contain only letters, digits, '.', '_' or '-'"
            )
        train = (
            f"{self.split_counts['train'] / 1000:.1f}".rstrip("0").rstrip(".") + "k"
            if self.split_counts["train"] >= 1000
            else str(self.split_counts["train"])
        )
        return (
            f"fineweb-multisegment-k{self.window.capacity}"
            f"-seg{self.window.min_segment_ratio:g}to{self.window.max_segment_ratio:g}x"
            f"_train{train}_{run_id}"
        )
