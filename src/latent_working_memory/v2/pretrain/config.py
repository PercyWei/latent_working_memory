"""一次实验的模型、数据选择和执行配置。"""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class SelectionConfig:
    source_glob: str
    capacity: int = 512
    continuation_tokens: int = 512
    max_documents: int = 100000
    source_seed: int = 20260907
    seed: int = 20260916
    split_fractions: tuple = (0.9, 0.05, 0.05)
    warmup: dict = field(default_factory=lambda: {"train": 32000, "dev": 128, "test": 128})
    multiround: dict = field(default_factory=lambda: {"train": 32000, "dev": 128, "test": 128})

    def __post_init__(self):
        object.__setattr__(self, "split_fractions", tuple(self.split_fractions))
        for name in ("capacity", "continuation_tokens", "max_documents"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if (
            len(self.split_fractions) != 3
            or any(x <= 0 for x in self.split_fractions)
            or not math.isclose(sum(self.split_fractions), 1)
        ):
            raise ValueError("split_fractions must be three positive fractions summing to one")
        for counts in (self.warmup, self.multiround):
            if (
                set(counts) != {"train", "dev", "test"}
                or any(type(n) is not int or n < 0 for n in counts.values())
                or counts["train"] == 0
            ):
                raise ValueError("stage counts require positive train and nonnegative dev/test")


@dataclass(frozen=True)
class TrainingConfig:
    objective: str = "ae"
    warmup_epochs: int = 1
    multiround_epochs: int = 2
    global_batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    lm_weight: float = 1.0
    seed: int = 42
    save_every: int = 1000
    eval_every: int = 1000
    generation_samples: int = 8
    ae_prompt: str = "Reconstruct the text stored in memory:\n"
    lm_prompt: str = "Continue the text stored in memory:\n"

    def __post_init__(self):
        if self.objective not in {"ae", "ae_lm"}:
            raise ValueError("objective must be ae or ae_lm")
        for name in ("warmup_epochs", "multiround_epochs", "generation_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.warmup_epochs + self.multiround_epochs == 0:
            raise ValueError("at least one training epoch is required")
        for name in ("global_batch_size", "save_every", "eval_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "gradient_clip", "lm_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not self.ae_prompt or not self.lm_prompt:
            raise ValueError("read prompts must not be empty")
