"""一次实验的模型、数据选择和执行配置。"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SelectionConfig:
    dataset_dir: str

    def __post_init__(self):
        if not isinstance(self.dataset_dir, str) or not self.dataset_dir.strip():
            raise ValueError("dataset_dir must name a prepared reconstruction dataset")


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
    checkpoint_limit: int = 2
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
        for name in ("global_batch_size", "save_every", "eval_every", "checkpoint_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "gradient_clip", "lm_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not self.ae_prompt or not self.lm_prompt:
            raise ValueError("read prompts must not be empty")
