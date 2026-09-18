"""一次实验的模型、数据选择和执行配置。"""

from dataclasses import asdict, dataclass
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
    micro_batch_size: int = 1
    micro_batch_encoder_tokens: int = 4096
    micro_batch_decoder_tokens: int = 8192
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    lm_ratio: float = 0.0
    seed: int = 42
    save_every: int = 1000
    checkpoint_limit: int = 2
    eval_every: int | None = 1000
    evals_per_epoch: int | None = None
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
        for name in (
            "global_batch_size",
            "micro_batch_size",
            "micro_batch_encoder_tokens",
            "micro_batch_decoder_tokens",
            "save_every",
            "checkpoint_limit",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if (self.eval_every is None) == (self.evals_per_epoch is None):
            raise ValueError("set exactly one of eval_every and evals_per_epoch")
        for name in ("eval_every", "evals_per_epoch"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or null")
        for name in ("learning_rate", "gradient_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.lm_ratio) or not 0 <= self.lm_ratio <= 1:
            raise ValueError("lm_ratio must be finite and between 0 and 1")
        if self.objective == "ae" and self.lm_ratio != 0:
            raise ValueError("AE-only training requires lm_ratio=0")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not self.ae_prompt or not self.lm_prompt:
            raise ValueError("read prompts must not be empty")

    def to_dict(self):
        values = asdict(self)
        # Keep the serialized contract of already-running interval-based experiments.
        if self.evals_per_epoch is None:
            values.pop("evals_per_epoch")
        return values

    def evaluation_steps(self, epoch_steps, previous_steps):
        if self.evals_per_epoch is not None:
            count = self.evals_per_epoch
            return {
                previous_steps + (epoch_steps * i + count - 1) // count for i in range(1, count + 1)
            }
        return {
            step
            for step in range(previous_steps + 1, previous_steps + epoch_steps + 1)
            if step % self.eval_every == 0 or step == previous_steps + epoch_steps
        }
