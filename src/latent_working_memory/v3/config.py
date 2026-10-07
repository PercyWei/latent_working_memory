"""v3 实验的单一配置；数据与输出的相对路径以项目运行目录为基准。"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re


METHODS = ("icae_single", "icae_multi", "autocompressors", "memory_change", "information_loss")
DYNAMIC_METHODS = ("memory_change", "information_loss")


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def finite(value, name, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"{name} must be finite and >= {minimum}")


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    revision: str | None = None
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    memory_slots: int = 64
    lora_rank: int = 128
    lora_alpha: int = 32
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __post_init__(self):
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path is required")
        object.__setattr__(self, "model_name_or_path", os.path.expanduser(self.model_name_or_path))
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        if self.attention_implementation not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError("unsupported attention_implementation")
        for name in ("memory_slots", "lora_rank", "lora_alpha"):
            positive_integer(getattr(self, name), name)
        if not isinstance(self.lora_target_modules, (list, tuple)):
            raise ValueError("lora_target_modules must be a sequence of module names")
        object.__setattr__(self, "lora_target_modules", tuple(self.lora_target_modules))
        if not self.lora_target_modules or any(
            not isinstance(item, str) or not item for item in self.lora_target_modules
        ):
            raise ValueError("lora_target_modules must contain module names")


@dataclass(frozen=True)
class ObjectiveConfig:
    method: str = "memory_change"
    stage: str = "pretrain"
    qa_batch_size: int = 8
    segment_tokens: int = 1024
    ac_min_segment_tokens: int = 768
    ac_max_segment_tokens: int = 1024
    ac_bptt_steps: int = 2
    append_probability: float = 0.5
    threshold_i: float = 0.1
    threshold_d: float = 0.1
    threshold_g: float = 0.1
    eta: float = 0.0
    rms_epsilon: float = 1e-6
    seed: int = 20261004
    ae_prompt: str = "Reconstruct the text stored in memory:\n"
    lm_prompt: str = "Continue the text stored in memory:\n"
    qa_prompt: str = "Question: {question}\nAnswer:"

    def __post_init__(self):
        allowed = {
            "icae_single": {"pretrain", "qa"},
            "icae_multi": {"pretrain", "qa"},
            "autocompressors": {"lm"},
            "memory_change": {"pretrain", "warmup", "policy"},
            "information_loss": {"pretrain", "warmup", "policy"},
        }
        if self.method not in allowed or self.stage not in allowed[self.method]:
            raise ValueError(f"unsupported method/stage: {self.method}/{self.stage}")
        for name in (
            "qa_batch_size",
            "segment_tokens",
            "ac_min_segment_tokens",
            "ac_max_segment_tokens",
            "ac_bptt_steps",
        ):
            positive_integer(getattr(self, name), name)
        if self.ac_min_segment_tokens > self.ac_max_segment_tokens:
            raise ValueError("invalid AutoCompressors segment interval")
        if self.method == "autocompressors" and (
            self.ac_bptt_steps < 2 or self.ac_min_segment_tokens < 2
        ):
            raise ValueError(
                "AutoCompressors needs at least two segments and two tokens per segment"
            )
        for name in (
            "threshold_i",
            "threshold_d",
            "threshold_g",
            "eta",
            "rms_epsilon",
            "append_probability",
        ):
            finite(getattr(self, name), name)
        if not 0 <= self.eta < self.threshold_g or self.rms_epsilon == 0:
            raise ValueError("require 0 <= eta < threshold_g and rms_epsilon > 0")
        if self.append_probability > 1 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("invalid warmup probability or seed")
        if self.qa_prompt.count("{question}") != 1 or not self.ae_prompt or not self.lm_prompt:
            raise ValueError("prompts must be nonempty; qa_prompt needs one {question}")
        try:
            self.qa_prompt.format(question="example")
        except (KeyError, ValueError, IndexError) as error:
            raise ValueError("qa_prompt only supports the {question} placeholder") from error


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: str
    output_dir: str
    epochs: int = 1
    micro_batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 20261004
    eval_every: int = 25
    save_every: int = 25
    pretrain_data_view: str = "text_samples"
    min_input_tokens: int = 1
    max_input_tokens: int = 1024
    max_train_samples: int | None = None
    max_dev_samples: int | None = None
    init_checkpoint: str | None = None
    experiment_dir: str | None = None
    experiment_id: str | None = None
    swanlab_project: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.dataset_dir or not self.output_dir:
            raise ValueError("dataset_dir and output_dir are required")
        for name in (
            "epochs",
            "micro_batch_size_per_gpu",
            "gradient_accumulation_steps",
            "eval_every",
            "save_every",
            "min_input_tokens",
            "max_input_tokens",
        ):
            positive_integer(getattr(self, name), name)
        for name in ("max_train_samples", "max_dev_samples"):
            if getattr(self, name) is not None:
                positive_integer(getattr(self, name), name)
        if self.min_input_tokens > self.max_input_tokens:
            raise ValueError("invalid pretraining length interval")
        if self.pretrain_data_view not in {
            "text_samples",
            "reconstruction_single",
            "reconstruction_first_write",
        }:
            raise ValueError("unsupported pretrain_data_view")
        for name in ("learning_rate", "weight_decay", "gradient_clip"):
            finite(getattr(self, name), name)
        if self.learning_rate == 0 or self.gradient_clip == 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "tags", tuple(self.tags))
        if self.swanlab_project is not None and not self.group:
            raise ValueError("SwanLab logging requires an explicit experiment group")
        if (self.experiment_dir is None) != (self.experiment_id is None):
            raise ValueError("experiment_dir and experiment_id must be specified together")
        if self.experiment_id is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", self.experiment_id
        ):
            raise ValueError("experiment_id must contain only letters, digits, '-' and '_'")

    def global_batch_size(self, world_size):
        return world_size * self.micro_batch_size_per_gpu * self.gradient_accumulation_steps


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    objective: ObjectiveConfig
    training: TrainingConfig

    def __post_init__(self):
        if self.training.experiment_dir is not None:
            expected = Path(self.training.experiment_dir) / self.objective.stage
            if Path(self.training.output_dir).resolve() != expected.resolve():
                raise ValueError("output_dir must be experiment_dir / objective.stage")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))


def load_experiment(path):
    raw = json.loads(Path(path).read_text())
    if set(raw) != {"model", "objective", "training"}:
        raise ValueError("experiment must contain model, objective and training")
    return ExperimentConfig(
        ModelConfig(**raw["model"]),
        ObjectiveConfig(**raw["objective"]),
        TrainingConfig(**raw["training"]),
    )
