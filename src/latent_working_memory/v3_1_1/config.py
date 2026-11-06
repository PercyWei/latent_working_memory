"""v3.1.1 的模型、目标与训练配置。"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re


def _positive_ints(config, names):
    for name in names:
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _number(value, name, minimum=0.0, inclusive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
        or (value < minimum if inclusive else value <= minimum)):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite and {relation} {minimum}")


def _path(config, name, optional=False):
    value = getattr(config, name)
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")
    object.__setattr__(config, name, str(Path(value).expanduser().resolve()))


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str = "~/models/Qwen3-4B-Instruct-2507"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    memory_slots: int = 128
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])
    boundary_hidden_size: int = 128

    def __post_init__(self):
        source = self.model_name_or_path
        if not isinstance(source, str) or not source.strip():
            raise ValueError("model_name_or_path must be a nonempty string")
        if source.startswith(("/", "~", ".")) or Path(source).is_dir():
            object.__setattr__(self, "model_name_or_path", str(Path(source).expanduser().resolve()))
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        if self.attention_implementation not in {"sdpa", "eager"}:
            raise ValueError("attention_implementation must be sdpa or eager")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")
        _positive_ints(self, ("memory_slots", "lora_rank", "boundary_hidden_size"))
        _number(self.lora_alpha, "lora_alpha")
        if (not isinstance(self.lora_target_modules, list) or not self.lora_target_modules
            or any(not isinstance(name, str) or not name.strip() for name in self.lora_target_modules)
            or len(set(self.lora_target_modules)) != len(self.lora_target_modules)):
            raise ValueError("lora_target_modules must be a nonempty list of distinct module names")


@dataclass(frozen=True)
class ObjectiveConfig:
    lm_ratio: float = 0.5
    lm_target_tokens: int = 512
    unit_tokens: int = 256
    group_size: int = 4
    slot_cost: float = 0.001
    policy_weight: float = 1.0
    clip_epsilon: float = 0.2
    advantage_epsilon: float = 1e-6

    def __post_init__(self):
        _positive_ints(self, ("lm_target_tokens", "unit_tokens", "group_size"))
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        for name in ("lm_ratio", "slot_cost", "policy_weight"):
            _number(getattr(self, name), name, inclusive=True)
        if self.lm_ratio > 1:
            raise ValueError("lm_ratio must not exceed 1")
        for name in ("clip_epsilon", "advantage_epsilon"):
            _number(getattr(self, name), name)
        if self.clip_epsilon >= 1:
            raise ValueError("clip_epsilon must be less than 1")


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: str = "data/fineweb-4096-docall_20261104"
    max_train_data_tokens: int | None = 12288
    max_train_samples: int | None = None
    max_dev_samples: int | None = 64
    init_checkpoint: str | None = None
    output_root: str = "artifacts/v3-1-1"
    output_dir: str | None = None
    run_id: str | None = None
    epochs: int | None = 1
    max_steps: int | None = None
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    compressor_learning_rate: float = 1e-4
    boundary_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    dev_every: int = 100
    save_every: int | str = "after_each_epoch"
    save_total_limit: int = 2
    tracking: str = "disabled"
    swanlab_project: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("dataset_dir", "output_root"):
            _path(self, name)
        for name in ("init_checkpoint", "output_dir"):
            _path(self, name, optional=True)
        if self.run_id is not None and (
            not isinstance(self.run_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.run_id) is None
        ):
            raise ValueError("run_id must contain only letters, digits, '-' and '_'")
        _positive_ints(self, (
            "micro_batch_size", "gradient_accumulation_steps",
            "dev_every", "save_total_limit",
        ))
        if self.epochs is None and self.max_steps is None:
            raise ValueError("at least one of epochs and max_steps must be specified")
        for name in ("epochs", "max_steps", "max_train_data_tokens", "max_train_samples", "max_dev_samples"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be null or a positive integer")
        if self.save_every != "after_each_epoch" and (type(self.save_every) is not int or self.save_every < 1):
            raise ValueError("save_every must be positive or after_each_epoch")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if self.max_steps is not None and self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be less than max_steps")
        for name in ("compressor_learning_rate", "boundary_learning_rate", "adam_epsilon", "gradient_clip"):
            _number(getattr(self, name), name)
        for name in ("weight_decay", "min_lr_ratio", "beta1", "beta2"):
            _number(getattr(self, name), name, inclusive=True)
        if self.min_lr_ratio > 1 or self.beta1 >= 1 or self.beta2 >= 1:
            raise ValueError("min_lr_ratio must be <= 1; beta1 and beta2 must be < 1")
        if self.tracking not in {"disabled", "swanlab"}:
            raise ValueError("tracking must be disabled or swanlab")
        for name in ("swanlab_project", "group"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be null or a nonempty string")
            if self.tracking == "swanlab" and value is None:
                raise ValueError(f"online tracking requires an explicitly configured {name}")
        if not isinstance(self.tags, (list, tuple)) or any(
            not isinstance(tag, str) or ":" not in tag or not all(tag.split(":", 1)) for tag in self.tags
        ):
            raise ValueError("tags must be namespaced strings")
        object.__setattr__(self, "tags", tuple(self.tags))


@dataclass(frozen=True)
class ExperimentConfig:
    stage: str = "pretrain"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if self.stage not in {"pretrain", "post"}:
            raise ValueError("stage must be pretrain or post")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        maximum = self.training.max_train_data_tokens
        if maximum is not None and maximum <= self.model.memory_slots:
            raise ValueError("max_train_data_tokens must be greater than memory_slots")
        if self.stage == "pretrain" and self.training.init_checkpoint is not None:
            raise ValueError("pretrain starts from the base model; use --resume to continue it")
        if self.stage == "post" and self.training.init_checkpoint is None:
            raise ValueError("post requires a pretrain init_checkpoint")


def config_from_dict(payload):
    """只读取当前配置；未知字段由 dataclass 构造器直接报错。"""
    if not isinstance(payload, dict) or set(payload) != {"stage", "seed", "model", "objective", "training"}:
        raise ValueError("config must contain exactly stage, seed, model, objective and training")
    return ExperimentConfig(
        stage=payload["stage"], seed=payload["seed"], model=ModelConfig(**payload["model"]),
        objective=ObjectiveConfig(**payload["objective"]), training=TrainingConfig(**payload["training"]),
    )


def load_config(path, overrides=None, stage=None):
    """校验显式阶段，应用 CLI 覆盖；tags 与配置中的已有标签取并集。"""
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if stage is not None and payload["stage"] != stage:
        raise ValueError(f"--stage {stage} differs from configuration stage {payload['stage']}")
    for section, values in (overrides or {}).items():
        if section == "seed":
            payload[section] = values
        else:
            merged = {**payload[section], **values}
            if section == "training" and "tags" in values:
                merged["tags"] = sorted(set(payload[section].get("tags", ())) | set(values["tags"]))
            payload[section] = merged
    return config_from_dict(payload)
