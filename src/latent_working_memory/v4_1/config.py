"""v4.1 的统一配置；相对路径以运行目录为基准。"""

from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
import re


def _positive_ints(config, names):
    for name in names:
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _number(value, name, minimum=0.0, inclusive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value < minimum if inclusive else value <= minimum)
    ):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite and {relation} {minimum}")


def _path(config, name, optional=False):
    value = getattr(config, name)
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")
    object.__setattr__(config, name, os.path.expanduser(value))


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str = "~/models/Qwen3-4B-Instruct-2507"
    backbone_dtype: str = "bfloat16"
    pending_size: int = 1024
    recent_size: int = 256
    num_slots: int = 256
    memory_dim: int = 512
    decoder_hidden_dim: int = 1024
    init_std: float = 0.02
    inner_steps: int = 3
    inner_lr_init: float = 1.0
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    am_std_floor: float = 1e-4
    gradient_checkpointing: bool = True
    loss_chunk_size: int = 256
    prefill_chunk_size: int = 256

    def __post_init__(self):
        _path(self, "model_name_or_path")
        if self.backbone_dtype not in {"bfloat16", "float32"}:
            raise ValueError("backbone_dtype must be bfloat16 or float32")
        _positive_ints(self, (
            "pending_size", "recent_size", "num_slots", "memory_dim",
            "decoder_hidden_dim", "inner_steps", "lora_rank",
            "loss_chunk_size", "prefill_chunk_size",
        ))
        for name in ("init_std", "inner_lr_init", "lora_alpha", "am_std_floor"):
            _number(getattr(self, name), name)
        _number(self.lora_dropout, "lora_dropout", inclusive=True)
        if self.lora_dropout != 0:
            raise ValueError("v4.1 requires lora_dropout=0 for deterministic memory trajectories")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be a boolean")


@dataclass(frozen=True)
class TeacherConfig:
    num_prompts: int = 480
    response_max_tokens: int = 24
    pool_size: int = 4096
    am_sample_size: int = 128
    prepare_batch_size: int = 64
    batch_size: int = 64

    def __post_init__(self):
        _positive_ints(self, (
            "num_prompts", "response_max_tokens", "pool_size", "am_sample_size",
            "prepare_batch_size", "batch_size",
        ))
        if self.num_prompts > 480:
            raise ValueError("num_prompts cannot exceed the fixed 480-prompt bank")
        if self.am_sample_size > self.pool_size:
            raise ValueError("am_sample_size cannot exceed pool_size")


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: str = "data/delta-mem-qasper"
    output_root: str = "artifacts/v4-1"
    teacher_cache_dir: str | None = None
    output_dir: str | None = None
    run_id: str | None = None
    train_only: bool = False
    test_data: str | None = None
    max_test_samples: int | None = None
    max_new_tokens: int = 128
    eval_id: str | None = None
    epochs: int | None = None
    max_steps: int | None = 1000
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    lora_learning_rate: float = 2e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    lambda_am: float = 1.0
    dev_every: int = 50
    save_every: int | str = 100
    save_total_limit: int = 2
    max_train_data_tokens: int | None = 12288
    max_train_samples: int | None = None
    max_dev_samples: int | None = None
    tracking: str = "swanlab"
    swanlab_project: str = "latent-working-memory-v4.1"
    group: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        _path(self, "dataset_dir")
        _path(self, "output_root")
        object.__setattr__(self, "output_root", str(Path(self.output_root).resolve()))
        _path(self, "teacher_cache_dir", optional=True)
        _path(self, "output_dir", optional=True)
        _path(self, "test_data", optional=True)
        if type(self.train_only) is not bool:
            raise ValueError("train_only must be a boolean")
        if self.eval_id is not None and (
            not isinstance(self.eval_id, str) or not self.eval_id.strip()
            or Path(self.eval_id).name != self.eval_id or self.eval_id in {".", ".."}
            or "\\" in self.eval_id
        ):
            raise ValueError("eval_id must be null or a nonempty directory name")
        if self.run_id is not None and (
            not isinstance(self.run_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.run_id) is None
        ):
            raise ValueError("run_id must contain only letters, digits, '-' and '_'")
        _positive_ints(self, (
            "micro_batch_size", "gradient_accumulation_steps", "dev_every", "save_total_limit",
            "max_new_tokens",
        ))
        if self.epochs is None and self.max_steps is None:
            raise ValueError("at least one of epochs and max_steps must be specified")
        if self.save_every != "after_each_epoch" and (
            type(self.save_every) is not int or self.save_every < 1
        ):
            raise ValueError("save_every must be a positive integer or after_each_epoch")
        if self.micro_batch_size != 1:
            raise ValueError("v4.1 currently supports micro_batch_size=1")
        for name in (
            "epochs", "max_steps", "max_train_data_tokens", "max_train_samples",
            "max_dev_samples", "max_test_samples",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be null or a positive integer")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if self.max_steps is not None and self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be less than max_steps")
        for name in ("learning_rate", "lora_learning_rate", "adam_epsilon", "gradient_clip"):
            _number(getattr(self, name), name)
        for name in ("weight_decay", "lambda_am", "min_lr_ratio", "beta1", "beta2"):
            _number(getattr(self, name), name, inclusive=True)
        if self.min_lr_ratio > 1:
            raise ValueError("min_lr_ratio must not exceed 1")
        if self.beta1 >= 1 or self.beta2 >= 1:
            raise ValueError("beta1 and beta2 must be less than 1")
        if self.tracking not in {"disabled", "swanlab"}:
            raise ValueError("tracking must be disabled or swanlab")
        if not isinstance(self.swanlab_project, str) or not self.swanlab_project.strip():
            raise ValueError("swanlab_project must be a nonempty string")
        if self.group is None and self.output_dir is not None:
            object.__setattr__(self, "group", Path(self.output_dir).parent.name)
        if self.group is not None and (not isinstance(self.group, str) or not self.group.strip()):
            raise ValueError("group must be null or a nonempty string")
        if not isinstance(self.tags, (list, tuple)) or any(
            not isinstance(tag, str) or ":" not in tag or not all(tag.split(":", 1))
            for tag in self.tags
        ):
            raise ValueError("tags must be namespaced strings, for example study:online-ttt")
        object.__setattr__(self, "tags", tuple(self.tags))


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.training.teacher_cache_dir is None:
            directory = Path(self.training.output_root) / "teacher" / (
                f"p{self.model.pending_size}-r{self.model.recent_size}"
            )
            object.__setattr__(self, "training", replace(self.training, teacher_cache_dir=str(directory)))


def config_from_dict(payload):
    """读取唯一配置结构，未知字段由 dataclass 构造器直接报错。"""
    if not isinstance(payload, dict) or set(payload) != {"seed", "model", "teacher", "training"}:
        raise ValueError("config must contain exactly seed, model, teacher and training")
    return ExperimentConfig(
        seed=payload["seed"],
        model=ModelConfig(**payload["model"]),
        teacher=TeacherConfig(**payload["teacher"]),
        training=TrainingConfig(**payload["training"]),
    )


def load_config(path, overrides=None):
    """先应用显式 CLI 覆盖，再校验同一份配置；附加 tags 与配置取并集。"""
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    for section, values in (overrides or {}).items():
        if section == "seed":
            payload[section] = values
        else:
            merged = {**payload[section], **values}
            if section == "training" and "tags" in values:
                merged["tags"] = sorted(set(payload[section].get("tags", ())) | set(values["tags"]))
            payload[section] = merged
    return config_from_dict(payload)
