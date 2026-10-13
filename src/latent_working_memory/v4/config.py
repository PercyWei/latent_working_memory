"""v4 实验配置；相对路径以项目运行目录为基准。"""

from dataclasses import dataclass, fields, replace
import json
import math
import os
from pathlib import Path


def positive_ints(config, names):
    for name in names:
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def finite_number(value, name, minimum=0, inclusive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value < minimum if inclusive else value <= minimum)
    ):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite and {relation} {minimum}")


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str = "~/models/Qwen3-4B-Instruct-2507"
    revision: str | None = None
    backbone_dtype: str = "float32"
    attention_implementation: str = "sdpa"
    loss_chunk_size: int = 256
    pending_size: int = 256
    recent_size: int = 128
    num_slots: int = 32
    memory_dim: int = 128
    query_dim: int = 64
    value_dim: int = 128
    num_probes: int = 16
    query_mode: str = "conditioned"
    query_normalization: str = "none"
    read_query_norm: float | None = None
    compression_query_norm: float | None = None
    inner_steps: int = 1
    inner_lr: float = 0.1
    correction_scale: float = 0.1

    def __post_init__(self):
        if not isinstance(self.model_name_or_path, str) or not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be a nonempty string")
        object.__setattr__(self, "model_name_or_path", os.path.expanduser(self.model_name_or_path))
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ValueError("revision must be null or a nonempty string")
        if self.backbone_dtype not in {"float32", "bfloat16"}:
            raise ValueError("backbone_dtype must be float32 or bfloat16")
        if self.attention_implementation not in {"eager", "sdpa"}:
            raise ValueError("attention_implementation must be eager or sdpa")
        positive_ints(self, [field.name for field in fields(self) if field.type is int])
        if self.query_mode not in {"fixed", "conditioned"}:
            raise ValueError("query_mode must be fixed or conditioned")
        if self.query_normalization not in {"none", "fixed_norm"}:
            raise ValueError("query_normalization must be none or fixed_norm")
        for name in ("read_query_norm", "compression_query_norm"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, math.sqrt(self.query_dim))
            finite_number(getattr(self, name), name)
        finite_number(self.inner_lr, "inner_lr")
        finite_number(self.correction_scale, "correction_scale", inclusive=True)


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: str
    output_dir: str | None = None
    epochs: int = 1
    global_batch_size: int = 8
    micro_batch_size_per_gpu: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    max_seq_length: int | None = None
    eval_every: int = 50
    save_every: int | str = 50
    save_total_limit: int = 2
    swanlab_project: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.dataset_dir, str) or not self.dataset_dir.strip():
            raise ValueError("dataset_dir must be a nonempty path string")
        if self.output_dir is not None and (
            not isinstance(self.output_dir, str) or not self.output_dir.strip()
        ):
            raise ValueError("output_dir must be null or a nonempty path string")
        positive_ints(
            self,
            [field.name for field in fields(self) if field.type is int and field.name != "seed"],
        )
        if self.save_every != "after_each_epoch" and (
            type(self.save_every) is not int or self.save_every < 1
        ):
            raise ValueError("save_every must be a positive integer or after_each_epoch")
        if self.max_seq_length is not None and (
            type(self.max_seq_length) is not int or self.max_seq_length < 1
        ):
            raise ValueError("max_seq_length must be null or a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        finite_number(self.learning_rate, "learning_rate")
        finite_number(self.weight_decay, "weight_decay", inclusive=True)
        finite_number(self.gradient_clip, "gradient_clip")
        for name in ("swanlab_project", "group"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be null or a nonempty string")
        if not isinstance(self.tags, (list, tuple)) or any(
            not isinstance(tag, str) or ":" not in tag or not all(tag.split(":", 1))
            for tag in self.tags
        ):
            raise ValueError("tags must be namespaced strings, for example study:query-generation")
        object.__setattr__(self, "tags", tuple(self.tags))


def validate_experiment(model, training):
    """检查模型与训练配置共同决定的约束。"""
    min_prompt_length = model.pending_size + model.recent_size + 1
    if training.max_seq_length is not None and training.max_seq_length <= min_prompt_length:
        raise ValueError(f"max_seq_length must exceed min_prompt_length ({min_prompt_length})")


def load_experiment(path):
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"model", "training"}:
        raise ValueError("experiment must contain exactly model and training objects")
    model = ModelConfig(**raw["model"])
    training = TrainingConfig(**raw["training"])
    training = replace(
        training,
        **{
            name: str(Path(getattr(training, name)).expanduser().resolve())
            for name in ("dataset_dir", "output_dir")
            if getattr(training, name) is not None
        },
    )
    validate_experiment(model, training)
    return model, training
