"""v4 的单一实验配置；相对数据与产物路径以配置文件目录为基准。"""

from dataclasses import dataclass, fields, replace
import json
import math
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
    model_name_or_path: str
    revision: str | None = None
    backbone_dtype: str = "float32"
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
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ValueError("revision must be null or a nonempty string")
        if self.backbone_dtype not in {"float32", "bfloat16"}:
            raise ValueError("backbone_dtype must be float32 or bfloat16")
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
    train_file: str
    dev_file: str
    output_dir: str
    epochs: int = 1
    global_batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    max_seq_length: int = 4096
    eval_every: int = 50
    save_every: int = 50

    def __post_init__(self):
        for name in ("train_file", "dev_file", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty path string")
        positive_ints(
            self,
            [field.name for field in fields(self) if field.type is int and field.name != "seed"],
        )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        finite_number(self.learning_rate, "learning_rate")
        finite_number(self.weight_decay, "weight_decay", inclusive=True)
        finite_number(self.gradient_clip, "gradient_clip")


def load_experiment(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {"model", "training"}:
        raise ValueError("experiment must contain exactly model and training objects")
    model = ModelConfig(**raw["model"])
    training = TrainingConfig(**raw["training"])
    training = replace(
        training,
        **{
            name: str((path.parent / getattr(training, name)).resolve())
            for name in ("train_file", "dev_file", "output_dir")
        },
    )
    if training.max_seq_length < model.pending_size + model.recent_size + 2:
        raise ValueError("max_seq_length must allow a next-token target after the first write")
    return model, training
