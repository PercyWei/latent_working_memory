"""配置默认值、显式覆盖与输入边界校验。"""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.json")


@dataclass
class ModelConfig:
    hidden_dim: int = 8
    noise_std: float = 0.02


@dataclass
class TrainingConfig:
    dataset_dir: str = field(default_factory=lambda: str(DEFAULT_CONFIG.parent / "datasets"))
    output_root: str = "artifacts/experiment-template"
    run_id: str | None = None
    output_dir: str | None = None
    eval_id: str | None = None
    train_only: bool = False
    test_data: str | None = None
    max_test_samples: int | None = None
    max_new_tokens: int = 128
    epochs: int | None = 2
    max_steps: int | None = None
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    dev_every: int = 2
    save_every: str | int = "after_each_epoch"
    save_total_limit: int = 2
    max_train_data_tokens: int | None = None
    max_train_samples: int | None = None
    max_dev_samples: int | None = None
    tracking: str = "disabled"
    swanlab_project: str | None = None
    group: str | None = None


def _section(cls, values, name):
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(sorted(unknown))}")
    return asdict(cls(**values))


def _positive_integer(value, name, nullable=False):
    if nullable and value is None:
        return
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer" + (" or null" if nullable else ""))


def load_config(path=None, overrides=None):
    """补齐默认值；只接受本模板的一套配置字段。"""
    values = json.loads(Path(path or DEFAULT_CONFIG).expanduser().read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) - {"seed", "model", "training"}:
        raise ValueError("config must contain only seed, model and training")
    values = {"seed": 42, "model": {}, "training": {}, **values}
    if overrides:
        if set(overrides) - {"seed", "model", "training"}:
            raise ValueError("unknown configuration override section")
        for name, updates in overrides.items():
            if name == "seed":
                values[name] = updates
            else:
                if not isinstance(values[name], dict) or not isinstance(updates, dict):
                    raise ValueError(f"{name} must be a JSON object")
                values[name] = {**values[name], **updates}
    config = {
        "seed": values["seed"],
        "model": _section(ModelConfig, values["model"], "model"),
        "training": _section(TrainingConfig, values["training"], "training"),
    }
    model, training = config["model"], config["training"]
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    _positive_integer(model["hidden_dim"], "model.hidden_dim")
    for name in (
        "epochs",
        "max_steps",
        "max_train_data_tokens",
        "max_train_samples",
        "max_dev_samples",
        "max_test_samples",
    ):
        _positive_integer(training[name], f"training.{name}", nullable=True)
    for name in (
        "micro_batch_size",
        "gradient_accumulation_steps",
        "dev_every",
        "save_total_limit",
        "max_new_tokens",
    ):
        _positive_integer(training[name], f"training.{name}")
    if training["epochs"] is None and training["max_steps"] is None:
        raise ValueError("epochs or max_steps must set the training budget")
    if training["save_every"] != "after_each_epoch":
        _positive_integer(training["save_every"], "training.save_every")
    if type(training["warmup_steps"]) is not int or training["warmup_steps"] < 0:
        raise ValueError("warmup_steps must be a non-negative integer")
    for name, value in (
        ("noise_std", model["noise_std"]),
        ("weight_decay", training["weight_decay"]),
        ("gradient_clip", training["gradient_clip"]),
        ("learning_rate", training["learning_rate"]),
        ("min_lr_ratio", training["min_lr_ratio"]),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    if training["learning_rate"] == 0 or training["gradient_clip"] == 0:
        raise ValueError("learning_rate and gradient_clip must be positive")
    if training["min_lr_ratio"] > 1:
        raise ValueError("min_lr_ratio must not exceed 1")
    if type(training["train_only"]) is not bool:
        raise ValueError("train_only must be a boolean")
    if training["tracking"] not in {"disabled", "online"}:
        raise ValueError("tracking must be disabled or online")
    if training["tracking"] == "online" and not training["swanlab_project"]:
        raise ValueError("online tracking requires an explicit swanlab_project")
    for name in (
        "dataset_dir",
        "output_root",
        "run_id",
        "output_dir",
        "eval_id",
        "test_data",
        "swanlab_project",
        "group",
    ):
        value = training[name]
        if value is None and name not in {"dataset_dir", "output_root"}:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"training.{name} must be a non-empty string")
    training["dataset_dir"] = str(Path(training["dataset_dir"]).expanduser().resolve())
    if training["test_data"] is not None:
        training["test_data"] = str(Path(training["test_data"]).expanduser().resolve())
    return config
