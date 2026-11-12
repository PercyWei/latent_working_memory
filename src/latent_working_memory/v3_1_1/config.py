"""v3.1.1 的三层模型、方法与训练配置。"""

from dataclasses import asdict, dataclass, field, replace
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
class ModelStructureConfig:
    memory_slots: int = 128
    boundary_hidden_size: int | None = None

    def __post_init__(self):
        _positive_ints(self, ("memory_slots",))
        if self.boundary_hidden_size is not None:
            _positive_ints(self, ("boundary_hidden_size",))


@dataclass(frozen=True)
class ModelLoraConfig:
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])

    def __post_init__(self):
        _positive_ints(self, ("lora_rank",))
        _number(self.lora_alpha, "lora_alpha")
        if (not isinstance(self.lora_target_modules, list) or not self.lora_target_modules
            or any(not isinstance(name, str) or not name.strip() for name in self.lora_target_modules)
            or len(set(self.lora_target_modules)) != len(self.lora_target_modules)):
            raise ValueError("lora_target_modules must be a nonempty list of distinct module names")


@dataclass(frozen=True)
class ComputeConfig:
    dtype: str = "float32"
    gradient_checkpointing: bool = True
    mixed_precision: bool = False
    save_on_cpu: bool = False

    def __post_init__(self):
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        for name in ("gradient_checkpointing", "mixed_precision", "save_on_cpu"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.mixed_precision and self.dtype != "float32":
            raise ValueError("mixed_precision requires compute.dtype=float32")


@dataclass(frozen=True)
class ModelConfig:
    structure: ModelStructureConfig = field(default_factory=ModelStructureConfig)
    lora: ModelLoraConfig = field(default_factory=ModelLoraConfig)


@dataclass(frozen=True)
class PretrainConfig:
    lm_ratio: float = 0.5
    lm_target_tokens: int = 512

    def __post_init__(self):
        _positive_ints(self, ("lm_target_tokens",))
        _number(self.lm_ratio, "lm_ratio", inclusive=True)
        if self.lm_ratio > 1:
            raise ValueError("lm_ratio must not exceed 1")


@dataclass(frozen=True)
class RolloutConfig:
    unit_tokens: int = 256
    group_size: int = 4

    def __post_init__(self):
        _positive_ints(self, ("unit_tokens", "group_size"))
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")


@dataclass(frozen=True)
class GRPOConfig:
    slot_cost: float = 0.001
    policy_weight: float = 1.0
    clip_epsilon: float = 0.2
    advantage_epsilon: float = 1e-6

    def __post_init__(self):
        for name in ("slot_cost", "policy_weight"):
            _number(getattr(self, name), name, inclusive=True)
        for name in ("clip_epsilon", "advantage_epsilon"):
            _number(getattr(self, name), name)
        if self.clip_epsilon >= 1:
            raise ValueError("clip_epsilon must be less than 1")


@dataclass(frozen=True)
class MethodConfig:
    pretrain: PretrainConfig | None = None
    rollout: RolloutConfig | None = None
    grpo: GRPOConfig | None = None


@dataclass(frozen=True)
class TrainingDataConfig:
    dataset_dir: str
    max_train_data_tokens: int | None = None
    max_train_samples: int | None = None
    max_dev_samples: int | None = None
    length_curriculum: bool = False

    def __post_init__(self):
        _path(self, "dataset_dir")
        if type(self.length_curriculum) is not bool:
            raise ValueError("length_curriculum must be boolean")
        for name in ("max_train_data_tokens", "max_train_samples", "max_dev_samples"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be null or a positive integer")


@dataclass(frozen=True)
class TrainingLoopConfig:
    epochs: int | None = 1
    curriculum_stage_epochs: int = 1
    curriculum_replay_ratio: float = 0.2
    max_steps: int | None = None
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    dev_every: int | str = "after_each_epoch"
    save_every: int | str = "after_each_epoch"
    save_total_limit: int = 2

    def __post_init__(self):
        _positive_ints(self, (
            "curriculum_stage_epochs", "micro_batch_size", "gradient_accumulation_steps",
            "save_total_limit",
        ))
        _number(self.curriculum_replay_ratio, "curriculum_replay_ratio", inclusive=True)
        if self.curriculum_replay_ratio > 1:
            raise ValueError("curriculum_replay_ratio must not exceed 1")
        for name in ("epochs", "max_steps"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be null or a positive integer")
        for name in ("dev_every", "save_every"):
            value = getattr(self, name)
            if value != "after_each_epoch" and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be positive or after_each_epoch")


@dataclass(frozen=True)
class OptimizerConfig:
    compressor_learning_rate: float = 1e-4
    boundary_learning_rate: float | None = None
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    adam_epsilon: float = 1e-8
    gradient_clip: float = 1.0

    def __post_init__(self):
        for name in ("compressor_learning_rate", "adam_epsilon", "gradient_clip"):
            _number(getattr(self, name), name)
        if self.boundary_learning_rate is not None:
            _number(self.boundary_learning_rate, "boundary_learning_rate")
        for name in ("weight_decay", "beta1", "beta2"):
            _number(getattr(self, name), name, inclusive=True)
        if self.beta1 >= 1 or self.beta2 >= 1:
            raise ValueError("beta1 and beta2 must be < 1")


@dataclass(frozen=True)
class SchedulerConfig:
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1

    def __post_init__(self):
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        _number(self.min_lr_ratio, "min_lr_ratio", inclusive=True)
        if self.min_lr_ratio > 1:
            raise ValueError("min_lr_ratio must be <= 1")


@dataclass(frozen=True)
class BackwardConfig:
    bptt_steps: int | None = None

    def __post_init__(self):
        if self.bptt_steps is not None and (type(self.bptt_steps) is not int or self.bptt_steps < 1):
            raise ValueError("bptt_steps must be null or a positive integer")


@dataclass(frozen=True)
class RunConfig:
    init_checkpoint: str | None = None
    output_root: str = "artifacts/v3-1"
    output_dir: str | None = None
    run_id: str | None = None

    def __post_init__(self):
        _path(self, "output_root")
        for name in ("init_checkpoint", "output_dir"):
            _path(self, name, optional=True)
        if self.run_id is not None and (
            not isinstance(self.run_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.run_id) is None
        ):
            raise ValueError("run_id must contain only letters, digits, '-' and '_'")


@dataclass(frozen=True)
class SwanLabConfig:
    tracking: bool = True
    project: str | None = "latent-working-memory-v3.1"
    group: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.tracking) is not bool:
            raise ValueError("swanlab.tracking must be boolean")
        for name in ("project", "group"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be null or a nonempty string")
        if self.tracking and self.project is None:
            raise ValueError("enabled SwanLab tracking requires project")
        if not isinstance(self.tags, (list, tuple)) or any(
            not isinstance(tag, str) or ":" not in tag or not all(tag.split(":", 1)) for tag in self.tags
        ):
            raise ValueError("tags must be namespaced strings")
        object.__setattr__(self, "tags", tuple(self.tags))


@dataclass(frozen=True)
class TrainingConfig:
    data: TrainingDataConfig
    loop: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    backward: BackwardConfig | None = None
    run: RunConfig = field(default_factory=RunConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)

    def __post_init__(self):
        if (not self.data.length_curriculum
            and self.loop.epochs is None and self.loop.max_steps is None):
            raise ValueError("ordinary training requires epochs or max_steps")
        if self.loop.max_steps is not None and self.scheduler.warmup_steps >= self.loop.max_steps:
            raise ValueError("warmup_steps must be less than max_steps")
        if self.swanlab.group is None and self.run.output_dir is not None:
            object.__setattr__(self, "swanlab", replace(
                self.swanlab, group=Path(self.run.output_dir).parent.name,
            ))


@dataclass(frozen=True)
class ExperimentConfig:
    stage: str
    model_name_or_path: str
    training: TrainingConfig
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    method: MethodConfig = field(default_factory=lambda: MethodConfig(pretrain=PretrainConfig()))

    def __post_init__(self):
        if self.stage not in {"pretrain", "post"}:
            raise ValueError("stage must be pretrain or post")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        source = self.model_name_or_path
        if not isinstance(source, str) or not source.strip():
            raise ValueError("model_name_or_path must be a nonempty string")
        if source.startswith(("/", "~", ".")) or Path(source).is_dir():
            object.__setattr__(self, "model_name_or_path", str(Path(source).expanduser().resolve()))
        maximum = self.training.data.max_train_data_tokens
        if maximum is not None and maximum <= self.model.structure.memory_slots:
            raise ValueError("max_train_data_tokens must be greater than memory_slots")
        if self.stage == "pretrain":
            if self.method.pretrain is None:
                raise ValueError("pretrain requires method.pretrain")
            if any(value is not None for value in (
                self.method.rollout, self.method.grpo, self.training.backward,
                self.model.structure.boundary_hidden_size,
                self.training.optimizer.boundary_learning_rate,
            )):
                raise ValueError("pretrain cannot contain rollout, GRPO, boundary or BPTT settings")
        else:
            if self.method.pretrain is not None:
                raise ValueError("post cannot contain method.pretrain")
            if any(value is None for value in (
                self.method.rollout, self.method.grpo, self.training.backward,
                self.model.structure.boundary_hidden_size,
                self.training.optimizer.boundary_learning_rate,
            )):
                raise ValueError("post requires rollout, GRPO, boundary and BPTT settings")
        if self.stage == "pretrain" and self.training.run.init_checkpoint is not None:
            raise ValueError("pretrain starts from the base model; use --resume to continue it")
        if self.stage == "post" and self.training.run.init_checkpoint is None:
            raise ValueError("post requires a pretrain init_checkpoint")


def config_from_dict(payload):
    """只读取当前三层配置；未知字段由 dataclass 构造器直接报错。"""
    required = {"stage", "model_name_or_path", "model", "compute", "method", "training"}
    if (not isinstance(payload, dict) or not required <= payload.keys()
        or payload.keys() - required - {"seed"}):
        raise ValueError(
            "config must contain stage, model_name_or_path, model, compute, method and training; seed is optional"
        )
    model, method, training = payload["model"], payload["method"], payload["training"]
    method_types = ({"pretrain": PretrainConfig} if payload["stage"] == "pretrain" else {
        "rollout": RolloutConfig, "grpo": GRPOConfig,
    })
    if set(method) != set(method_types):
        raise ValueError(f"{payload['stage']} method must contain exactly {', '.join(method_types)}")
    training_types = {
        "data": TrainingDataConfig, "loop": TrainingLoopConfig,
        "optimizer": OptimizerConfig, "scheduler": SchedulerConfig,
        "run": RunConfig, "swanlab": SwanLabConfig,
    }
    if payload["stage"] == "post":
        training_types["backward"] = BackwardConfig
    if set(training) != set(training_types):
        raise ValueError(f"{payload['stage']} training must contain exactly {', '.join(training_types)}")
    return ExperimentConfig(
        stage=payload["stage"], seed=payload.get("seed", 42), model_name_or_path=payload["model_name_or_path"],
        model=ModelConfig(**{
            **model,
            "structure": ModelStructureConfig(**model["structure"]),
            "lora": ModelLoraConfig(**model["lora"]),
        }),
        compute=ComputeConfig(**payload["compute"]),
        method=MethodConfig(**{name: cls(**method[name]) for name, cls in method_types.items()}),
        training=TrainingConfig(**{name: cls(**training[name]) for name, cls in training_types.items()}),
    )


def config_to_dict(config):
    """计划、checkpoint 与实验记录只保存当前阶段适用的配置。"""
    payload = asdict(config)
    payload = {key: payload[key] for key in (
        "stage", "seed", "model_name_or_path", "model", "compute", "method", "training",
    )}
    if config.stage == "pretrain":
        del payload["model"]["structure"]["boundary_hidden_size"]
        del payload["method"]["rollout"], payload["method"]["grpo"]
        del payload["training"]["backward"]
        del payload["training"]["optimizer"]["boundary_learning_rate"]
        del payload["training"]["run"]["init_checkpoint"]
    else:
        del payload["method"]["pretrain"]
    return payload


def load_config(path, overrides=None, stage=None):
    """校验显式阶段，按分组覆盖显式提供的参数。"""
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if stage is not None and payload["stage"] != stage:
        raise ValueError(f"--stage {stage} differs from configuration stage {payload['stage']}")
    for section, groups in (overrides or {}).items():
        if section in {"seed", "model_name_or_path"}:
            payload[section] = groups
            continue
        if section == "compute":
            payload[section] = {**payload[section], **groups}
            continue
        for group, values in groups.items():
            merged = {**payload[section][group], **values}
            payload[section][group] = merged
    return config_from_dict(payload)
