"""v3.1.1 的模型、计算与训练配置。"""

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
    pending_stride: int | None = None
    pending_size: int | None = None
    recent_size: int | None = None
    boundary_hidden_size: int | None = None

    def __post_init__(self):
        _positive_ints(self, ("memory_slots",))
        if self.pending_stride is not None:
            _positive_ints(self, ("pending_stride",))
        if self.pending_size is not None:
            _positive_ints(self, ("pending_size",))
            if self.pending_size < self.memory_slots:
                raise ValueError("pending_size must be at least memory_slots")
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
    attention_implementation: str = "sdpa"
    gradient_checkpointing: bool = False
    mixed_precision: bool = False
    save_on_cpu: bool = False

    def __post_init__(self):
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        if self.attention_implementation not in {"sdpa", "flash_attention_2"}:
            raise ValueError("attention_implementation must be sdpa or flash_attention_2")
        for name in ("gradient_checkpointing", "mixed_precision", "save_on_cpu"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.mixed_precision and self.dtype != "float32":
            raise ValueError("mixed_precision requires compute.dtype=float32")
        if (self.attention_implementation == "flash_attention_2"
            and self.dtype == "float32" and not self.mixed_precision):
            raise ValueError("flash_attention_2 requires compute.dtype=bfloat16 or mixed_precision=true")


@dataclass(frozen=True)
class ModelConfig:
    structure: ModelStructureConfig = field(default_factory=ModelStructureConfig)
    lora: ModelLoraConfig = field(default_factory=ModelLoraConfig)


@dataclass(frozen=True)
class SFTObjectiveConfig:
    type: str
    strategy: str = "sft"
    lm_ratio: float | None = None
    lm_target_tokens: int | None = None

    def __post_init__(self):
        if self.strategy != "sft":
            raise ValueError("SFT objective.strategy must be sft")
        if self.type not in {"ae_lm", "qa"}:
            raise ValueError("SFT objective.type must be ae_lm or qa")
        if self.type == "ae_lm":
            _positive_ints(self, ("lm_target_tokens",))
            _number(self.lm_ratio, "lm_ratio", inclusive=True)
            if self.lm_ratio > 1:
                raise ValueError("lm_ratio must not exceed 1")
        elif self.lm_ratio is not None or self.lm_target_tokens is not None:
            raise ValueError("QA objective cannot contain lm_ratio or lm_target_tokens")


@dataclass(frozen=True)
class CompressionConfig:
    mode: str = "single"
    ratios: list[int] | None = None
    recent_size: int | None = None

    def __post_init__(self):
        if self.mode not in {"single", "multi", "policy"}:
            raise ValueError("compression.mode must be single, multi or policy")
        if self.mode == "single":
            if self.ratios is not None or self.recent_size is not None:
                raise ValueError("single compression cannot contain ratios or recent_size")
        elif self.mode == "multi":
            if (not isinstance(self.ratios, list) or not self.ratios
                or any(type(ratio) is not int or ratio < 1 for ratio in self.ratios)
                or len(set(self.ratios)) != len(self.ratios)):
                raise ValueError(
                    "multi compression ratios must be a nonempty list of distinct positive integers"
                )
        elif self.ratios is not None:
            raise ValueError("policy compression cannot contain ratios")
        if self.recent_size is not None and (
            type(self.recent_size) is not int or self.recent_size < 0
        ):
            raise ValueError("compression.recent_size must be null or a nonnegative integer")


@dataclass(frozen=True)
class DevConfig:
    fixed_ratios: list[int] = field(default_factory=list)
    max_new_tokens: int = 128

    def __post_init__(self):
        if (not isinstance(self.fixed_ratios, list)
            or any(type(ratio) is not int or ratio < 1 for ratio in self.fixed_ratios)
            or len(set(self.fixed_ratios)) != len(self.fixed_ratios)):
            raise ValueError("dev.fixed_ratios must be a list of distinct positive integers")
        _positive_ints(self, ("max_new_tokens",))


@dataclass(frozen=True)
class PolicyTransitionConfig:
    method: str
    schedule: str
    start_temperature: float
    end_temperature: float
    start_progress: float
    end_progress: float

    def __post_init__(self):
        if self.method != "temperature_annealing":
            raise ValueError("policy_transition.method must be temperature_annealing")
        if self.schedule != "linear":
            raise ValueError("policy_transition.schedule must be linear")
        for name in ("start_temperature", "end_temperature"):
            _number(getattr(self, name), name)
        if self.start_temperature <= self.end_temperature:
            raise ValueError("start_temperature must be greater than end_temperature")
        for name in ("start_progress", "end_progress"):
            _number(getattr(self, name), name, inclusive=True)
        if not self.start_progress < self.end_progress <= 1:
            raise ValueError("progress must satisfy 0 <= start_progress < end_progress <= 1")

    def temperature(self, step, total_steps):
        """按从 1 开始的 optimizer update 编号计算温度，首尾进度分别为 0、1。"""
        if type(total_steps) is not int or total_steps < 2:
            raise ValueError("temperature annealing requires at least two optimizer steps")
        if type(step) is not int or not 1 <= step <= total_steps:
            raise ValueError("temperature step must be between 1 and total_steps")
        progress = (step - 1) / (total_steps - 1)
        fraction = min(1.0, max(0.0, (progress - self.start_progress)
                                / (self.end_progress - self.start_progress)))
        return ((1 - fraction) * self.start_temperature
                + fraction * self.end_temperature)


@dataclass(frozen=True)
class PolicyObjectiveConfig:
    strategy: str
    type: str = "qa"
    reward_mode: str = "terminal"
    stepwise_cost_basis: str = "observed"
    group_size: int | None = None
    gamma: float = 1.0
    gae_lambda: float | None = None
    ppo_epochs: int | None = None
    value_loss_weight: float | None = None
    policy_transition: PolicyTransitionConfig | None = None
    memory_cost_weight: float = 0.1
    memory_cost_power: float = 1.0
    policy_weight: float = 1.0
    clip_epsilon: float = 0.2
    advantage_epsilon: float | None = None

    def __post_init__(self):
        if self.type != "qa":
            raise ValueError("policy objective.type must be qa")
        if self.strategy not in {"grpo", "scst", "ppo"}:
            raise ValueError("policy objective.strategy must be grpo, scst or ppo")
        if self.reward_mode not in {"terminal", "stepwise"}:
            raise ValueError("objective.reward_mode must be terminal or stepwise")
        if self.stepwise_cost_basis not in {"observed", "full"}:
            raise ValueError("objective.stepwise_cost_basis must be observed or full")
        if self.reward_mode != "stepwise" and self.stepwise_cost_basis != "observed":
            raise ValueError("only stepwise uses stepwise_cost_basis=full")
        _positive_ints(self, ("group_size",))
        _number(self.gamma, "gamma", inclusive=True)
        if self.gamma > 1:
            raise ValueError("gamma must not exceed 1")
        if self.strategy == "grpo":
            if self.group_size < 2:
                raise ValueError("GRPO group_size must be at least 2")
            _number(self.advantage_epsilon, "advantage_epsilon")
        elif self.advantage_epsilon is not None:
            raise ValueError("only GRPO uses advantage_epsilon")
        if self.strategy == "ppo":
            _number(self.gae_lambda, "gae_lambda", inclusive=True)
            if self.gae_lambda > 1:
                raise ValueError("gae_lambda must not exceed 1")
            if self.ppo_epochs is None:
                object.__setattr__(self, "ppo_epochs", 4)
            if self.value_loss_weight is None:
                object.__setattr__(self, "value_loss_weight", 0.5)
            _positive_ints(self, ("ppo_epochs",))
            _number(self.value_loss_weight, "value_loss_weight", inclusive=True)
        elif any(value is not None for value in (
            self.gae_lambda, self.ppo_epochs, self.value_loss_weight,
        )):
            raise ValueError("only PPO uses gae_lambda, ppo_epochs and value_loss_weight")
        for name in ("memory_cost_weight", "policy_weight"):
            _number(getattr(self, name), name, inclusive=True)
        for name in ("memory_cost_power", "clip_epsilon"):
            _number(getattr(self, name), name)
        if self.clip_epsilon >= 1:
            raise ValueError("clip_epsilon must be less than 1")

    @property
    def effective_gamma(self):
        return 1.0 if self.reward_mode == "terminal" else self.gamma


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
    critic_warmup: int = 0
    dev_every: int | str = "after_each_epoch"
    save_every: int | str = "after_each_epoch"
    save_total_limit: int = 2

    def __post_init__(self):
        _positive_ints(self, (
            "curriculum_stage_epochs", "micro_batch_size", "gradient_accumulation_steps",
            "save_total_limit",
        ))
        _number(self.curriculum_replay_ratio, "curriculum_replay_ratio", inclusive=True)
        if type(self.critic_warmup) is not int or self.critic_warmup < 0:
            raise ValueError("critic_warmup must be a nonnegative integer")
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
    objective: SFTObjectiveConfig | PolicyObjectiveConfig
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    dev: DevConfig = field(default_factory=DevConfig)
    update_mode: str = "compressor_only"
    loop: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    backward: BackwardConfig | None = None
    run: RunConfig = field(default_factory=RunConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)

    @property
    def updates_per_batch(self):
        return self.objective.ppo_epochs if self.objective.strategy == "ppo" else 1

    def __post_init__(self):
        if self.update_mode not in {"compressor_only", "joint", "boundary_only"}:
            raise ValueError("update_mode must be compressor_only, joint or boundary_only")
        if self.objective.strategy == "sft":
            if self.update_mode != "compressor_only":
                raise ValueError("SFT requires update_mode=compressor_only")
        elif self.update_mode == "compressor_only":
            raise ValueError("policy objectives require update_mode=joint or boundary_only")
        if self.objective.strategy == "ppo" and self.update_mode != "boundary_only":
            raise ValueError("PPO requires update_mode=boundary_only")
        if (self.backward is not None) != (self.update_mode == "joint"):
            raise ValueError("only joint training requires backward settings")
        if self.objective.strategy != "ppo" and self.loop.critic_warmup:
            raise ValueError("critic_warmup must be 0 for non-PPO training")
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

    @property
    def uses_boundary_policy(self):
        return self.training.objective.strategy in {"grpo", "scst", "ppo"}

    def recent_size(self, training=False):
        """single 的窗口固定为 0，multi／policy 在训练时可覆盖阶段默认窗口。"""
        compression = self.training.compression
        if compression.mode == "single":
            return 0
        if training and compression.recent_size is not None:
            return compression.recent_size
        return self.model.structure.recent_size or 0

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
        objective = self.training.objective
        recent_size = self.model.structure.recent_size
        if self.stage == "pretrain":
            if recent_size is not None:
                raise ValueError("pretrain requires model.structure.recent_size=null")
            if objective.type != "ae_lm" or objective.strategy != "sft":
                raise ValueError("pretrain requires objective.type=ae_lm and strategy=sft")
        else:
            if type(recent_size) is not int or recent_size < 0:
                raise ValueError("post requires model.structure.recent_size to be a nonnegative integer")
            if objective.type != "qa":
                raise ValueError("post requires objective.type=qa")
        maximum = self.training.data.max_train_data_tokens
        compression = self.training.compression
        slots = self.model.structure.memory_slots
        if not self.uses_boundary_policy:
            if compression.mode == "policy":
                raise ValueError("SFT requires compression.mode=single or multi")
            minimum_ratio = min(compression.ratios) if compression.mode == "multi" else 1
            minimum = (self.recent_size(training=True) + minimum_ratio * slots
                       if compression.mode == "multi" else slots + 1)
            if any(ratio < minimum_ratio for ratio in self.training.dev.fixed_ratios):
                raise ValueError("dev.fixed_ratios cannot be below the minimum training ratio")
        else:
            if compression.mode != "policy":
                raise ValueError("boundary policy training requires compression.mode=policy")
            if self.model.structure.pending_size is None:
                raise ValueError("boundary policy training requires pending_size")
            minimum = self.recent_size(training=True) + self.model.structure.pending_size
            if any(ratio * slots < self.model.structure.pending_size
                   for ratio in self.training.dev.fixed_ratios):
                raise ValueError("dev.fixed_ratios must produce segments at least pending_size tokens long")
        if maximum is not None and maximum < minimum:
            raise ValueError("max_train_data_tokens is shorter than the minimum input length")
        policy_settings = (
            self.model.structure.pending_stride, self.model.structure.pending_size,
            self.model.structure.boundary_hidden_size,
            self.training.optimizer.boundary_learning_rate,
        )
        if self.uses_boundary_policy:
            if any(value is None for value in policy_settings):
                raise ValueError("boundary policy training requires pending_stride, pending_size and boundary settings")
        elif any(value is not None for value in policy_settings):
            raise ValueError("compressor training requires pending_stride=null, pending_size=null and no boundary settings")
        if self.stage == "pretrain" and self.training.run.init_checkpoint is not None:
            raise ValueError("pretrain starts from the base model; use --resume to continue it")
        if self.uses_boundary_policy and self.training.run.init_checkpoint is None:
            raise ValueError("boundary policy training requires a compressor init_checkpoint")


def config_from_dict(payload):
    """只读取当前配置；未知字段由 dataclass 构造器直接报错。"""
    required = {"stage", "model_name_or_path", "model", "compute", "training"}
    if (not isinstance(payload, dict) or not required <= payload.keys()
        or payload.keys() - required - {"seed"}):
        raise ValueError(
            "config must contain stage, model_name_or_path, model, compute and training; seed is optional"
        )
    model, training = payload["model"], payload["training"]
    objective = training.get("objective")
    if not isinstance(objective, dict) or not {"type", "strategy"} <= objective.keys():
        raise ValueError("training.objective must contain type and strategy")
    uses_boundary_policy = objective["strategy"] in {"grpo", "scst", "ppo"}
    training_types = {
        "data": TrainingDataConfig, "loop": TrainingLoopConfig,
        "optimizer": OptimizerConfig, "scheduler": SchedulerConfig,
        "run": RunConfig, "swanlab": SwanLabConfig,
        "compression": CompressionConfig, "dev": DevConfig,
    }
    if training["update_mode"] == "joint":
        training_types["backward"] = BackwardConfig
    expected = set(training_types) | {"objective", "update_mode"}
    if set(training) != expected:
        raise ValueError(f"{payload['stage']} training must contain exactly {', '.join(sorted(expected))}")
    compression = training["compression"]
    compression_keys = {"mode"}
    if compression.get("mode") in {"multi", "policy"}:
        compression_keys.add("recent_size")
    if compression.get("mode") == "multi":
        compression_keys.add("ratios")
    if not uses_boundary_policy:
        objective_keys = {"type", "strategy"}
        if objective["type"] == "ae_lm":
            objective_keys.update(("lm_ratio", "lm_target_tokens"))
        if set(objective) != objective_keys:
            raise ValueError("SFT objective fields must match its type")
    if set(compression) != compression_keys:
        raise ValueError("compression fields must match stage and mode")
    if objective["strategy"] != "ppo" and "critic_warmup" in training["loop"]:
        raise ValueError("only PPO configuration can contain critic_warmup")
    if (training["update_mode"] == "boundary_only"
        and "compressor_learning_rate" in training["optimizer"]):
        raise ValueError("boundary_only configuration cannot contain compressor_learning_rate")
    if (training["update_mode"] == "compressor_only"
        and "boundary_learning_rate" in training["optimizer"]):
        raise ValueError("compressor_only configuration cannot contain boundary_learning_rate")
    if objective["type"] != "qa" and "max_new_tokens" in training["dev"]:
        raise ValueError("only QA configuration can contain dev.max_new_tokens")
    groups = {name: cls(**training[name]) for name, cls in training_types.items()}
    groups["update_mode"] = training["update_mode"]
    if uses_boundary_policy:
        if objective.get("reward_mode", "terminal") != "stepwise" and "stepwise_cost_basis" in objective:
            raise ValueError("only stepwise configuration can contain stepwise_cost_basis")
        if objective["strategy"] != "grpo" and "advantage_epsilon" in objective:
            raise ValueError("only GRPO configuration can contain advantage_epsilon")
        if objective["strategy"] != "ppo" and objective.keys() & {
            "gae_lambda", "ppo_epochs", "value_loss_weight",
        }:
            raise ValueError("only PPO configuration can contain gae_lambda, ppo_epochs and value_loss_weight")
        transition = objective.get("policy_transition")
        groups["objective"] = PolicyObjectiveConfig(**{
            **objective,
            "policy_transition": (None if transition is None
                                  else PolicyTransitionConfig(**transition)),
        })
    else:
        groups["objective"] = SFTObjectiveConfig(**objective)
    return ExperimentConfig(
        stage=payload["stage"], seed=payload.get("seed", 42), model_name_or_path=payload["model_name_or_path"],
        model=ModelConfig(**{
            **model,
            "structure": ModelStructureConfig(**model["structure"]),
            "lora": ModelLoraConfig(**model["lora"]),
        }),
        compute=ComputeConfig(**payload["compute"]),
        training=TrainingConfig(**groups),
    )


def config_to_dict(config):
    """完整保留模型属性，训练配置只保存当前阶段适用的分组与参数。"""
    payload = asdict(config)
    payload = {key: payload[key] for key in (
        "stage", "seed", "model_name_or_path", "model", "compute", "training",
    )}
    if config.training.objective.strategy != "ppo":
        del payload["training"]["loop"]["critic_warmup"]
    if config.training.update_mode != "joint":
        del payload["training"]["backward"]
    if config.training.update_mode == "boundary_only":
        del payload["training"]["optimizer"]["compressor_learning_rate"]
    if config.training.objective.type != "qa":
        del payload["training"]["dev"]["max_new_tokens"]
    if config.training.compression.mode != "multi":
        del payload["training"]["compression"]["ratios"]
    if config.training.compression.mode == "single":
        del payload["training"]["compression"]["recent_size"]
    if not config.uses_boundary_policy:
        del payload["training"]["optimizer"]["boundary_learning_rate"]
        if config.training.objective.type == "qa":
            del payload["training"]["objective"]["lm_ratio"]
            del payload["training"]["objective"]["lm_target_tokens"]
    else:
        if config.training.objective.reward_mode != "stepwise":
            del payload["training"]["objective"]["stepwise_cost_basis"]
        if config.training.objective.strategy != "grpo":
            del payload["training"]["objective"]["advantage_epsilon"]
        if config.training.objective.strategy != "ppo":
            for name in ("gae_lambda", "ppo_epochs", "value_loss_weight"):
                del payload["training"]["objective"][name]
    if config.stage == "pretrain":
        del payload["training"]["run"]["init_checkpoint"]
    return payload


def load_config(path, overrides=None):
    """从配置读取阶段，按分组覆盖显式提供的参数。"""
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    for section, groups in (overrides or {}).items():
        if section in {"seed", "model_name_or_path"}:
            payload[section] = groups
            continue
        if section == "compute":
            payload[section] = {**payload[section], **groups}
            continue
        for group, values in groups.items():
            if section == "training" and group == "update_mode":
                payload[section][group] = values
                continue
            merged = {**payload[section][group], **values}
            payload[section][group] = merged
    return config_from_dict(payload)
