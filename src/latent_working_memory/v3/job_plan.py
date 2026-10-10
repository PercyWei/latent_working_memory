"""v3 运行计划：展开方法阶段，解析配置、命名与初始化依赖。"""

from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from latent_working_memory.v3.config import (
    DYNAMIC_METHODS,
    DYNAMIC_PRETRAIN_METHODS,
    METHODS,
    ExperimentConfig,
    load_preset,
)


PRESETS = {
    "icae_single": "icae_single.json",
    "icae_multi": "icae_multi.json",
    "autocompressors": "autocompressors.json",
    "memory_change": "memory_change.json",
    "information_loss": "information_loss.json",
}
LEVELS = {
    "smoke": {
        "train_samples": 16,
        "dev_samples": 4,
        "max_steps": 2,
        "eval_trajectories": 2,
        "eval_every": 1,
        "save_every": 1,
        "eval_split": "dev",
    },
    "pilot": {
        "train_samples": 256,
        "dev_samples": 32,
        "max_steps": 20,
        "eval_trajectories": 16,
        "eval_every": 5,
        "save_every": 5,
        "eval_split": "dev",
    },
    "full": {
        "train_samples": None,
        "dev_samples": None,
        "max_steps": None,
        "eval_trajectories": None,
        "eval_every": 25,
        "save_every": 25,
        "eval_split": "test",
    },
}


@dataclass
class TrainingJob:
    key: str
    method: str
    config: ExperimentConfig
    config_path: Path
    initialize_from: str | None
    evaluate: bool


def resolve_level(args):
    level = dict(LEVELS[args.mode])
    for name in level:
        if name == "train_samples":
            continue
        value = getattr(args, name)
        if value is not None:
            level[name] = None if value == 0 else value
    return level


def resolve_experiment_id(args, init_checkpoint=None):
    """只读来源 run.json 获取身份；预览不加载 checkpoint 权重。"""
    source = init_checkpoint if init_checkpoint is not None else args.init_checkpoint
    if source is not None:
        checkpoint = source.resolve()
        if not args.dry_run and not (checkpoint / "state.pt").is_file():
            raise FileNotFoundError(checkpoint)
        metadata = checkpoint.parent.parent / "run.json"
        if metadata.exists():
            source = json.loads(metadata.read_text(encoding="utf-8"))
            objective = source["config"]["objective"]
            if (
                objective["stage"] != "pretrain"
                or objective["method"] not in DYNAMIC_PRETRAIN_METHODS
            ):
                raise ValueError(
                    f"initialization requires a shared dynamic pretraining run: {metadata}"
                )
            identity = source["config"]["training"].get("experiment_id")
            if not isinstance(identity, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_-]*", identity
            ):
                raise ValueError(f"source run.json has no valid experiment_id: {metadata}")
            if args.run_id is not None and args.run_id != identity:
                raise ValueError(
                    f"--run-id {args.run_id!r} differs from source experiment_id {identity!r}"
                )
            return identity
        if not args.dry_run:
            raise FileNotFoundError(
                f"source run.json required for checkpoint {checkpoint}: {metadata}"
            )
        if args.run_id is None:
            raise ValueError(
                "preview requires explicit --run-id when source run.json is unavailable"
            )
    return args.run_id or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")


def _training_job(config, args, directory, level, initialize_from=None, key=None):
    method, stage = config.objective.method, config.objective.stage
    key = key or f"{method.replace('_', '-')}-{stage}"
    objective = {}
    for name in ("qa_batch_size", "threshold_i", "threshold_d", "threshold_g", "eta"):
        value = getattr(args, name)
        if value is not None:
            objective[name] = value
    if method in DYNAMIC_METHODS and args.append_slots is not None:
        objective["append_slots"] = args.append_slots
    if method == "icae_multi":
        for name in ("icae_min_segments", "icae_max_segments"):
            value = getattr(args, name)
            if value is not None:
                objective[name] = value
    if method == "autocompressors" and args.ac_num_segments is not None:
        objective["ac_num_segments"] = args.ac_num_segments
    uses_bptt = (
        method == "autocompressors"
        and stage in {"pretrain", "lm"}
        or (method in DYNAMIC_METHODS and stage in {"warmup", "policy"})
    )
    if uses_bptt and args.bptt_steps is not None:
        objective["bptt_steps"] = args.bptt_steps or None
    is_pretrain = stage in {"pretrain", "lm"}
    dataset = args.pretrain_data if is_pretrain else args.qa_data
    run_method = (
        "dynamic_pretrain"
        if method in DYNAMIC_PRETRAIN_METHODS and stage == "pretrain"
        else method
    )
    method_directory = f"{run_method.replace('_', '-')}-k{config.model.memory_slots}"
    if args.mode != "full":
        method_directory += f"_{args.mode}"
    experiment_dir = directory / "train" / method_directory
    stage_limit = getattr(args, f"{'pretrain' if is_pretrain else stage}_train_samples")
    if stage_limit is None:
        stage_limit = config.training.max_train_samples
    elif stage_limit == 0:
        stage_limit = None
    limits = [limit for limit in (stage_limit, level["train_samples"]) if limit is not None]
    training = {
        "dataset_dir": str(dataset.resolve()),
        "experiment_dir": str(experiment_dir),
        "experiment_id": args.run_id,
        "experiment_name": f"{method_directory}_{args.run_id}",
        "output_dir": str(experiment_dir / stage),
        "init_checkpoint": str(args.init_checkpoint.resolve())
        if args.init_checkpoint is not None and initialize_from is None
        else config.training.init_checkpoint
        if initialize_from is None
        else None,
        "max_train_samples": min(limits) if limits else None,
        "max_dev_samples": level["dev_samples"],
        "eval_every": level["eval_every"],
        "save_every": level["save_every"],
        "swanlab_project": args.swanlab_project if args.tracking == "online" else None,
        "group": args.group or args.run_id,
        "tags": (f"study:{'main' if args.mode == 'full' else args.mode}",),
    }
    if stage == "pretrain" and method != "autocompressors" and args.lm_ratio is not None:
        training["lm_ratio"] = args.lm_ratio
    if is_pretrain and args.max_input_tokens is not None:
        training["max_input_tokens"] = args.max_input_tokens or None
    if is_pretrain and args.lm_target_tokens is not None:
        training["lm_target_tokens"] = args.lm_target_tokens
    if args.max_qa_input_tokens is not None:
        training["max_qa_input_tokens"] = args.max_qa_input_tokens or None
    for name in (
        "epochs",
        "micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "save_total_limit",
    ):
        value = getattr(args, name)
        if value is not None:
            training[name] = value
    model = (
        replace(config.model, model_name_or_path=args.model_path)
        if args.model_path
        else config.model
    )
    config = replace(
        config,
        model=model,
        objective=replace(config.objective, **objective),
        training=replace(config.training, **training),
    )
    return TrainingJob(
        key=key,
        method=run_method,
        config=config,
        config_path=directory / "plan" / method_directory / f"{stage}.json",
        initialize_from=initialize_from,
        evaluate=stage in {"qa", "policy"}
        or (method == "autocompressors" and is_pretrain),
    )


def build_jobs(args):
    level = resolve_level(args)
    methods = METHODS if args.method == "all" else (args.method,)
    presets = {
        method: load_preset(args.config or Path("configs/v3") / PRESETS[method])
        for method in methods
        if method != "dynamic_pretrain"
    }
    for method, configs in presets.items():
        if any(config.objective.method != method for config in configs):
            raise ValueError(f"preset method differs from requested method: {method}")
        if method in DYNAMIC_METHODS and any(
            config.objective.stage == "pretrain" for config in configs
        ):
            raise ValueError("dynamic pretraining must be declared in dynamic_pretrain.json")
    initializations = (
        [args.init_checkpoint]
        if args.init_checkpoint is not None
        else [
            Path(configs[0].training.init_checkpoint)
            for method, configs in presets.items()
            if method in DYNAMIC_METHODS and configs[0].training.init_checkpoint is not None
        ]
    )
    identities = {
        resolve_experiment_id(args, checkpoint) for checkpoint in dict.fromkeys(initializations)
    }
    if len(identities) > 1:
        raise ValueError("dynamic initialization checkpoints must share one experiment_id")
    args.run_id = next(iter(identities)) if identities else resolve_experiment_id(args)
    directory = args.output_root.resolve() / args.run_id
    jobs = []

    shared = None
    needs_pretrain = args.method == "dynamic_pretrain" or (
        args.init_checkpoint is None
        and any(
            method in DYNAMIC_METHODS and configs[0].training.init_checkpoint is None
            for method, configs in presets.items()
        )
    )
    if needs_pretrain:
        pretrain = load_preset(
            args.config
            if args.method == "dynamic_pretrain" and args.config is not None
            else Path("configs/v3/dynamic_pretrain.json")
        )
        if (
            len(pretrain) != 1
            or pretrain[0].objective.method not in DYNAMIC_PRETRAIN_METHODS
            or pretrain[0].objective.stage != "pretrain"
        ):
            raise ValueError("dynamic_pretrain.json must declare only the pretrain stage")
        pretrain_job = _training_job(pretrain[0], args, directory, level, key="dynamic-pretrain")
        jobs.append(pretrain_job)
        shared = pretrain_job.key
    for method, configs in presets.items():
        initialize_from = (
            shared
            if method in DYNAMIC_METHODS and configs[0].training.init_checkpoint is None
            else None
        )
        for config in configs:
            job = _training_job(config, args, directory, level, initialize_from)
            if shared is not None and initialize_from == shared:
                # 与运行时初始化保持相同契约；激活重算不影响已有权重。
                source_model = replace(
                    pretrain_job.config.model,
                    gradient_checkpointing=job.config.model.gradient_checkpointing,
                )
                if source_model != job.config.model:
                    raise ValueError(
                        "dynamic model differs from automatic shared pretraining; "
                        "prepare a matching checkpoint with --method dynamic_pretrain --config "
                        "and pass --init-checkpoint"
                    )
            jobs.append(job)
            initialize_from = job.key
    return directory, level, jobs
