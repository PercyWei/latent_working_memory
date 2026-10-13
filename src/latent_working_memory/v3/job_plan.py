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
    ExperimentConfig,
    load_experiment,
    load_preset,
)


PRESETS = {
    "icae_single": "icae_single.json",
    "icae_multi": "icae_multi.json",
    "autocompressors": "autocompressors.json",
    "memory_change": "memory_change.json",
    "information_loss": "information_loss.json",
    "dynamic_pretrain": "dynamic_pretrain.json",
}
LEVELS = {
    "smoke": {
        "train_samples": 16,
        "max_dev_samples": 4,
        "max_test_samples": 2,
        "max_steps": 2,
        "dev_every": 1,
        "save_every": 1,
    },
    "full": {
        "train_samples": None,
        "max_dev_samples": None,
        "max_test_samples": None,
        "max_steps": None,
        "dev_every": 25,
        "save_every": None,
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
    for name, value in vars(args).items():
        if name in level and (
            value is not None or name in {"max_dev_samples", "max_test_samples", "max_steps"}
        ):
            level[name] = value
    return level


def resolve_experiment_id(args):
    """校验初始化来源；优先使用显式 run-id，否则沿用来源编号。"""
    source = args.checkpoint
    if source is not None:
        checkpoint = source.resolve()
        if not (checkpoint / "state.pt").is_file():
            raise FileNotFoundError(checkpoint)
        metadata = checkpoint.parent.parent / "run.json"
        if not metadata.exists():
            raise FileNotFoundError(
                f"source run.json required for checkpoint {checkpoint}: {metadata}"
            )
        source = json.loads(metadata.read_text(encoding="utf-8"))
        objective = source["config"]["objective"]
        if objective["stage"] != "pretrain" or objective["method"] not in DYNAMIC_PRETRAIN_METHODS:
            raise ValueError(
                f"initialization requires a shared dynamic pretraining run: {metadata}"
            )
        identity = source["config"]["training"].get("experiment_id")
        if not isinstance(identity, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", identity
        ):
            raise ValueError(f"source run.json has no valid experiment_id: {metadata}")
        return args.run_id or identity
    return args.run_id or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")


def _training_job(config, args, directory, level, method_directory, initialize_from=None):
    method, stage = config.objective.method, config.objective.stage
    key = (
        "dynamic-pretrain"
        if args.method == "dynamic_pretrain"
        else f"{method.replace('_', '-')}-{stage}"
    )
    objective = {}
    if args.objective_seed is not None:
        objective["seed"] = args.objective_seed
    for name in (
        "qa_batch_size",
        "task_qa_count",
        "gate_qa_count",
        "task_new_ratio",
        "threshold_i",
        "threshold_d",
        "threshold_g",
        "eta",
    ):
        value = getattr(args, name)
        if value is not None:
            objective[name] = value
    if method in DYNAMIC_METHODS and stage != "pretrain" and args.append_slots is not None:
        objective["append_slots"] = args.append_slots
    if method in DYNAMIC_METHODS and stage != "pretrain" and args.writer_mode is not None:
        objective["writer_mode"] = args.writer_mode
    if method in DYNAMIC_METHODS and stage != "pretrain" and args.tag_tokens is not None:
        if objective.get("writer_mode", config.objective.writer_mode) != "tag":
            raise ValueError("--tag-tokens requires writer_mode=tag")
        objective["tag_tokens"] = args.tag_tokens
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
    if uses_bptt and "bptt_steps" in vars(args):
        objective["bptt_steps"] = args.bptt_steps
    resolved_objective = replace(config.objective, **objective)
    is_pretrain = stage in {"pretrain", "lm"}
    dataset = args.pretrain_data if is_pretrain else args.qa_data
    experiment_dir = directory / "train" / method_directory
    stage_limit = getattr(
        args,
        f"{'pretrain' if is_pretrain else stage}_train_samples",
        config.training.max_train_samples,
    )
    limits = [limit for limit in (stage_limit, level["train_samples"]) if limit is not None]
    dev_limit = level["max_dev_samples"]
    if dev_limit is None and "max_dev_samples" not in vars(args):
        dev_limit = config.training.max_dev_samples
    training = {
        "dataset_dir": str(dataset.resolve()),
        "experiment_dir": str(experiment_dir),
        "experiment_id": args.run_id,
        "experiment_name": f"{method_directory}_{args.run_id}",
        "output_dir": str(experiment_dir / stage),
        "init_checkpoint": str(args.checkpoint.resolve())
        if args.checkpoint is not None and initialize_from is None
        else config.training.init_checkpoint
        if initialize_from is None
        else None,
        "max_train_samples": min(limits) if limits else None,
        "max_dev_samples": dev_limit,
        "dev_every": level["dev_every"],
        "save_every": config.training.save_every
        if level["save_every"] is None
        else level["save_every"],
        "swanlab_project": args.swanlab_project if args.tracking == "online" else None,
        "group": args.group or args.run_id,
        "tags": (f"study:{'main' if args.mode == 'full' else args.mode}",),
    }
    if args.train_seed is not None:
        training["seed"] = args.train_seed
    if stage == "pretrain" and method != "autocompressors" and args.lm_ratio is not None:
        training["lm_ratio"] = args.lm_ratio
    if is_pretrain and "max_input_tokens" in vars(args):
        training["max_input_tokens"] = args.max_input_tokens
    if is_pretrain and args.lm_target_tokens is not None:
        training["lm_target_tokens"] = args.lm_target_tokens
    if "max_qa_input_tokens" in vars(args):
        training["max_qa_input_tokens"] = args.max_qa_input_tokens
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
        objective=resolved_objective,
        training=replace(config.training, **training),
    )
    return TrainingJob(
        key=key,
        method=args.method,
        config=config,
        config_path=directory / "plan" / method_directory / f"{stage}.json",
        initialize_from=initialize_from,
        evaluate=not args.train_only
        and (stage in {"qa", "policy"} or (method == "autocompressors" and is_pretrain)),
    )


def build_jobs(args):
    level = resolve_level(args)
    configs = load_preset(args.config or Path("configs/v3") / PRESETS[args.method])
    expected_method = "dynamic" if args.method == "dynamic_pretrain" else args.method
    if any(config.objective.method != expected_method for config in configs):
        raise ValueError(f"preset method differs from requested method: {args.method}")
    if args.method == "dynamic_pretrain" and (
        len(configs) != 1 or configs[0].objective.stage != "pretrain"
    ):
        raise ValueError("dynamic_pretrain requires only the pretrain stage")
    if args.method in (*DYNAMIC_METHODS, "dynamic_pretrain"):
        if configs[0].objective.stage == "pretrain":
            if args.checkpoint is not None:
                print(
                    "[ignored] --checkpoint is ignored because stages include pretrain", flush=True
                )
            args.checkpoint = None
            configs = (
                replace(configs[0], training=replace(configs[0].training, init_checkpoint=None)),
                *configs[1:],
            )
        elif args.checkpoint is None:
            raise ValueError("dynamic stages without pretrain require --checkpoint")
    args.run_id = resolve_experiment_id(args)
    directory = args.output_root.resolve() / args.run_id
    method_directory = f"{args.method.replace('_', '-')}-k{configs[0].model.memory_slots}"
    if args.method in DYNAMIC_METHODS:
        objective = configs[-1].objective
        posttraining = objective.stage != "pretrain"
        append_slots = (
            args.append_slots
            if posttraining and args.append_slots is not None
            else objective.append_slots
        )
        writer_mode = (
            args.writer_mode
            if posttraining and args.writer_mode is not None
            else objective.writer_mode
        )
        method_directory += f"+{append_slots}-{writer_mode.replace('_', '-')}"
    if args.mode != "full":
        method_directory += f"_{args.mode}"
    jobs, initialize_from = [], None
    for config in configs:
        job = _training_job(config, args, directory, level, method_directory, initialize_from)
        jobs.append(job)
        initialize_from = job.key
    return directory, level, jobs


def checkpoint_stage_directory(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if not (checkpoint / "state.pt").is_file():
        raise FileNotFoundError(checkpoint / "state.pt")
    stage_directory = checkpoint.parent.parent
    if (
        re.fullmatch(r"global_step_[0-9]+", checkpoint.name) is None
        or checkpoint.parent.name != "checkpoints"
        or stage_directory.parent.parent.name != "train"
    ):
        raise ValueError(
            "--checkpoint requires <series>/train/<method>/<stage>/checkpoints/global_step_<step>"
        )
    return stage_directory


def build_resume_jobs(args):
    """沿用原计划，恢复当前阶段并衔接尚未启动的后续阶段。"""
    stage_directory = checkpoint_stage_directory(args.checkpoint)
    method_directory = stage_directory.parent
    directory = method_directory.parent.parent
    plan_directory = directory / "plan" / method_directory.name
    plan = json.loads((plan_directory / "job.json").read_text(encoding="utf-8"))
    config = load_experiment(stage_directory / "config.json")
    experiment = json.loads((method_directory / "experiment.json").read_text(encoding="utf-8"))
    if next(reversed(experiment["stages"])) != config.objective.stage:
        raise ValueError("cannot resume an earlier stage after a successor has started")
    first = next(
        index
        for index, item in enumerate(plan["jobs"])
        if item["config"]["objective"]["stage"] == config.objective.stage
    )
    jobs = []
    for index, item in enumerate(plan["jobs"][first:]):
        path = Path(item["config_path"])
        current = config if index == 0 else load_experiment(path)
        if index > 0 and Path(current.training.output_dir).exists():
            raise ValueError("cannot resume while a later stage output already exists")
        jobs.append(
            TrainingJob(
                key=item["key"],
                method=plan["method"],
                config=current,
                config_path=path,
                initialize_from=None if index == 0 else item["initialize_from"],
                evaluate=item["evaluate"],
            )
        )
    args.method = plan["method"]
    args.mode = plan["mode"]
    args.run_id = config.training.experiment_id
    args.tracking = "online" if config.training.swanlab_project is not None else "disabled"
    return directory, plan["limits"], jobs
