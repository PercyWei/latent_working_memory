"""GPU 实验编排：同一方法连续训练，终端与网站共用运行入口。"""

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import groupby
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

from latent_working_memory.v3.config import (
    DYNAMIC_METHODS,
    METHODS,
    ExperimentConfig,
    load_experiment,
)
from latent_working_memory.v3.tracking_credentials import swanlab_api_key


PRESETS = {
    "icae_single": "icae_single_pretrain.json",
    "icae_multi": "icae_multi_pretrain.json",
    "autocompressors": "autocompressors_lm.json",
    "memory_change": "dynamic_pretrain.json",
    "information_loss": "dynamic_pretrain.json",
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
    config: ExperimentConfig
    config_path: Path
    initialize_from: str | None
    evaluate: bool


def bounded_count(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0; 0 means unlimited")
    return number


def positive_count(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def physical_gpus(value):
    devices = [device.strip() for device in value.split(",")]
    if any(re.fullmatch(r"[0-9]+", device) is None for device in devices) or len(
        {int(device) for device in devices}
    ) != len(devices):
        raise argparse.ArgumentTypeError("select distinct non-negative physical GPU indices")
    return ",".join(str(int(device)) for device in devices)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=LEVELS, default="smoke")
    parser.add_argument(
        "--method", choices=(*METHODS, "shared_pretrain", "dynamic", "all"), default="all"
    )
    parser.add_argument(
        "--pretrain-data",
        type=Path,
        default=Path("data/fineweb-reconstruction-k512-doc100k_20260917"),
        help="共用 FineWeb reconstruction 索引根目录，包含 single、multi 和 preparation.json",
    )
    parser.add_argument(
        "--qa-data", type=Path, default=Path("data/fineweb-factqa-train1000_20260930")
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="动态方法使用共享预训练 checkpoint，连续执行 warmup/policy 与最终评估",
    )
    parser.add_argument(
        "--model-path", help="覆盖预设模型位置，当前预设为 ~/models/Qwen3-4B-Instruct-2507"
    )
    parser.add_argument(
        "--gpus", type=physical_gpus, default="0,1", help="逗号分隔的物理 GPU 编号，不允许重复"
    )
    parser.add_argument(
        "--epochs", type=positive_count, help="覆盖所选阶段的训练轮数，默认沿用预设"
    )
    parser.add_argument(
        "--micro-batch-size-per-gpu", type=positive_count, help="每张卡一次并行处理的样本/轨迹数"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=positive_count,
        help="每次优化器更新前累积的 microbatch 数",
    )
    parser.add_argument("--qa-batch-size", type=positive_count)
    parser.add_argument(
        "--append-slots",
        type=positive_count,
        help="动态方法每次追加的 slots 数，默认沿用预设 8；首次写入仍为 64 slots",
    )
    for name in ("train-samples", "dev-samples", "max-steps", "eval-trajectories"):
        parser.add_argument(f"--{name}", type=bounded_count, help="覆盖运行档位，0 表示不限")
    for name in ("eval-every", "save-every"):
        parser.add_argument(f"--{name}", type=positive_count)
    parser.add_argument("--eval-split", choices=("dev", "test"))
    parser.add_argument("--max-new-tokens", type=positive_count, default=64)
    for name in ("threshold-i", "threshold-d", "threshold-g", "eta"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--tracking", choices=("online", "disabled"), default="online")
    parser.add_argument("--swanlab-project", default="latent-working-memory-v3")
    parser.add_argument("--group", help="显式实验系列 group；默认使用本次运行标识")
    parser.add_argument(
        "--run-id",
        help="产物目录名和默认 group；默认上海时间 YYYYMMDD-HHMMSS，动态后训练继承预训练标识",
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/v3"))
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印解析后的计划，不加载数据/模型，不连接 SwanLab"
    )
    args = parser.parse_args(argv)
    if args.run_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("run-id must contain only letters, digits, '-' and '_'")
    if args.init_checkpoint is not None and args.method not in (*DYNAMIC_METHODS, "dynamic"):
        parser.error("--init-checkpoint is only supported by dynamic methods")
    if args.append_slots is not None and args.method in {
        "icae_single",
        "icae_multi",
        "autocompressors",
    }:
        parser.error("--append-slots is only supported by dynamic methods")
    return args


def resolve_level(args):
    level = dict(LEVELS[args.mode])
    for name in level:
        value = getattr(args, name)
        if value is not None:
            level[name] = None if value == 0 else value
    return level


def resolve_experiment_id(args):
    """只读来源 run.json 获取身份；预览不加载 checkpoint 权重。"""
    if args.init_checkpoint is not None:
        checkpoint = args.init_checkpoint.resolve()
        if not args.dry_run and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        metadata = checkpoint.parent.parent / "run.json"
        if metadata.exists():
            source = json.loads(metadata.read_text(encoding="utf-8"))
            objective = source["config"]["objective"]
            if objective["stage"] != "pretrain" or objective["method"] not in DYNAMIC_METHODS:
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


def build_jobs(args):
    args.run_id = resolve_experiment_id(args)
    level = resolve_level(args)
    directory = args.output_root.resolve() / args.run_id
    plan_directory = directory / "plan" / args.method.replace("_", "-")
    group = args.group or args.run_id
    methods = (
        METHODS
        if args.method == "all"
        else DYNAMIC_METHODS
        if args.method == "dynamic"
        else (args.method,)
    )
    jobs = []

    def append(method, stage, initialize_from=None, key=None, evaluate=False):
        key = key or f"{method.replace('_', '-')}-{stage}"
        config = load_experiment(Path("configs/v3") / PRESETS[method])
        objective = {"method": method, "stage": stage}
        for name in ("qa_batch_size", "threshold_i", "threshold_d", "threshold_g", "eta"):
            value = getattr(args, name)
            if value is not None:
                objective[name] = value
        if method in DYNAMIC_METHODS and args.append_slots is not None:
            objective["append_slots"] = args.append_slots
        is_pretrain = stage in {"pretrain", "lm"}
        dataset = args.pretrain_data if is_pretrain else args.qa_data
        root_method = (
            "shared-pretrain"
            if method in DYNAMIC_METHODS and stage == "pretrain"
            else method.replace("_", "-")
        )
        suffix = args.run_id if args.mode == "full" else f"{args.mode}_{args.run_id}"
        experiment_dir = (
            directory / "train" / f"{root_method}-k{config.model.memory_slots}_{suffix}"
        )
        training = {
            "dataset_dir": str(dataset.resolve()),
            "experiment_dir": str(experiment_dir),
            "experiment_id": args.run_id,
            "output_dir": str(experiment_dir / stage),
            "init_checkpoint": str(args.init_checkpoint.resolve())
            if args.init_checkpoint is not None and initialize_from is None
            else None,
            "max_train_samples": level["train_samples"],
            "max_dev_samples": level["dev_samples"],
            "eval_every": level["eval_every"],
            "save_every": level["save_every"],
            "swanlab_project": args.swanlab_project if args.tracking == "online" else None,
            "group": group,
            "tags": (f"study:{'main' if args.mode == 'full' else args.mode}",),
        }
        # QA 阶段不使用预训练输入长度筛选；真实段界与题池保持原样。
        for name in ("epochs", "micro_batch_size_per_gpu", "gradient_accumulation_steps"):
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
        jobs.append(
            TrainingJob(key, config, plan_directory / f"{key}.json", initialize_from, evaluate)
        )
        return key

    if args.method == "shared_pretrain":
        append("memory_change", "pretrain", key="dynamic-pretrain")
    else:
        shared = None
        if args.init_checkpoint is None and any(method in DYNAMIC_METHODS for method in methods):
            shared = append("memory_change", "pretrain", key="dynamic-pretrain")
        for method in methods:
            if method in DYNAMIC_METHODS:
                warmup = append(method, "warmup", shared)
                append(method, "policy", warmup, evaluate=True)
            elif method == "autocompressors":
                append(method, "lm", evaluate=True)
            else:
                pretrain = append(method, "pretrain")
                append(method, "qa", pretrain, evaluate=True)
    return directory, level, jobs


def training_command(jobs, gpu_count, max_steps):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        "-m",
        "latent_working_memory.v3.train",
        "--config",
        *(str(job.config_path) for job in jobs),
        "--device",
        "cuda",
    ]
    if max_steps is not None:
        command.extend(("--stop-after-steps", str(max_steps)))
    return command


def evaluation_command(args, level, checkpoint, output):
    command = [
        sys.executable,
        "-m",
        "latent_working_memory.v3.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--dataset-dir",
        str(args.qa_data.resolve()),
        "--output-dir",
        str(output),
        "--split",
        level["eval_split"],
        "--device",
        "cuda",
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if level["eval_trajectories"] is not None:
        command.extend(("--max-trajectories", str(level["eval_trajectories"])))
    if args.tracking == "online":
        command.append("--log-to-swanlab")
    return command


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def execute(command, environment, log_path):
    print(f"$ {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def run_job(args):
    directory, level, jobs = build_jobs(args)
    training_runs = [
        list(stages)
        for _, stages in groupby(jobs, key=lambda job: job.config.training.experiment_dir)
    ]
    gpu_count = len(args.gpus.split(","))
    invocation = args.method.replace("_", "-")
    plan_directory = directory / "plan" / invocation
    compare_directory = directory / "compare" / invocation
    evaluation_directories = {
        job.key: directory
        / "eval"
        / Path(job.config.training.experiment_dir).name
        / job.config.objective.stage
        for job in jobs
        if job.evaluate
    }
    plan = {
        "mode": args.mode,
        "method": args.method,
        "experiment_id": args.run_id,
        "directory": str(directory),
        "plan_directory": str(plan_directory),
        "gpus": args.gpus,
        "limits": level,
        "max_new_tokens": args.max_new_tokens,
        "jobs": [
            {
                "key": job.key,
                "initialize_from": job.initialize_from,
                "evaluate": job.evaluate,
                "batching": {
                    "world_size": gpu_count,
                    "micro_batch_size_per_gpu": job.config.training.micro_batch_size_per_gpu,
                    "gradient_accumulation_steps": job.config.training.gradient_accumulation_steps,
                    "global_batch_size": job.config.training.global_batch_size(gpu_count),
                },
                "config": job.config.to_dict(),
            }
            for job in jobs
        ],
        "training_runs": [
            {
                "stages": [job.key for job in stages],
                "command": training_command(stages, gpu_count, level["max_steps"]),
                "log_path": str(
                    plan_directory
                    / f"{Path(stages[0].config.training.experiment_dir).name}-train.log"
                ),
            }
            for stages in training_runs
        ],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return plan
    targets = [plan_directory]
    targets.extend(Path(job.config.training.output_dir) for job in jobs)
    targets.extend(job.config_path for job in jobs)
    targets.extend(evaluation_directories.values())
    if len(evaluation_directories) > 1:
        targets.append(compare_directory)
    for path in targets:
        if path.exists():
            raise ValueError(f"job output already exists: {path}; refusing to overwrite")
    environment = dict(
        os.environ,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=args.gpus,
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    if args.tracking == "online":
        environment["SWANLAB_API_KEY"] = swanlab_api_key()
    # 只检查本任务使用的规范数据入口；不扫描下载分片或创建派生数据副本。
    required_files = set()
    for job in jobs:
        dataset = Path(job.config.training.dataset_dir)
        is_qa = job.config.objective.stage not in {"pretrain", "lm"}
        view = job.config.training.pretrain_data_view
        if is_qa or view != "text_samples":
            required_files.add(dataset / "preparation.json")
        if not is_qa and view != "text_samples":
            dataset /= {
                "reconstruction_single": "single",
                "reconstruction_first_write": "multi",
            }[view]
        required_files.update(dataset / f"{split}.jsonl" for split in ("train", "dev", "test"))
    if any(job.evaluate for job in jobs):
        required_files.update(
            args.qa_data.resolve() / name
            for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "preparation.json")
        )
    for path in sorted(required_files):
        if not path.is_file():
            raise FileNotFoundError(f"required dataset entry is missing: {path}")
    plan_directory.mkdir(parents=True)
    save_json(plan_directory / "job.json", plan)
    evaluation_environment = dict(environment, CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0])
    checkpoints, summaries = {}, []
    try:
        for stages, training_run in zip(training_runs, plan["training_runs"], strict=True):
            first = stages[0]
            if first.initialize_from is not None:
                first.config = replace(
                    first.config,
                    training=replace(
                        first.config.training,
                        init_checkpoint=str(checkpoints[first.initialize_from]),
                    ),
                )
            for job in stages:
                save_json(job.config_path, job.config.to_dict())
            execute(
                training_run["command"],
                environment,
                Path(training_run["log_path"]),
            )
            for job in stages:
                result = json.loads(
                    (Path(job.config.training.output_dir) / "training-result.json").read_text()
                )
                checkpoint = Path(result["checkpoint"])
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"training did not save its reported checkpoint: {checkpoint}"
                    )
                save_json(
                    job.config_path,
                    json.loads((Path(job.config.training.output_dir) / "config.json").read_text()),
                )
                checkpoints[job.key] = checkpoint
                if job.evaluate:
                    output = evaluation_directories[job.key]
                    execute(
                        evaluation_command(args, level, checkpoint, output),
                        evaluation_environment,
                        plan_directory / f"{job.key}-eval.log",
                    )
                    summaries.append(output / "summary.json")
        if len(summaries) > 1:
            execute(
                [
                    sys.executable,
                    "-m",
                    "latent_working_memory.v3.compare",
                    *(str(path) for path in summaries),
                    "--output-dir",
                    str(compare_directory),
                ],
                environment,
                plan_directory / "compare.log",
            )
    except Exception as error:
        # 同一训练进程可能在后续阶段失败，仍记录它已完成阶段的产物。
        for job in jobs:
            result_path = Path(job.config.training.output_dir) / "training-result.json"
            if job.key not in checkpoints and result_path.is_file():
                checkpoint = Path(json.loads(result_path.read_text())["checkpoint"])
                if checkpoint.is_file():
                    save_json(
                        job.config_path,
                        json.loads((result_path.parent / "config.json").read_text()),
                    )
                    checkpoints[job.key] = checkpoint
        save_json(
            plan_directory / "result.json",
            {
                "status": "failed",
                "error": str(error),
                "checkpoints": {key: str(value) for key, value in checkpoints.items()},
                "summaries": [str(value) for value in summaries],
            },
        )
        raise
    result = {
        "status": "finished",
        "mode": args.mode,
        "experiment_id": args.run_id,
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "summaries": [str(value) for value in summaries],
    }
    save_json(plan_directory / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    args = parse_args(argv)
    run_job(args)


if __name__ == "__main__":
    main()
