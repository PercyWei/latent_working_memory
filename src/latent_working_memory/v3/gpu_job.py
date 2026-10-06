"""GPU 分阶段任务编排：终端与网站共用入口控制试跑规模和完整实验。"""

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
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
    parser.add_argument("--method", choices=(*METHODS, "dynamic", "all"), default="dynamic")
    parser.add_argument(
        "--stage", choices=("auto", "pretrain", "qa", "warmup", "policy", "lm"), default="auto"
    )
    parser.add_argument(
        "--pretrain-data", type=Path, default=Path("data/fineweb-4096-doc100k_20260910/semantic")
    )
    parser.add_argument(
        "--long-pretrain-data",
        type=Path,
        help="ICAE 与 AutoCompressors 预设所需长文本 TextSample 目录",
    )
    parser.add_argument(
        "--qa-data", type=Path, default=Path("data/fineweb-factqa-train1000_20260930")
    )
    parser.add_argument(
        "--init-checkpoint", type=Path, help="单阶段 qa/warmup/policy 的初始化 checkpoint"
    )
    parser.add_argument("--model-path", help="默认 Qwen3-4B；可改为共享盘模型目录")
    parser.add_argument(
        "--gpus", type=physical_gpus, default="0,1", help="逗号分隔的物理 GPU 编号，不允许重复"
    )
    parser.add_argument(
        "--epochs", type=positive_count, help="覆盖所选阶段的训练轮数，默认沿用预设"
    )
    parser.add_argument("--global-batch-size", type=positive_count)
    parser.add_argument("--qa-batch-size", type=positive_count)
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
        "--run-id", default=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/v3"))
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印解析后的计划，不加载数据/模型，不连接 SwanLab"
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("run-id must contain only letters, digits, '-' and '_'")
    if args.stage != "auto" and args.method in {"all", "dynamic"}:
        parser.error("single-stage jobs require one specific --method")
    if args.init_checkpoint is not None and args.stage not in {"qa", "warmup", "policy"}:
        parser.error("--init-checkpoint is for a single qa/warmup/policy stage")
    if args.stage in {"qa", "warmup", "policy"} and args.init_checkpoint is None:
        parser.error("this stage requires --init-checkpoint")
    return args


def resolve_level(args):
    level = dict(LEVELS[args.mode])
    for name in level:
        value = getattr(args, name)
        if value is not None:
            level[name] = None if value == 0 else value
    return level


def build_jobs(args):
    level = resolve_level(args)
    series = f"capacity-{args.mode}_{args.run_id}"
    directory = args.output_root.resolve() / series
    group = args.group or series
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
        is_pretrain = stage in {"pretrain", "lm"}
        if is_pretrain:
            dataset = args.pretrain_data if method in DYNAMIC_METHODS else args.long_pretrain_data
            if dataset is None:
                raise ValueError(
                    f"{method}/{stage} requires --long-pretrain-data; keep full-history training lengths"
                )
        else:
            dataset = args.qa_data
        training = {
            "dataset_dir": str(dataset.resolve()),
            "output_dir": str(directory / "train" / f"{key}-k64_{args.mode}_{args.run_id}"),
            "init_checkpoint": str(args.init_checkpoint.resolve())
            if args.init_checkpoint
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
        for name in ("epochs", "global_batch_size"):
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
            TrainingJob(key, config, directory / "plan" / f"{key}.json", initialize_from, evaluate)
        )
        return key

    if args.stage != "auto":
        append(methods[0], args.stage, evaluate=args.stage in {"qa", "warmup", "policy", "lm"})
    else:
        shared = None
        if any(method in DYNAMIC_METHODS for method in methods):
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


def training_command(job, gpu_count, max_steps):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        "-m",
        "latent_working_memory.v3.train",
        "--config",
        str(job.config_path),
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
    gpu_count = len(args.gpus.split(","))
    plan = {
        "mode": args.mode,
        "method": args.method,
        "stage": args.stage,
        "directory": str(directory),
        "gpus": args.gpus,
        "limits": level,
        "max_new_tokens": args.max_new_tokens,
        "jobs": [
            {
                "key": job.key,
                "initialize_from": job.initialize_from,
                "evaluate": job.evaluate,
                "config": job.config.to_dict(),
                "command": training_command(job, gpu_count, level["max_steps"]),
            }
            for job in jobs
        ],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return plan
    if directory.exists():
        raise ValueError(f"job directory already exists: {directory}; choose a new --run-id")
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpus,
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    if args.tracking == "online":
        environment["SWANLAB_API_KEY"] = swanlab_api_key()
    # 只检查本任务使用的规范数据入口；不扫描下载分片或创建派生数据副本。
    datasets = {
        Path(job.config.training.dataset_dir): job.config.objective.stage not in {"pretrain", "lm"}
        for job in jobs
    }
    if any(job.evaluate for job in jobs):
        datasets[args.qa_data.resolve()] = True
    for dataset, is_qa in datasets.items():
        names = ["train.jsonl", "dev.jsonl", "test.jsonl"] + (["preparation.json"] if is_qa else [])
        for name in names:
            if not (dataset / name).is_file():
                raise FileNotFoundError(f"required dataset entry is missing: {dataset / name}")
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    (directory / "plan").mkdir(parents=True)
    save_json(directory / "plan" / "job.json", plan)
    evaluation_environment = dict(environment, CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0])
    checkpoints, summaries = {}, []
    try:
        for job in jobs:
            if job.initialize_from is not None:
                job.config = replace(
                    job.config,
                    training=replace(
                        job.config.training, init_checkpoint=str(checkpoints[job.initialize_from])
                    ),
                )
            save_json(job.config_path, job.config.to_dict())
            execute(
                training_command(job, gpu_count, level["max_steps"]),
                environment,
                directory / "plan" / f"{job.key}-train.log",
            )
            result = json.loads(
                (Path(job.config.training.output_dir) / "training-result.json").read_text()
            )
            checkpoint = Path(result["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"training did not save its reported checkpoint: {checkpoint}"
                )
            checkpoints[job.key] = checkpoint
            if job.evaluate:
                output = directory / "eval" / job.key
                execute(
                    evaluation_command(args, level, checkpoint, output),
                    evaluation_environment,
                    directory / "plan" / f"{job.key}-eval.log",
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
                    str(directory / "compare"),
                ],
                environment,
                directory / "plan" / "compare.log",
            )
    except Exception as error:
        save_json(
            directory / "plan" / "result.json",
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
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "summaries": [str(value) for value in summaries],
    }
    save_json(directory / "plan" / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    args = parse_args(argv)
    run_job(args)


if __name__ == "__main__":
    main()
