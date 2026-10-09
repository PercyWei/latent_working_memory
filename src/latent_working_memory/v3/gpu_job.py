"""GPU 实验入口与公共执行：启动连续训练、评估并收集结果。"""

import argparse
from dataclasses import replace
from itertools import groupby
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from latent_working_memory.v3 import job_plan
from latent_working_memory.v3.config import DYNAMIC_METHODS, METHODS
from latent_working_memory.v3.tracking_credentials import swanlab_api_key


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


def probability(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be a finite probability in [0, 1]")
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
    parser.add_argument("--mode", choices=job_plan.LEVELS, default="smoke")
    parser.add_argument(
        "--method", choices=(*METHODS, "dynamic_pretrain", "dynamic", "all"), default="all"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="替换所选单方法的 JSON 预设；须配合 --method，使用 objective.stages",
    )
    parser.add_argument(
        "--pretrain-data",
        type=Path,
        default=Path("data/fineweb-multisegment-k512-seg1to3x_train32k_20261008"),
        help="共用 FineWeb 多段文本数据根目录，包含 train/dev/test.jsonl 和 preparation.json",
    )
    parser.add_argument(
        "--qa-data",
        type=Path,
        default=Path("data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008"),
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
        "--lm-ratio",
        type=probability,
        help="AE＋LM 预训练中每条来源选择 LM 的概率，默认 0.5；AutoCompressors 始终使用 LM",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=positive_count,
        help="预训练压缩输入的 token 上限，默认 8192；QA 阶段沿用数据中的段界",
    )
    parser.add_argument(
        "--lm-target-tokens",
        type=positive_count,
        help="预训练 LM 目标的真实 token 数，默认 512；续文候选不足时沿用任务回退规则",
    )
    parser.add_argument(
        "--icae-segment-ratio",
        type=positive_count,
        help="ICAE-multi 预训练块长为 memory_slots × ratio，默认 3；QA 保留原段界",
    )
    parser.add_argument(
        "--bptt-steps",
        type=bounded_count,
        help="动态 warmup/policy 每个反传窗口的写入轮数，0 表示完整 BPTT；默认完整展开",
    )
    parser.add_argument(
        "--append-slots",
        type=positive_count,
        help="动态方法每次追加的 slots 数，默认沿用预设 32；首次写入为 512 slots",
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
    if args.config is not None and args.method in {"all", "dynamic"}:
        parser.error("--config requires a single --method; all and dynamic use default presets")
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
    if args.bptt_steps is not None and args.method not in (*DYNAMIC_METHODS, "dynamic", "all"):
        parser.error("--bptt-steps is only supported by dynamic warmup/policy")
    return args


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
    directory, level, jobs = job_plan.build_jobs(args)
    training_runs = [
        list(stages)
        for _, stages in groupby(jobs, key=lambda job: job.config.training.experiment_dir)
    ]
    gpu_count = len(args.gpus.split(","))
    plan_directory = directory / "plan"
    compare_directory = directory / "compare"
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
                "config_path": str(job.config_path),
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
                "log_path": str(stages[0].config_path.parent / "train.log"),
            }
            for stages in training_runs
        ],
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return plan
    targets = [stages[0].config_path.parent for stages in training_runs]
    targets.extend(Path(stages[0].config.training.experiment_dir) for stages in training_runs)
    targets.extend(path.parent for path in evaluation_directories.values())
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
        required_files.update(dataset / f"{split}.jsonl" for split in ("train", "dev", "test"))
    if any(job.evaluate for job in jobs):
        required_files.update(
            args.qa_data.resolve() / name
            for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "preparation.json")
        )
    for path in sorted(required_files):
        if not path.is_file():
            raise FileNotFoundError(f"required dataset entry is missing: {path}")
    checkpoints, summaries = {}, []

    def save_method_result(stages, status, error=None):
        keys = {job.key for job in stages}
        outputs = {
            evaluation_directories[job.key] / "summary.json" for job in stages if job.evaluate
        }
        record = {
            "status": status,
            "mode": args.mode,
            "experiment_id": args.run_id,
            "checkpoints": {key: str(value) for key, value in checkpoints.items() if key in keys},
            "summaries": [str(path) for path in summaries if path in outputs],
        }
        if error is not None:
            record["error"] = str(error)
        save_json(stages[0].config_path.parent / "result.json", record)

    for stages, training_run in zip(training_runs, plan["training_runs"], strict=True):
        first = stages[0]
        keys = {job.key for job in stages}
        first.config_path.parent.mkdir(parents=True)
        save_json(
            first.config_path.parent / "job.json",
            {
                **plan,
                "method": first.method,
                "plan_directory": str(first.config_path.parent),
                "jobs": [job for job in plan["jobs"] if job["key"] in keys],
                "training_runs": [training_run],
            },
        )
        for job in stages:
            save_json(job.config_path, job.config.to_dict())
        save_method_result(stages, "pending")
    print(
        f"[start] method={args.method} mode={args.mode} runs={len(training_runs)} "
        f"stages={len(jobs)} gpus={args.gpus} plan={plan_directory}",
        flush=True,
    )
    evaluation_environment = dict(environment, CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0])
    active_stages, comparing = None, False
    try:
        for stages, training_run in zip(training_runs, plan["training_runs"], strict=True):
            active_stages = stages
            save_method_result(stages, "running")
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
                        job.config_path.parent / "eval.log",
                    )
                    summaries.append(output / "summary.json")
            save_method_result(stages, "finished")
            active_stages = None
        if len(summaries) > 1:
            comparing = True
            compare_directory.mkdir(parents=True)
            save_json(
                compare_directory / "result.json",
                {"status": "running", "summaries": [str(path) for path in summaries]},
            )
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
                compare_directory / "compare.log",
            )
            save_json(
                compare_directory / "result.json",
                {"status": "finished", "summaries": [str(path) for path in summaries]},
            )
            comparing = False
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
        if active_stages is not None:
            save_method_result(active_stages, "failed", error)
        if comparing:
            save_json(
                compare_directory / "result.json",
                {
                    "status": "failed",
                    "error": str(error),
                    "summaries": [str(path) for path in summaries],
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
    print(
        f"[finished] method={args.method} mode={args.mode} "
        f"checkpoints={len(checkpoints)} evaluations={len(summaries)} "
        f"plan={plan_directory}",
        flush=True,
    )
    return result


def main(argv=None):
    args = parse_args(argv)
    run_job(args)


if __name__ == "__main__":
    main()
