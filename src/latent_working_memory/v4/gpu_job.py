"""v4 GPU 入口：创建训练运行、恢复 checkpoint 或执行单条生成。"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

from latent_working_memory.v4.checkpoint import read_checkpoint
from latent_working_memory.v4.config import load_experiment
from latent_working_memory.v4.train import write_json


DEFAULT_CONFIG = Path("configs/v4/attention_matching/experiment.json")
TRAINING_OPTIONS = (
    "config",
    "run_id",
    "run_name",
    "output_root",
    "model_path",
    "train_file",
    "dev_file",
    "global_batch_size",
    "micro_batch_size_per_gpu",
    "max_seq_length",
    "tracking",
    "swanlab_project",
    "group",
    "tag",
)


def physical_gpus(value):
    devices = [device.strip() for device in value.split(",")]
    if any(re.fullmatch(r"[0-9]+", device) is None for device in devices) or len(
        {int(device) for device in devices}
    ) != len(devices):
        raise argparse.ArgumentTypeError("select distinct non-negative physical GPU indices")
    return ",".join(str(int(device)) for device in devices)


def positive_count(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def token_limit(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative; 0 disables the training limit")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--resume", action="store_true", help="按原配置恢复训练")
    actions.add_argument("--generate-only", action="store_true", help="从 checkpoint 生成单条文本")
    parser.add_argument("--checkpoint", type=Path, help="global_step_<step> 目录")
    parser.add_argument("--config", type=Path, help=f"训练预设，默认 {DEFAULT_CONFIG}")
    parser.add_argument("--gpus", type=physical_gpus, default="0,1", help="物理 GPU 编号，默认 0,1")
    parser.add_argument("--run-id", help="实验系列目录，默认上海时间 YYYYMMDD-HHMMSS")
    parser.add_argument("--run-name", help="训练目录和展示名，默认模型名_query模式-m容量")
    parser.add_argument("--output-root", type=Path, help="产物父目录，默认 artifacts/v4")
    parser.add_argument("--model-path", type=Path, help="覆盖预设中的本地基座模型位置")
    parser.add_argument("--train-file", type=Path, help="训练 JSONL 文件")
    parser.add_argument("--dev-file", type=Path, help="开发 JSONL 文件")
    for name in ("global-batch-size", "micro-batch-size-per-gpu"):
        parser.add_argument(f"--{name}", type=positive_count)
    parser.add_argument(
        "--max-seq-length",
        type=token_limit,
        help="训练完整输入的人为上限；0 关闭；模型窗口始终强制过滤",
    )
    parser.add_argument(
        "--stop-after-steps", type=positive_count, help="停止于指定全局 optimizer step，不修改预算"
    )
    parser.add_argument("--tracking", choices=("online", "disabled"), help="默认沿用配置")
    parser.add_argument("--swanlab-project", help="已有 SwanLab project；不自动创建")
    parser.add_argument("--group", help="实验系列 group，未配置时使用 run-id")
    parser.add_argument("--tag", action="append", help="追加命名空间标签，可重复指定")
    parser.add_argument("--prompt-file", type=Path, help="生成使用的纯文本 prompt")
    parser.add_argument("--output", type=Path, help="生成结果 JSON 的保存路径")
    parser.add_argument("--max-new-tokens", type=positive_count, help="生成 token 上限，默认 128")
    args = parser.parse_args(argv)
    if args.resume or args.generate_only:
        if args.checkpoint is None:
            parser.error("--resume and --generate-only require --checkpoint")
        supplied = [name for name in TRAINING_OPTIONS if getattr(args, name) is not None]
        if supplied:
            parser.error(
                "checkpoint actions retain the original configuration: " + ", ".join(supplied)
            )
    elif args.checkpoint is not None:
        parser.error("--checkpoint requires --resume or --generate-only")
    if args.generate_only:
        if args.prompt_file is None:
            parser.error("--generate-only requires --prompt-file")
        if len(args.gpus.split(",")) != 1:
            parser.error("generation uses one GPU; explicitly select --gpus 0 or --gpus 1")
        if args.stop_after_steps is not None:
            parser.error("--stop-after-steps applies to training")
    elif any(
        getattr(args, name) is not None for name in ("prompt_file", "output", "max_new_tokens")
    ):
        parser.error("prompt/output/max-new-tokens require --generate-only")
    return args


def prepare_training(args):
    model, training = load_experiment(args.config or DEFAULT_CONFIG)
    if args.model_path is not None:
        model = replace(model, model_name_or_path=str(args.model_path.expanduser().resolve()))
    overrides = {
        name: str(getattr(args, name).expanduser().resolve())
        for name in ("train_file", "dev_file")
        if getattr(args, name) is not None
    }
    overrides.update(
        {
            name: getattr(args, name)
            for name in (
                "global_batch_size",
                "micro_batch_size_per_gpu",
                "max_seq_length",
            )
            if getattr(args, name) is not None
        }
    )
    if args.max_seq_length == 0:
        overrides["max_seq_length"] = None
    run_id = args.run_id or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    run_name = (
        args.run_name
        or training.run_name
        or (f"{Path(model.model_name_or_path).name.lower()}_{model.query_mode}-m{model.num_slots}")
    )
    for name, value in (("run-id", run_id), ("run-name", run_name)):
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
            raise ValueError(f"{name} must contain only letters, digits, '.', '-' and '_'")
    directory = (args.output_root or Path("artifacts/v4")).expanduser().resolve() / run_id
    output = directory / "train" / run_name
    plan = directory / "plan" / run_name
    if output.exists() or plan.exists():
        raise ValueError(f"run already exists: {output}; use a new run-id/run-name or --resume")
    project = args.swanlab_project if args.swanlab_project is not None else training.swanlab_project
    if args.tracking == "disabled":
        project = None
    elif args.tracking == "online" and project is None:
        raise ValueError("online tracking requires --swanlab-project or a project in the config")
    training = replace(
        training,
        **overrides,
        output_dir=str(output),
        run_name=run_name,
        swanlab_project=project,
        group=args.group or training.group or run_id,
        tags=tuple(sorted(set(training.tags) | set(args.tag or []))),
    )
    if (
        training.max_seq_length is not None
        and training.max_seq_length < model.pending_size + model.recent_size + 2
    ):
        raise ValueError("max-seq-length must allow a target after the first memory write")
    plan.mkdir(parents=True)
    config_path = plan / "config.json"
    write_json(config_path, {"model": asdict(model), "training": asdict(training)})
    return output, plan, config_path


def execute(command, environment, log_path):
    with log_path.open("a", encoding="utf-8") as log:
        invocation = "$ " + shlex.join(command) + "\n"
        print(invocation, end="", flush=True)
        log.write(invocation)
        log.flush()
        with subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
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
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else None
    if checkpoint is None:
        output, plan, config_path = prepare_training(args)
    else:
        run = read_checkpoint(checkpoint)["run"]
        output = Path(run["training"]["output_dir"])
        if output.parent.name != "train" or checkpoint.parent != output / "checkpoints":
            raise ValueError("checkpoint must belong to <series>/train/<run-name>/checkpoints")
        plan = output.parent.parent / "plan" / output.name
        plan.mkdir(parents=True, exist_ok=True)
        config_path = plan / "resume-config.json"
        if args.resume:
            if len(args.gpus.split(",")) != run["world_size"]:
                raise ValueError("resume must retain the original GPU count")
            write_json(config_path, {"model": run["model"], "training": run["training"]})

    environment = dict(
        os.environ,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=args.gpus,
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    command = [sys.executable]
    if args.generate_only:
        destination = (
            args.output.expanduser().resolve()
            if args.output is not None
            else output.parent.parent / "eval" / output.name / "greedy-sample.json"
        )
        command += [
            "-m",
            "latent_working_memory.v4.inference",
            "--checkpoint",
            str(checkpoint),
            "--prompt-file",
            str(args.prompt_file.expanduser().resolve()),
            "--output",
            str(destination),
            "--max-new-tokens",
            str(args.max_new_tokens or 128),
            "--device",
            "cuda",
        ]
        execute(command, environment, plan / "generate.log")
        return
    gpu_count = len(args.gpus.split(","))
    if gpu_count > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpu_count}"]
    command += [
        "-m",
        "latent_working_memory.v4.train",
        "--config",
        str(config_path),
        "--device",
        "cuda",
    ]
    if args.resume:
        command += ["--resume", str(checkpoint)]
    if args.stop_after_steps is not None:
        command += ["--stop-after-steps", str(args.stop_after_steps)]
    write_json(
        plan / "job.json",
        {"action": "resume" if args.resume else "train", "gpus": args.gpus, "command": command},
    )
    try:
        execute(command, environment, plan / "train.log")
    except subprocess.CalledProcessError as error:
        write_json(plan / "result.json", {"status": "failed", "return_code": error.returncode})
        raise
    result = json.loads((output / "training-result.json").read_text(encoding="utf-8"))
    write_json(
        plan / "result.json", {"status": "finished" if result["complete"] else "stopped", **result}
    )


def main(argv=None):
    args = parse_args(argv)
    try:
        run_job(args)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error


if __name__ == "__main__":
    main()
