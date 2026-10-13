"""v4 GPU 入口：训练、恢复、单条生成和独立评估。"""

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

from latent_working_memory.v4.config import load_experiment, validate_experiment
from latent_working_memory.v4.io import write_json


DEFAULT_CONFIG = Path("configs/v4/attention_matching/experiment.json")
TRAINING_OPTIONS = (
    "config",
    "run_id",
    "output_root",
    "model_path",
    "data_dir",
    "micro_batch_size_per_gpu",
    "gradient_accumulation_steps",
    "epochs",
    "seed",
    "eval_every",
    "save_every",
    "save_total_limit",
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


def nonnegative_count(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def optional_positive_count(value):
    return None if value == "null" else positive_count(value)


def save_frequency(value):
    return value if value == "after_each_epoch" else positive_count(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--resume", action="store_true", help="按原配置恢复训练")
    actions.add_argument("--generate-only", action="store_true", help="从 checkpoint 生成单条文本")
    actions.add_argument("--evaluate-only", action="store_true", help="从 checkpoint 评估 test")
    parser.add_argument("--checkpoint", type=Path, help="global_step_<step> 目录")
    parser.add_argument("--config", type=Path, help=f"训练预设，默认 {DEFAULT_CONFIG}")
    parser.add_argument("--gpus", type=physical_gpus, default="0,1", help="物理 GPU 编号，默认 0,1")
    parser.add_argument("--run-id", help="实验系列目录，默认上海时间 YYYYMMDD-HHMMSS")
    parser.add_argument("--output-root", type=Path, help="产物父目录，默认 artifacts/v4")
    parser.add_argument("--model-path", type=Path, help="覆盖预设中的本地基座模型位置")
    parser.add_argument("--data-dir", type=Path, help="含 train/dev/test.jsonl 的数据目录")
    for name in (
        "micro-batch-size-per-gpu",
        "gradient-accumulation-steps",
        "epochs",
        "eval-every",
        "save-total-limit",
    ):
        parser.add_argument(f"--{name}", type=positive_count)
    parser.add_argument("--seed", type=nonnegative_count)
    parser.add_argument(
        "--save-every",
        type=save_frequency,
        help="保存间隔：正整数 optimizer steps 或 after_each_epoch",
    )
    parser.add_argument(
        "--max-seq-length",
        type=optional_positive_count,
        default=argparse.SUPPRESS,
        help="训练完整输入的人为上限；null 关闭；模型窗口始终强制过滤",
    )
    parser.add_argument(
        "--stop-after-steps", type=positive_count, help="停止于指定全局 optimizer step，不修改预算"
    )
    parser.add_argument(
        "--tracking", choices=("online", "disabled"), help="沿用配置；默认预设开启在线记录"
    )
    parser.add_argument("--swanlab-project", help="覆盖配置中的 SwanLab project")
    parser.add_argument("--group", help="实验系列 group，未配置时使用 run-id")
    parser.add_argument("--tag", action="append", help="追加命名空间标签，可重复指定")
    parser.add_argument("--prompt-file", type=Path, help="生成使用的纯文本 prompt")
    parser.add_argument("--output", type=Path, help="生成结果 JSON 的保存路径")
    parser.add_argument(
        "--output-dir", type=Path, help="评估结果目录，默认 <series>/eval/<method-dir>/test"
    )
    parser.add_argument("--max-new-tokens", type=positive_count, help="生成 token 上限，默认 128")
    args = parser.parse_args(argv)
    if args.resume or args.generate_only or args.evaluate_only:
        if args.checkpoint is None:
            parser.error("--resume, --generate-only and --evaluate-only require --checkpoint")
        supplied = [
            name
            for name in TRAINING_OPTIONS
            if (
                getattr(args, name, None) is not None
                or (name == "max_seq_length" and hasattr(args, name))
            )
            and not (args.evaluate_only and name in {"tracking", "data_dir"})
        ]
        if supplied:
            parser.error(
                "checkpoint actions retain the original configuration: " + ", ".join(supplied)
            )
    elif args.checkpoint is not None:
        parser.error("--checkpoint requires --resume, --generate-only or --evaluate-only")
    if args.generate_only:
        if args.prompt_file is None:
            parser.error("--generate-only requires --prompt-file")
    elif args.prompt_file is not None or args.output is not None:
        parser.error("--prompt-file and --output require --generate-only")
    if not args.evaluate_only and args.output_dir is not None:
        parser.error("--output-dir requires --evaluate-only")
    if args.generate_only or args.evaluate_only:
        if len(args.gpus.split(",")) != 1:
            parser.error("generation/evaluation uses one GPU; select --gpus 0 or --gpus 1")
        if args.stop_after_steps is not None:
            parser.error("--stop-after-steps applies to training")
    elif args.max_new_tokens is not None:
        parser.error("--max-new-tokens requires --generate-only or --evaluate-only")
    return args


def prepare_training(args):
    model, training = load_experiment(args.config or DEFAULT_CONFIG)
    if args.model_path is not None:
        model = replace(model, model_name_or_path=str(args.model_path.expanduser().resolve()))
    overrides = {
        name: getattr(args, name)
        for name in (
            "micro_batch_size_per_gpu",
            "gradient_accumulation_steps",
            "epochs",
            "seed",
            "eval_every",
            "save_every",
            "save_total_limit",
        )
        if getattr(args, name) is not None
    }
    if args.data_dir is not None:
        overrides["dataset_dir"] = str(args.data_dir.expanduser().resolve())
    if hasattr(args, "max_seq_length"):
        overrides["max_seq_length"] = args.max_seq_length
    run_id = args.run_id or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    method_dir = f"{model.query_mode}-m{model.num_slots}"
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id) is None:
        raise ValueError("run-id must contain only letters, digits, '-' and '_'")
    directory = (args.output_root or Path("artifacts/v4")).expanduser().resolve() / run_id
    output = directory / "train" / method_dir
    plan = directory / "plan" / method_dir
    if output.exists() or plan.exists():
        raise ValueError(f"run already exists: {output}; use a new run-id or --resume")
    project = args.swanlab_project if args.swanlab_project is not None else training.swanlab_project
    if args.tracking == "disabled":
        project = None
    elif args.tracking == "online" and project is None:
        raise ValueError("online tracking requires --swanlab-project or a project in the config")
    training = replace(
        training,
        **overrides,
        output_dir=str(output),
        swanlab_project=project,
        group=args.group or training.group or run_id,
        tags=tuple(sorted(set(training.tags) | set(args.tag or []))),
    )
    validate_experiment(model, training)
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
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def run_job(args):
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else None
    if checkpoint is None:
        output, plan, config_path = prepare_training(args)
    else:
        output = checkpoint.parent.parent
        if output.parent.name != "train" or checkpoint.parent.name != "checkpoints":
            raise ValueError("checkpoint must belong to <series>/train/<method-dir>/checkpoints")
        if re.fullmatch(r"global_step_[0-9]+", checkpoint.name) is None or not checkpoint.is_dir():
            raise ValueError("checkpoint must be an existing global_step_<step> directory")
        run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if Path(run["training"]["output_dir"]).resolve() != output:
            raise ValueError("checkpoint directory differs from the saved run output_dir")
        plan = output.parent.parent / "plan" / output.name
        plan.mkdir(parents=True, exist_ok=True)
        config_path = plan / "config.json"
        if args.resume:
            if len(args.gpus.split(",")) != run["world_size"]:
                raise ValueError("resume must retain the original GPU count")

    environment = dict(
        os.environ,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=args.gpus,
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    command = [sys.executable]
    action = "resume" if args.resume else "train"
    result_path, log_path = plan / "result.json", plan / "train.log"
    destination = None
    if args.generate_only:
        action = "generate"
        result_path = plan / "generate-result.json"
        log_path = plan / "generate.log"
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
    elif args.evaluate_only:
        action = "evaluate"
        result_path = plan / "evaluation-result.json"
        log_path = plan / "evaluation.log"
        destination = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else output.parent.parent / "eval" / output.name / "test"
        )
        command += [
            "-m",
            "latent_working_memory.v4.evaluate",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(destination),
            "--max-new-tokens",
            str(args.max_new_tokens or 128),
            "--device",
            "cuda",
        ]
        if args.data_dir is not None:
            command += ["--data-dir", str(args.data_dir.expanduser().resolve())]
        if args.tracking is not None:
            command += ["--tracking", args.tracking]
    else:
        gpu_count = len(args.gpus.split(","))
        if gpu_count > 1:
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={gpu_count}",
            ]
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
    job = {
        "action": action,
        "gpus": args.gpus,
        "command": command,
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
    }
    write_json(result_path, {**job, "status": "running"})
    try:
        execute(command, environment, log_path)
        if args.generate_only or args.evaluate_only:
            result = {
                "status": "finished",
                "output" if args.generate_only else "output_dir": str(destination),
            }
        else:
            result = json.loads((output / "training-result.json").read_text(encoding="utf-8"))
            result = {**result, "status": "finished" if result["complete"] else "stopped"}
    except (Exception, KeyboardInterrupt) as error:
        write_json(
            result_path,
            {
                **job,
                "status": "stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                "return_code": (
                    error.returncode
                    if isinstance(error, subprocess.CalledProcessError)
                    else 130
                    if isinstance(error, KeyboardInterrupt)
                    else None
                ),
                "error": f"{type(error).__name__}: {error}",
                "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
                    timespec="seconds"
                ),
            },
        )
        raise
    write_json(
        result_path,
        {
            **job,
            **result,
            "return_code": 0,
            "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        },
    )


def main(argv=None):
    args = parse_args(argv)
    try:
        run_job(args)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    except KeyboardInterrupt as error:
        raise SystemExit(130) from error


if __name__ == "__main__":
    main()
