"""四种运行模式的参数边界；只覆盖显式提供的配置。"""

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def optional_positive_int(value):
    return None if value == "null" else positive_int(value)


def positive_float(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def save_frequency(value):
    return value if value == "after_each_epoch" else positive_int(value)


def parse_run_args(argv=None):
    parser = argparse.ArgumentParser(
        description="完整运行、只训练、恢复与独立评估。", argument_default=argparse.SUPPRESS
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train-only", action="store_true", help="只训练及 dev 验证")
    modes.add_argument("--resume", action="store_true", help="从单个 checkpoint 沿用计划恢复")
    modes.add_argument("--evaluate-only", action="store_true", help="独立评估已有 checkpoint")
    parser.add_argument("--config", help=f"JSON 配置，默认 {DEFAULT_CONFIG}")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="默认 cpu；CUDA 由可见设备选择")
    parser.add_argument("--cpu-processes", type=positive_int, help="CPU worker 数，默认 1")
    parser.add_argument("--output-root", help="产物父目录")
    parser.add_argument("--run-id", help="运行标识，默认 YYYYMMDD-HHMMSS")
    parser.add_argument("--seed", type=nonnegative_int)
    parser.add_argument("--train-data", help="包含 train.jsonl 和 dev.jsonl 的目录")
    parser.add_argument(
        "--epochs", type=optional_positive_int, help="训练轮数，null 只按 max-steps 限制"
    )
    parser.add_argument("--max-steps", type=optional_positive_int)
    parser.add_argument("--micro-batch-size-per-gpu", type=positive_int)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int)
    parser.add_argument("--learning-rate", type=positive_float)
    parser.add_argument("--dev-every", type=positive_int)
    parser.add_argument("--save-every", type=save_frequency)
    parser.add_argument("--save-total-limit", type=positive_int)
    for name in ("max-train-data-tokens", "max-train-samples", "max-dev-samples"):
        parser.add_argument(
            f"--{name}", type=optional_positive_int, help="正整数上限；null 取消限制"
        )
    parser.add_argument("--tracking", choices=("disabled", "online"))
    parser.add_argument("--swanlab-project", help="online 记录时必须显式指定")
    parser.add_argument("--group", help="SwanLab 实验组")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--checkpoint", nargs="+", help="恢复用单个路径；独立评估可传多个")
    sources.add_argument("--eval-run", help="评估该运行目录仍保留的全部 checkpoint")
    parser.add_argument("--eval-id", help="独立评估默认 YYYYMMDD-HHMMSS；自动 test 默认 auto")
    parser.add_argument("--test-data", help="包含 test.jsonl 的目录")
    parser.add_argument("--max-test-samples", type=optional_positive_int)
    parser.add_argument(
        "--max-new-tokens", type=positive_int, help="保留的任务生成预算；本示例回归任务不使用"
    )
    args = parser.parse_args(argv)
    supplied = set(vars(args))
    resume = getattr(args, "resume", False)
    evaluate_only = getattr(args, "evaluate_only", False)
    if resume:
        if "checkpoint" not in supplied or len(args.checkpoint) != 1:
            parser.error("--resume requires exactly one --checkpoint")
        forbidden = supplied - {"resume", "checkpoint", "device", "cpu_processes"}
        args.checkpoint = args.checkpoint[0]
    elif evaluate_only:
        if not supplied & {"checkpoint", "eval_run"}:
            parser.error("--evaluate-only requires --checkpoint or --eval-run")
        forbidden = supplied - {
            "evaluate_only",
            "checkpoint",
            "eval_run",
            "eval_id",
            "test_data",
            "max_test_samples",
            "max_new_tokens",
            "device",
            "cpu_processes",
        }
    else:
        forbidden = supplied & {"checkpoint", "eval_run"}
        if getattr(args, "train_only", False):
            forbidden |= supplied & {"test_data", "max_test_samples", "max_new_tokens", "eval_id"}
    if forbidden:
        options = ", ".join("--" + name.replace("_", "-") for name in sorted(forbidden))
        parser.error(f"options are not valid in this mode: {options}")
    if getattr(args, "device", "cpu") == "cuda" and "cpu_processes" in supplied:
        parser.error("--cpu-processes requires --device cpu")
    args.train_only = getattr(args, "train_only", False)
    args.resume = resume
    args.evaluate_only = evaluate_only
    args.checkpoint = getattr(args, "checkpoint", None)
    args.eval_run = getattr(args, "eval_run", None)
    args.device = getattr(args, "device", "cpu")
    args.cpu_processes = getattr(args, "cpu_processes", 1)
    args._supplied = supplied
    return args


def config_overrides(args):
    """nullable 参数的显式 null 也覆盖配置；进程参数不写入方法配置。"""
    supplied = args._supplied
    overrides = {"seed": args.seed} if "seed" in supplied else {}
    training = {}
    for name in (
        "output_root",
        "run_id",
        "train_only",
        "eval_id",
        "epochs",
        "max_steps",
        "gradient_accumulation_steps",
        "learning_rate",
        "dev_every",
        "save_every",
        "save_total_limit",
        "max_train_data_tokens",
        "max_train_samples",
        "max_dev_samples",
        "max_test_samples",
        "max_new_tokens",
        "tracking",
        "swanlab_project",
        "group",
    ):
        if name in supplied:
            training[name] = getattr(args, name)
    if "micro_batch_size_per_gpu" in supplied:
        training["micro_batch_size"] = args.micro_batch_size_per_gpu
    for option, field in (("train_data", "dataset_dir"), ("test_data", "test_data")):
        if option in supplied:
            training[field] = str(Path(getattr(args, option)).expanduser().resolve())
    if training:
        overrides["training"] = training
    return overrides
