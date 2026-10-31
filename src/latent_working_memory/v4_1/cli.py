"""公共运行参数与显式配置覆盖。"""

import argparse
from pathlib import Path


DEFAULT_CONFIG = "configs/v4-1/online_ttt_memory.json"


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


def save_frequency(value):
    return value if value == "after_each_epoch" else positive_int(value)


def add_config_arguments(parser):
    parser.add_argument("--config", default=argparse.SUPPRESS,
                        help=f"JSON 配置文件，默认 {DEFAULT_CONFIG}")
    parser.add_argument("--output-root", default=argparse.SUPPRESS, help="实验产物父目录")
    parser.add_argument("--model-path", default=argparse.SUPPRESS, help="覆盖基座模型路径")
    parser.add_argument("--seed", type=nonnegative_int, default=argparse.SUPPRESS)
    for name in ("max-train-samples", "max-train-data-tokens", "max-dev-samples"):
        parser.add_argument(
            f"--{name}", type=optional_positive_int, default=argparse.SUPPRESS,
            help="覆盖配置中的上限；null 取消限制",
        )


def parse_run_args(argv=None):
    parser = argparse.ArgumentParser(
        description="v4.1 完整运行、训练恢复与独立评估。", argument_default=argparse.SUPPRESS,
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train-only", action="store_true", help="只训练及 dev 验证，跳过最终 test")
    modes.add_argument("--resume", action="store_true", help="沿用 checkpoint 配置恢复训练及评估")
    modes.add_argument("--evaluate-only", action="store_true", help="只评估已有 checkpoint")
    add_config_arguments(parser)
    parser.add_argument("--train-data", help="训练与 dev 数据目录，默认使用 training.dataset_dir")
    parser.add_argument("--run-id", help="运行目录的末尾标识，默认 YYYYMMDD-HHMMSS")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--checkpoint", nargs="+", help="恢复用单个 checkpoint；独立评估可传多个")
    sources.add_argument("--eval-run", help="评估该训练目录仍保留的全部 checkpoint")
    parser.add_argument("--eval-type", choices=("test", "benchmark"), help="独立评估类型，默认 test")
    parser.add_argument("--benchmark", choices=("locomo", "ruler"), help="Benchmark 名称")
    parser.add_argument("--benchmark-data", help="Benchmark 数据目录，首次创建计划必填")
    parser.add_argument("--lengths", nargs="+", type=positive_int, help="仅 RULER；默认 8192 16384 32768")
    parser.add_argument("--test-data", help="Test 数据目录；独立评估首次创建计划时必填")
    parser.add_argument("--max-test-samples", type=optional_positive_int, help="Test 样本上限，null 使用全量")
    parser.add_argument("--max-new-tokens", type=positive_int, help="Test 生成上限，默认 128")
    parser.add_argument("--eval-id", help="评估计划标识；独立评估默认 YYYYMMDD-HHMMSS，自动 test 默认 auto")
    parser.add_argument("--micro-batch-size-per-gpu", type=positive_int,
                        help="每卡训练 microbatch，默认 2")
    parser.add_argument("--gradient-accumulation-steps", type=positive_int,
                        help="每卡 optimizer step 的 microbatch 累积次数，默认 8")
    parser.add_argument("--teacher-batch-size-per-gpu", type=positive_int,
                        help="每卡 teacher forcing 批量，默认 64")
    parser.add_argument("--epochs", type=positive_int, help="完整训练轮数，默认使用配置值")
    parser.add_argument("--dev-every", type=positive_int)
    parser.add_argument("--save-every", type=save_frequency,
                        help="正整数 step 间隔或 after_each_epoch；轮末和最终步始终保存")
    parser.add_argument("--save-total-limit", type=positive_int)
    parser.add_argument("--tracking", choices=("online", "disabled"))
    parser.add_argument("--swanlab-project")
    parser.add_argument("--group", help="实验分组，默认使用运行目录名")
    parser.add_argument("--tag", action="append", help="追加命名空间标签，可重复指定")
    args = parser.parse_args(argv)
    supplied = set(vars(args))
    args.resume = getattr(args, "resume", False)
    args.evaluate_only = getattr(args, "evaluate_only", False)
    args.checkpoint = getattr(args, "checkpoint", None)
    args.eval_run = getattr(args, "eval_run", None)

    if args.resume:
        if args.checkpoint is None:
            parser.error("--resume requires --checkpoint")
        if len(args.checkpoint) != 1:
            parser.error("--resume accepts exactly one checkpoint")
        overrides = supplied - {"resume", "checkpoint"}
        if overrides:
            options = ", ".join("--" + name.replace("_", "-") for name in sorted(overrides))
            parser.error(f"--resume does not accept configuration overrides: {options}")
        args.checkpoint = args.checkpoint[0]
    elif args.evaluate_only:
        if args.checkpoint is None and args.eval_run is None:
            parser.error("--evaluate-only requires --checkpoint or --eval-run")
        allowed = {
            "evaluate_only", "checkpoint", "eval_run", "eval_type", "benchmark",
            "benchmark_data", "test_data", "max_test_samples", "max_new_tokens",
            "lengths", "eval_id", "output_root",
        }
        overrides = supplied - allowed
        if overrides:
            options = ", ".join("--" + name.replace("_", "-") for name in sorted(overrides))
            parser.error(f"--evaluate-only retains the checkpoint configuration: {options}")
        args.eval_type = getattr(args, "eval_type", "test")
        args.benchmark = getattr(args, "benchmark", None)
        args.eval_id = getattr(args, "eval_id", None)
        args.output_root = getattr(args, "output_root", None)
        if args.eval_type == "test":
            if args.benchmark is not None or any(name in supplied for name in ("benchmark_data", "lengths")):
                parser.error("--benchmark, --benchmark-data and --lengths require --eval-type benchmark")
        else:
            if args.benchmark is None:
                parser.error("--eval-type benchmark requires --benchmark")
            if any(name in supplied for name in ("test_data", "max_test_samples", "max_new_tokens")):
                parser.error("--test-data, --max-test-samples and --max-new-tokens require --eval-type test")
            if "lengths" in supplied and args.benchmark != "ruler":
                parser.error("--lengths requires --benchmark ruler")
    else:
        if args.checkpoint is not None:
            parser.error("--checkpoint requires --resume or --evaluate-only")
        if args.eval_run is not None:
            parser.error("--eval-run requires --evaluate-only")
        if supplied & {"eval_type", "benchmark", "benchmark_data", "lengths"}:
            parser.error("--eval-type, --benchmark, --benchmark-data and --lengths require --evaluate-only")
        if getattr(args, "train_only", False) and supplied & {
            "test_data", "max_test_samples", "max_new_tokens", "eval_id",
        }:
            parser.error("--train-only cannot be combined with test evaluation options")
    return args


def config_overrides(args):
    """只收集实际提供的选项；nullable 字段的显式 null 不被过滤。"""
    values = vars(args)
    overrides = {}
    if "seed" in values:
        overrides["seed"] = values["seed"]
    if "model_path" in values:
        overrides["model"] = {
            "model_name_or_path": str(Path(values["model_path"]).expanduser().resolve()),
        }
    teacher = {}
    for option, field in (
        ("prepare_batch_size_per_gpu", "prepare_batch_size"),
        ("teacher_batch_size_per_gpu", "batch_size"),
    ):
        if option in values:
            teacher[field] = values[option]
    if teacher:
        overrides["teacher"] = teacher
    training = {}
    if "output_root" in values:
        training.update(output_root=values["output_root"], teacher_cache_dir=None, output_dir=None)
    for name in ("prepare_data", "train_data"):
        if name in values:
            training["dataset_dir"] = str(Path(values[name]).expanduser().resolve())
    if "test_data" in values:
        training["test_data"] = str(Path(values["test_data"]).expanduser().resolve())
    for option, field in (
        ("run_id", "run_id"),
        ("train_only", "train_only"), ("eval_id", "eval_id"),
        ("max_test_samples", "max_test_samples"), ("max_new_tokens", "max_new_tokens"),
        ("micro_batch_size_per_gpu", "micro_batch_size"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
        ("epochs", "epochs"),
        ("dev_every", "dev_every"), ("save_every", "save_every"),
        ("save_total_limit", "save_total_limit"),
        ("max_train_samples", "max_train_samples"),
        ("max_train_data_tokens", "max_train_data_tokens"),
        ("max_dev_samples", "max_dev_samples"),
        ("swanlab_project", "swanlab_project"), ("group", "group"), ("tag", "tags"),
    ):
        if option in values:
            training[field] = values[option]
    if "tracking" in values:
        training["tracking"] = {"online": "swanlab", "disabled": "disabled"}[values["tracking"]]
    if training:
        overrides["training"] = training
    return overrides
