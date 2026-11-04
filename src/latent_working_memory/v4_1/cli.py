"""公共运行参数与显式配置覆盖。"""

import argparse
from pathlib import Path
import sys


DEFAULT_CONFIG = "configs/v4-1/default.json"
DEFAULT_EVAL_BATCH_SIZE = 8


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


def ignore_resume_options(args, retained):
    """删除恢复模式不使用的显式选项，并集中提示。"""
    ignored = sorted(set(vars(args)) - retained)
    for name in ignored:
        delattr(args, name)
    if ignored:
        options = ", ".join("--" + name.replace("_", "-") for name in ignored)
        print(f"[resume] ignored options; using the saved plan: {options}", file=sys.stderr)


def parse_run_args(argv=None):
    parser = argparse.ArgumentParser(
        description="v4.1 训练、训练恢复与独立评估。", argument_default=argparse.SUPPRESS,
    )
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true", help="恢复已有训练或评估计划")
    restart.add_argument("--overwrite", action="store_true", help="清理目标 run 或当前评估 task 后重新执行")
    parser.add_argument("--evaluate-only", action="store_true", help="只评估已有 checkpoint")
    add_config_arguments(parser)
    parser.add_argument("--train-data", help="训练与 dev 数据目录，默认使用 training.dataset_dir")
    parser.add_argument("--run-id", help="运行目录名，默认方法设置加上海时间 YYYYMMDD-HHMM")
    parser.add_argument("--checkpoint", nargs="+",
                        help="训练恢复用单个路径；评估可选 global_step_<N> 名称列表，续跑可补入计划")
    parser.add_argument("--eval-run", help="独立评估必填，指定包含 train/、plan/ 与 eval/ 的运行根目录")
    parser.add_argument("--eval-task", choices=("test", "ruler", "locomo"), help="评估任务，默认 test")
    parser.add_argument("--eval-data", help="评估数据目录，新建或覆盖计划时必填")
    parser.add_argument("--lengths", nargs="+", type=positive_int, help="仅 RULER；默认 8192 16384 32768")
    parser.add_argument("--max-test-samples", type=optional_positive_int, help="Test 样本上限，null 使用全量")
    parser.add_argument("--max-new-tokens", type=positive_int, help="Test 生成上限，默认 128")
    parser.add_argument("--eval-id", help="评估计划标识，默认 YYYYMMDD-HHMMSS")
    parser.add_argument("--eval-batch-size-per-gpu", type=positive_int,
                        help="每卡评估 batch，默认 8")
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
    if getattr(args, "resume", False):
        retained = {"resume", "overwrite", "evaluate_only"}
        if getattr(args, "evaluate_only", False):
            retained.update({
                "checkpoint", "eval_run", "eval_id", "eval_task", "output_root", "eval_batch_size_per_gpu",
            })
        else:
            retained.add("checkpoint")
        ignore_resume_options(args, retained)
    supplied = set(vars(args))
    args.resume = getattr(args, "resume", False)
    args.evaluate_only = getattr(args, "evaluate_only", False)
    args.checkpoint = getattr(args, "checkpoint", None)
    args.eval_run = getattr(args, "eval_run", None)
    args.overwrite = getattr(args, "overwrite", False)

    if args.evaluate_only:
        if args.eval_run is None:
            parser.error("--evaluate-only requires --eval-run")
        allowed = {
            "evaluate_only", "resume", "checkpoint", "eval_run", "eval_task",
            "eval_data", "max_test_samples", "max_new_tokens",
            "lengths", "eval_id", "output_root", "overwrite", "eval_batch_size_per_gpu",
        }
        overrides = supplied - allowed
        if overrides:
            options = ", ".join("--" + name.replace("_", "-") for name in sorted(overrides))
            parser.error(f"--evaluate-only retains the checkpoint configuration: {options}")
        args.eval_task = getattr(args, "eval_task", "test")
        args.eval_id = getattr(args, "eval_id", None)
        args.output_root = getattr(args, "output_root", None)
        args.eval_batch_size_per_gpu = getattr(args, "eval_batch_size_per_gpu", DEFAULT_EVAL_BATCH_SIZE)
        if args.resume and not args.eval_id:
            parser.error("evaluation --resume requires an explicit --eval-id")
        if args.resume and "eval_task" not in supplied:
            parser.error("evaluation --resume requires an explicit --eval-task")
        if args.eval_task != "test" and supplied & {"max_test_samples", "max_new_tokens"}:
            parser.error("--max-test-samples and --max-new-tokens require --eval-task test")
        if "lengths" in supplied and args.eval_task != "ruler":
            parser.error("--lengths requires --eval-task ruler")
    elif args.resume:
        if args.checkpoint is None:
            parser.error("--resume requires --checkpoint")
        if len(args.checkpoint) != 1:
            parser.error("--resume accepts exactly one checkpoint")
        args.checkpoint = args.checkpoint[0]
    else:
        if args.checkpoint is not None:
            parser.error("--checkpoint requires --resume or --evaluate-only")
        if args.eval_run is not None:
            parser.error("--eval-run requires --evaluate-only")
        if supplied & {"eval_task", "eval_data", "lengths",
                       "max_test_samples", "max_new_tokens", "eval_id", "eval_batch_size_per_gpu"}:
            parser.error("evaluation options require --evaluate-only")
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
    for option, field in (
        ("run_id", "run_id"),
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
