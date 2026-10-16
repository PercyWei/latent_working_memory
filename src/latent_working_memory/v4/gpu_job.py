"""v4 GPU 入口：训练、恢复和独立评估。"""

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

from latent_working_memory.v4.config import load_experiment
from latent_working_memory.v4.io import write_json


DEFAULT_CONFIG = Path("configs/v4/attention_matching/experiment.json")
TRAINING_OPTIONS = (
    "config",
    "run_id",
    "output_root",
    "model_path",
    "train_data",
    "micro_batch_size_per_gpu",
    "gradient_accumulation_steps",
    "epochs",
    "seed",
    "dev_every",
    "save_every",
    "save_total_limit",
    "max_train_data_tokens",
    "max_train_samples",
    "max_dev_samples",
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
    actions.add_argument(
        "--train-only", action="store_true", help="只启动新训练，保留 dev 验证，跳过最终 test 评估"
    )
    actions.add_argument("--resume", action="store_true", help="按原配置恢复训练")
    actions.add_argument(
        "--evaluate-only", action="store_true", help="从 checkpoint 评估 QASPER test 或 benchmark"
    )
    parser.add_argument(
        "--eval-type", choices=("test", "benchmark"),
        help="独立评估的数据类型，默认 test",
    )
    parser.add_argument(
        "--benchmark", choices=("ruler", "locomo"), default=None,
        help="benchmark 名称，默认 null；--eval-type benchmark 时必填",
    )
    parser.add_argument("--lengths", type=positive_count, nargs="+", help="RULER 输入与生成预算的总长度列表")
    checkpoint_sources = parser.add_mutually_exclusive_group()
    checkpoint_sources.add_argument(
        "--checkpoint", type=Path, nargs="+",
        help="global_step_<step> 目录；独立评估可传多个，恢复训练只接受一个",
    )
    checkpoint_sources.add_argument(
        "--eval-run", type=Path,
        help="评估 <series>/train/<method-dir> 中全部保留的 checkpoint；仅用于 --evaluate-only",
    )
    parser.add_argument("--config", type=Path, help=f"训练预设，默认 {DEFAULT_CONFIG}")
    parser.add_argument("--gpus", type=physical_gpus, default="0,1", help="物理 GPU 编号，默认 0,1")
    parser.add_argument("--run-id", help="实验系列目录，默认上海时间 YYYYMMDD-HHMMSS")
    parser.add_argument("--output-root", type=Path, help="产物父目录，默认 artifacts/v4")
    parser.add_argument("--model-path", type=Path, help="覆盖预设中的本地基座模型位置")
    parser.add_argument("--train-data", type=Path, help="训练数据目录，包含 train.jsonl 与 dev.jsonl")
    parser.add_argument("--test-data", type=Path, help="test 评估数据目录，默认沿用训练记录")
    parser.add_argument("--benchmark-data", type=Path, help="benchmark 数据目录，默认沿用对应 benchmark 的位置")
    for name in (
        "micro-batch-size-per-gpu",
        "gradient-accumulation-steps",
        "epochs",
        "dev-every",
        "save-total-limit",
    ):
        parser.add_argument(f"--{name}", type=positive_count)
    parser.add_argument(
        "--seed", type=nonnegative_count, help="控制参数初始化、过滤后样本抽取及训练数据打乱"
    )
    parser.add_argument(
        "--save-every",
        type=save_frequency,
        help="保存间隔：正整数 optimizer steps 或 after_each_epoch",
    )
    for name, help_text in (
        (
            "max-train-data-tokens",
            "训练正文 text 的 token 上限；null 关闭；完整序列始终受模型窗口限制",
        ),
        ("max-train-samples", "长度过滤后按 seed 固定抽取的训练 QA 样本上限；null 使用全量"),
        ("max-dev-samples", "长度过滤后按 seed 固定抽取的 dev QA 样本上限；null 使用全量"),
    ):
        parser.add_argument(
            f"--{name}",
            type=optional_positive_count,
            default=argparse.SUPPRESS,
            help=help_text,
        )
    parser.add_argument(
        "--tracking", choices=("online", "disabled"), help="沿用配置；默认预设开启在线记录"
    )
    parser.add_argument("--swanlab-project", help="覆盖配置中的 SwanLab project")
    parser.add_argument("--group", help="实验系列 group，未配置时使用 run-id")
    parser.add_argument("--tag", action="append", help="追加命名空间标签，可重复指定")
    parser.add_argument(
        "--max-new-tokens", type=positive_count,
        help="QASPER test 的生成 token 上限，默认 128",
    )
    args = parser.parse_args(argv)
    if args.eval_run is not None and not args.evaluate_only:
        parser.error("--eval-run requires --evaluate-only")
    if args.resume or args.evaluate_only:
        if args.resume and args.checkpoint is None:
            parser.error("--resume requires --checkpoint")
        if args.evaluate_only and args.checkpoint is None and args.eval_run is None:
            parser.error("--evaluate-only requires --checkpoint or --eval-run")
        supplied = [
            name
            for name in TRAINING_OPTIONS
            if (
                getattr(args, name, None) is not None
                or (
                    name in {"max_train_data_tokens", "max_train_samples", "max_dev_samples"}
                    and hasattr(args, name)
                )
            )
            and not (args.evaluate_only and name == "tracking")
        ]
        if supplied:
            parser.error(
                "checkpoint actions retain the original configuration: " + ", ".join(supplied)
            )
    elif args.checkpoint is not None:
        parser.error("--checkpoint requires --resume or --evaluate-only")
    if args.evaluate_only:
        args.eval_type = args.eval_type or "test"
        if args.eval_type == "test":
            if any(value is not None for value in (args.benchmark, args.benchmark_data, args.lengths)):
                parser.error("--benchmark, --benchmark-data and --lengths require --eval-type benchmark")
        else:
            if args.benchmark is None:
                parser.error("--eval-type benchmark requires --benchmark")
            if args.test_data is not None or args.max_new_tokens is not None:
                parser.error("--test-data and --max-new-tokens require --eval-type test")
            if args.lengths is not None and args.benchmark != "ruler":
                parser.error("--lengths requires --benchmark ruler")
        if len(args.gpus.split(",")) != 1:
            parser.error("evaluation uses one GPU; select --gpus 0 or --gpus 1")
    elif any(
        value is not None
        for value in (
            args.eval_type, args.benchmark, args.test_data, args.benchmark_data,
            args.lengths, args.max_new_tokens,
        )
    ):
        parser.error(
            "--eval-type, --benchmark, --test-data, --benchmark-data, --lengths and "
            "--max-new-tokens require --evaluate-only"
        )
    if args.checkpoint is not None and not args.evaluate_only:
        if len(args.checkpoint) != 1:
            parser.error("multiple checkpoints require --evaluate-only")
        args.checkpoint = args.checkpoint[0]
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
            "dev_every",
            "save_every",
            "save_total_limit",
        )
        if getattr(args, name) is not None
    }
    if args.train_data is not None:
        overrides["dataset_dir"] = str(args.train_data.expanduser().resolve())
    for name in ("max_train_data_tokens", "max_train_samples", "max_dev_samples"):
        if hasattr(args, name):
            overrides[name] = getattr(args, name)
    run_id = args.run_id or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    method_dir = f"{model.query_mode}-query-m{model.num_slots}"
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


def run_evaluation_series(args):
    if args.eval_run is not None:
        output = args.eval_run.expanduser().resolve()
        if not output.is_dir() or output.parent.name != "train":
            raise ValueError("--eval-run requires an existing <series>/train/<method-dir>")
        checkpoints = [
            path.parent
            for path in output.glob("checkpoints/global_step_*/state.pt")
            if path.is_file() and re.fullmatch(r"global_step_[0-9]+", path.parent.name)
        ]
        if not checkpoints:
            raise ValueError(f"no retained checkpoints found in {output}")
    else:
        checkpoints = args.checkpoint
    checkpoints = [path.expanduser().resolve() for path in checkpoints]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("checkpoint list contains duplicate directories")
    for checkpoint in checkpoints:
        if (
            checkpoint.parent.name != "checkpoints"
            or checkpoint.parent.parent.parent.name != "train"
        ):
            raise ValueError("checkpoint must belong to <series>/train/<method-dir>/checkpoints")
        if re.fullmatch(r"global_step_[0-9]+", checkpoint.name) is None or not checkpoint.is_dir():
            raise ValueError("checkpoint must be an existing global_step_<step> directory")
        if not (checkpoint / "state.pt").is_file():
            raise ValueError(f"checkpoint is incomplete: missing {checkpoint / 'state.pt'}")
    outputs = {checkpoint.parent.parent for checkpoint in checkpoints}
    if len(outputs) != 1 or (args.eval_run is not None and outputs != {output}):
        raise ValueError("batch evaluation requires checkpoints from the same method run")
    output = checkpoints[0].parent.parent
    if len({int(path.name.removeprefix("global_step_")) for path in checkpoints}) != len(checkpoints):
        raise ValueError("different checkpoints cannot share the same optimizer step")
    checkpoints.sort(key=lambda path: int(path.name.removeprefix("global_step_")))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if Path(run["training"]["output_dir"]).resolve() != output:
        raise ValueError("run directory differs from the saved run output_dir")
    plan = output.parent.parent / "plan" / output.name
    plan.mkdir(parents=True, exist_ok=True)
    evaluation_task = args.benchmark if args.eval_type == "benchmark" else "test"
    evaluation_name = f"{evaluation_task}-evaluation" if args.eval_type == "benchmark" else "evaluation"
    result_path = plan / f"{evaluation_name}-result.json"
    destination = (
        output.parent.parent / "eval" / output.name
        / evaluation_task
    )
    results = [
        {
            "checkpoint": str(checkpoint),
            "step": int(checkpoint.name.removeprefix("global_step_")),
            "status": "pending",
            "output_dir": str(destination / checkpoint.name),
            "result_path": str(plan / evaluation_name / f"{checkpoint.name}-result.json"),
            "log_path": str(plan / evaluation_name / f"{checkpoint.name}.log"),
        }
        for checkpoint in checkpoints
    ]
    summary = {
        "action": "evaluate",
        "eval_run": str(output),
        "eval_type": args.eval_type,
        "benchmark": args.benchmark,
        "gpus": args.gpus,
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "status": "running",
        "checkpoints": results,
    }
    write_json(result_path, summary)
    try:
        for checkpoint, result in zip(checkpoints, results, strict=True):
            result["status"] = "running"
            write_json(result_path, summary)
            checkpoint_args = argparse.Namespace(
                **{
                    **vars(args),
                    "checkpoint": [checkpoint],
                    "eval_run": None,
                }
            )
            try:
                run_job(checkpoint_args, checkpoint_series=True)
            except Exception as error:
                result["status"] = "failed"
                result["error"] = f"{type(error).__name__}: {error}"
                print(f"checkpoint evaluation failed: {checkpoint}: {error}", flush=True)
            else:
                result["status"] = "finished"
            write_json(result_path, summary)
    except KeyboardInterrupt:
        for result in results:
            if result["status"] == "running":
                result["status"] = "stopped"
        summary["status"] = "stopped"
        summary["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        )
        write_json(result_path, summary)
        raise
    failures = sum(result["status"] == "failed" for result in results)
    summary["status"] = "failed" if failures else "finished"
    summary["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    write_json(result_path, summary)
    if failures:
        raise RuntimeError(
            f"evaluation failed for {failures}/{len(checkpoints)} checkpoints; see {result_path}"
        )
    return summary


def run_job(args, checkpoint_series=False):
    if args.evaluate_only and (
        args.eval_run is not None or len(args.checkpoint) > 1
    ):
        return run_evaluation_series(args)
    checkpoint = args.checkpoint[0] if args.evaluate_only else args.checkpoint
    checkpoint = checkpoint.expanduser().resolve() if checkpoint is not None else None
    train_only = args.train_only
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
            train_only = json.loads((plan / "result.json").read_text(encoding="utf-8")).get(
                "train_only", False
            )
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
    if args.evaluate_only:
        action = "evaluate"
        evaluation_task = args.benchmark if args.eval_type == "benchmark" else "test"
        evaluation_name = f"{evaluation_task}-evaluation" if args.eval_type == "benchmark" else "evaluation"
        evaluation_plan = plan / evaluation_name
        evaluation_plan.mkdir(parents=True, exist_ok=True)
        result_path = evaluation_plan / f"{checkpoint.name}-result.json"
        log_path = evaluation_plan / f"{checkpoint.name}.log"
        destination = (
            output.parent.parent / "eval" / output.name
            / evaluation_task / checkpoint.name
        )
        command += [
            "-m",
            (
                "latent_working_memory.v4.benchmarks.evaluate"
                if args.eval_type == "benchmark"
                else "latent_working_memory.v4.evaluate"
            ),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(destination),
            "--device",
            "cuda",
        ]
        if args.eval_type == "benchmark":
            command += ["--benchmark", args.benchmark]
            if args.benchmark_data is not None:
                command += ["--benchmark-data", str(args.benchmark_data.expanduser().resolve())]
            if args.lengths is not None:
                command += ["--lengths", *(str(length) for length in args.lengths)]
        else:
            command += ["--max-new-tokens", str(args.max_new_tokens or 128)]
            if args.test_data is not None:
                command += ["--test-data", str(args.test_data.expanduser().resolve())]
        if args.tracking is not None:
            command += ["--tracking", args.tracking]
        if checkpoint_series:
            command += ["--checkpoint-series"]
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
    job = {
        "action": action,
        "gpus": args.gpus,
        "command": command,
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
    }
    if args.evaluate_only:
        job["eval_type"] = args.eval_type
    else:
        job["train_only"] = train_only
    write_json(result_path, {**job, "status": "running"})
    try:
        execute(command, environment, log_path)
        if args.evaluate_only:
            result = {
                "status": "finished",
                "output_dir": str(destination),
            }
        else:
            result = json.loads((output / "training-result.json").read_text(encoding="utf-8"))
            result = {**result, "status": "finished", "train_only": train_only}
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
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run_job(args)
        if not args.evaluate_only and not result["train_only"] and result["complete"]:
            # 训练子进程已退出；复用独立评估入口，在第一张卡上评估最终 checkpoint。
            print("训练完成，开始最终 test 评估。", flush=True)
            run_job(
                parse_args(
                    [
                        "--evaluate-only",
                        "--gpus",
                        args.gpus.split(",")[0],
                        "--checkpoint",
                        result["checkpoint"],
                    ]
                )
            )
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    except KeyboardInterrupt as error:
        raise SystemExit(130) from error


if __name__ == "__main__":
    main()
