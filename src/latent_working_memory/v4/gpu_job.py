"""v4 GPU 入口：训练、续训、恢复和独立评估。"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shlex
import signal
import subprocess
import sys
import threading
from zoneinfo import ZoneInfo

from latent_working_memory.v4.config import (
    ModelConfig,
    TrainingConfig,
    load_experiment,
    load_training_schedule,
)
from latent_working_memory.v4.continuation import load_continuation
from latent_working_memory.v4.evaluation_records import EvaluationHistory, evaluation_config_path
from latent_working_memory.v4.io import write_json
from latent_working_memory.v4.tracking import publish_evaluations, validate_evaluation_publication


DEFAULT_CONFIG = Path("configs/v4/attention_matching/experiment.json")
TRAINING_OPTIONS = (
    "config",
    "run_id",
    "output_root",
    "model_path",
    "data",
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
        "--fork-train", action="store_true",
        help="从 --checkpoint 分叉出新训练计划，并继承截至该点的 train/dev 曲线",
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
        help="global_step_<step> 目录；独立评估可传多个，恢复及新计划续训只接受一个",
    )
    checkpoint_sources.add_argument(
        "--eval-run", type=Path,
        help="评估指定 run 保留的 checkpoint；benchmark 权重全清理时复用已发布成员",
    )
    parser.add_argument("--config", type=Path, help=f"训练预设，默认 {DEFAULT_CONFIG}")
    parser.add_argument(
        "--gpus", type=physical_gpus, default="0,1",
        help="物理 GPU 编号，默认 0,1；独立评估每卡一个进程，并行计算不同 checkpoint",
    )
    parser.add_argument("--run-id", help="实验系列目录，默认上海时间 YYYYMMDD-HHMMSS")
    parser.add_argument("--output-root", type=Path, help="产物父目录，默认 artifacts/v4")
    parser.add_argument("--model-path", type=Path, help="覆盖预设中的本地基座模型位置")
    parser.add_argument("--data", type=Path, help="完整数据集目录，包含 train/dev/test.jsonl")
    parser.add_argument(
        "--test-data", type=Path,
        help="覆盖完整运行或独立 test 的数据目录；独立评估无 test.json 时必填",
    )
    parser.add_argument(
        "--benchmark-data", type=Path,
        help="benchmark 数据目录，默认沿用评估配置；首次计算无评估配置时必填",
    )
    parser.add_argument(
        "--baseline-results", nargs="+", action="extend", metavar="LABEL_OR_PATH",
        help="按 标签 基线运行的eval目录 成对传入；仅用于本次 benchmark 在线展示",
    )
    for name in (
        "micro-batch-size-per-gpu",
        "gradient-accumulation-steps",
        "epochs",
        "dev-every",
        "save-total-limit",
    ):
        parser.add_argument(f"--{name}", type=positive_count)
    parser.add_argument(
        "--seed", type=nonnegative_count, help="公共随机种子，控制初始化、选样、训练打乱及评估"
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
        ("max-test-samples", "长度过滤后抽取的 test QA 样本上限；null 使用全量，不用于 benchmark"),
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
        help="QASPER test 的生成 token 上限，沿用评估配置；无配置时默认 128",
    )
    defaults = [(action, action.default) for action in parser._actions]
    for action, _ in defaults:
        action.default = argparse.SUPPRESS
    args = parser.parse_args(argv)
    supplied_options = set(vars(args))
    for action, default in defaults:
        action.default = default
        if default != argparse.SUPPRESS and not hasattr(args, action.dest):
            setattr(args, action.dest, default)
    if args.baseline_results is not None:
        if len(args.baseline_results) % 2:
            parser.error("--baseline-results requires label/path pairs")
        labels = args.baseline_results[::2]
        if any(not label.strip() for label in labels) or len(set(labels)) != len(labels):
            parser.error("--baseline-results requires nonempty, unique labels")
        args.baseline_results = [
            (label, Path(path).expanduser().resolve())
            for label, path in zip(labels, args.baseline_results[1::2])
        ]
    if args.fork_train:
        if args.resume or args.evaluate_only:
            parser.error("--fork-train cannot be used with --resume or --evaluate-only")
        if args.checkpoint is None:
            parser.error("--fork-train requires --checkpoint")
    if args.resume:
        overrides = supplied_options - {"resume", "checkpoint", "gpus"}
        if overrides:
            options = sorted(
                action.option_strings[0]
                for action, _ in defaults
                if action.dest in overrides
            )
            parser.error(f"--resume does not accept configuration overrides: {', '.join(options)}")
    if args.eval_run is not None and not args.evaluate_only:
        parser.error("--eval-run requires --evaluate-only")
    if args.train_only and (args.test_data is not None or hasattr(args, "max_test_samples")):
        parser.error("--test-data and --max-test-samples cannot be used with --train-only")
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
    elif args.checkpoint is not None and not args.fork_train:
        parser.error("--checkpoint requires --resume, --fork-train or --evaluate-only")
    if args.evaluate_only:
        args.eval_type = args.eval_type or "test"
        if args.eval_type == "test":
            if any(value is not None for value in (
                args.benchmark, args.benchmark_data, args.lengths, args.baseline_results,
            )):
                parser.error(
                    "--benchmark, --benchmark-data, --lengths and --baseline-results "
                    "require --eval-type benchmark"
                )
        else:
            if args.benchmark is None:
                parser.error("--eval-type benchmark requires --benchmark")
            if (
                args.test_data is not None or args.max_new_tokens is not None
                or hasattr(args, "max_test_samples")
            ):
                parser.error(
                    "--test-data, --max-test-samples and --max-new-tokens require --eval-type test"
                )
            if args.lengths is not None and args.benchmark != "ruler":
                parser.error("--lengths requires --benchmark ruler")
            if args.baseline_results is not None and args.tracking == "disabled":
                parser.error("--baseline-results requires online benchmark publication")
    elif any(
        value is not None
        for value in (
            args.eval_type, args.benchmark, args.benchmark_data,
            args.lengths, args.max_new_tokens, args.baseline_results,
        )
    ):
        parser.error(
            "--eval-type, --benchmark, --benchmark-data, --lengths and "
            "--max-new-tokens and --baseline-results require --evaluate-only"
        )
    if args.checkpoint is not None and not args.evaluate_only:
        if len(args.checkpoint) != 1:
            parser.error("multiple checkpoints require --evaluate-only")
        args.checkpoint = args.checkpoint[0]
    return args


def prepare_training(args):
    source_model = None
    source_training = None
    if args.fork_train:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        source, _ = load_continuation(args.checkpoint)
        source_model = ModelConfig(**source["run"]["model"])
        source_training = TrainingConfig(**source["run"]["training"])
        model, training = (
            load_experiment(args.config)
            if args.config is not None else (source_model, source_training)
        )
    else:
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
    if args.data is not None:
        overrides["dataset_dir"] = str(args.data.expanduser().resolve())
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
        group=(
            args.group or training.group
            or (source_training.group if source_training is not None else None) or run_id
        ),
        tags=tuple(sorted(set(training.tags) | set(args.tag or []))),
    )
    if source_model is not None and model != source_model:
        different = [
            name for name, value in asdict(source_model).items()
            if getattr(model, name) != value
        ]
        raise ValueError(
            "--fork-train model must match the source checkpoint after CLI overrides; "
            "differing fields: " + ", ".join(different)
        )
    plan.mkdir(parents=True)
    config_path = plan / "config.json"
    training_values = asdict(training)
    seed = training_values.pop("seed")
    write_json(config_path, {"seed": seed, "model": asdict(model)})
    write_json(plan / "train.json", {
        "execution": {
            "train_only": args.train_only,
            "test_data": str(
                (args.test_data or Path(training.dataset_dir)).expanduser().resolve()
            ),
            "max_test_samples": getattr(args, "max_test_samples", None),
            **(
                {"continue_from": str(args.checkpoint)}
                if args.fork_train else {}
            ),
        },
        "training": training_values,
    })
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


def _terminate_evaluation_process(process):
    """回收调度器创建的进程组，避免中断后留下 GPU worker。"""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _write_evaluation_status(path, summary, results):
    """集中保存各 checkpoint 最近状态，进度仅统计本次选择。"""
    summary["progress"] = {
        "total": len(results),
        **{
            status: sum(result["status"] == status for result in results)
            for status in ("pending", "running", "finished", "failed", "stopped")
        },
    }
    write_json(path, summary)


def _compute_evaluations(args, results, summary, result_path):
    """每张 GPU 固定一个槽位，只并行计算，发布由父进程负责。"""
    pending = Queue()
    for result in results:
        if result["status"] == "pending":
            pending.put(result)
    stopping = threading.Event()
    state_lock = threading.Lock()
    processes = set()

    def worker(gpu):
        while not stopping.is_set():
            try:
                result = pending.get_nowait()
            except Empty:
                return
            command = result["command"]
            prefix = f"[evaluation] gpu={gpu} checkpoint=global_step_{result['step']}"
            error = None
            try:
                with state_lock:
                    if stopping.is_set():
                        return
                    result.update(
                        status="running", gpu=gpu, gpus=gpu,
                        started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                    )
                    _write_evaluation_status(result_path, summary, results)
                with Path(result["log_path"]).open("a", encoding="utf-8") as log:
                    invocation = "$ " + shlex.join(command) + "\n"
                    print(f"{prefix} {invocation}", end="", flush=True)
                    log.write(invocation)
                    log.flush()
                    with state_lock:
                        if stopping.is_set():
                            return
                        process = subprocess.Popen(
                            command,
                            env=dict(
                                os.environ,
                                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                                CUDA_VISIBLE_DEVICES=gpu,
                                PYTHONUNBUFFERED="1",
                                TOKENIZERS_PARALLELISM="false",
                            ),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            encoding="utf-8",
                            bufsize=1,
                            start_new_session=True,
                        )
                        processes.add(process)
                    try:
                        for line in process.stdout:
                            print(f"{prefix} {line}", end="", flush=True)
                            log.write(line)
                            log.flush()
                        code = process.wait()
                    finally:
                        if process.poll() is None:
                            _terminate_evaluation_process(process)
                        with state_lock:
                            processes.discard(process)
                        process.stdout.close()
                    if code != 0:
                        raise subprocess.CalledProcessError(code, command)
            except Exception as caught:
                error = caught
                print(f"{prefix} compute failed: {error}", flush=True)
            with state_lock:
                result["status"] = (
                    "stopped" if stopping.is_set() else "failed" if error is not None else "finished"
                )
                if error is not None:
                    result["error"] = f"{type(error).__name__}: {error}"
                result.update(
                    return_code=(
                        error.returncode if isinstance(error, subprocess.CalledProcessError)
                        else 1 if error is not None else 0
                    ),
                    finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                )
                _write_evaluation_status(result_path, summary, results)

    threads = []
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for gpu in args.gpus.split(",")[:len(results)]:
            thread = threading.Thread(target=worker, args=(gpu,), name=f"v4-eval-gpu-{gpu}")
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    except BaseException:
        stopping.set()
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        with state_lock:
            active = list(processes)
        for process in active:
            _terminate_evaluation_process(process)
        for thread in threads:
            thread.join()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _publish_evaluation_results(args, results, summary, result_path, run):
    """仅用完整结果更新报告；原始样本、tokenizer 和权重均不参与发布。"""
    publications = []
    for result in results:
        if result["status"] != "finished":
            continue
        result["publication_status"] = "pending"
        _write_evaluation_status(result_path, summary, results)
        checkpoint = Path(result["checkpoint"])
        output_dir = Path(result["output_dir"])
        try:
            saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            history = EvaluationHistory(output_dir, saved)
            with history.lock():
                saved = history.cached_summary()
            publications.append((result, checkpoint, saved, output_dir))
        except Exception as error:
            result.update(
                status="failed", publication_status="failed",
                error=f"publication failed: {type(error).__name__}: {error}",
            )
            print(f"[evaluation] {checkpoint.name} {result['error']}", flush=True)
    if publications:
        try:
            publish_evaluations(
                run, [row[1:] for row in publications], baseline_results=args.baseline_results
            )
        except Exception as error:
            for result, checkpoint, _, _ in publications:
                result.update(
                    status="failed", publication_status="failed",
                    error=f"publication failed: {type(error).__name__}: {error}",
                )
                print(f"[evaluation] {checkpoint.name} {result['error']}", flush=True)
        else:
            for result, _, _, _ in publications:
                result["publication_status"] = "finished"
    _write_evaluation_status(result_path, summary, results)


def _run_evaluation_locked(args, checkpoints, output, run):
    evaluation_task = args.benchmark if args.eval_type == "benchmark" else "test"
    destination = output.parent.parent / "eval" / output.name / evaluation_task
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "status.json"
    config_path = evaluation_config_path(destination / checkpoints[0].name)
    data_dir = args.benchmark_data if args.eval_type == "benchmark" else args.test_data
    data_option = "--benchmark-data" if args.eval_type == "benchmark" else "--test-data"
    cached_only = args.eval_type == "benchmark" and all(
        (destination / checkpoint.name / "summary.json").is_file() for checkpoint in checkpoints
    )
    if data_dir is None and not config_path.is_file() and not cached_only:
        raise ValueError(f"{config_path} is missing; specify {data_option}")
    tracking = args.tracking or (
        "online" if run["training"]["swanlab_project"] is not None else "disabled"
    )
    if args.baseline_results is not None and tracking != "online":
        raise ValueError("--baseline-results requires an existing online training run")
    if tracking == "online":
        for checkpoint in checkpoints:
            validate_evaluation_publication(
                run, checkpoint, int(checkpoint.name.removeprefix("global_step_"))
            )
    checkpoint_statuses = (
        json.loads(result_path.read_text(encoding="utf-8"))["checkpoints"]
        if result_path.is_file() else {}
    )
    results = []
    for checkpoint in checkpoints:
        output_dir = destination / checkpoint.name
        output_dir.mkdir(parents=True, exist_ok=True)
        cached = None
        if args.eval_type == "benchmark" and (output_dir / "summary.json").is_file():
            cached = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            if cached["benchmark"] != args.benchmark or cached["checkpoint"] != {
                "path": str(checkpoint),
                "step": int(checkpoint.name.removeprefix("global_step_")),
            }:
                raise ValueError(f"cached summary does not match the selected checkpoint: {output_dir}")
            if args.benchmark_data is not None and (
                str(args.benchmark_data.expanduser().resolve()) != cached["data"]["directory"]
            ):
                raise ValueError("cached publication cannot change --benchmark-data")
            if args.lengths is not None and args.lengths != cached["protocol"]["lengths"]:
                raise ValueError("cached publication cannot change --lengths")
            history = EvaluationHistory(output_dir, cached)
            with history.lock():
                cached = history.cached_summary()
        command = [
            sys.executable,
            "-m",
            (
                "latent_working_memory.v4.benchmarks.evaluate"
                if args.eval_type == "benchmark" else "latent_working_memory.v4.evaluate"
            ),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(output_dir),
            "--device", "cuda",
            "--tracking", tracking,
            "--defer-publication",
        ]
        if args.eval_type == "benchmark":
            command += ["--benchmark", args.benchmark]
            if args.benchmark_data is not None:
                command += ["--benchmark-data", str(args.benchmark_data.expanduser().resolve())]
            if args.lengths is not None:
                command += ["--lengths", *(str(length) for length in args.lengths)]
        else:
            if args.max_new_tokens is not None:
                command += ["--max-new-tokens", str(args.max_new_tokens)]
            if args.test_data is not None:
                command += ["--test-data", str(args.test_data.expanduser().resolve())]
            if hasattr(args, "max_test_samples"):
                limit = args.max_test_samples
                command += ["--max-test-samples", "null" if limit is None else str(limit)]
        results.append({
            "action": "evaluate",
            "eval_type": args.eval_type,
            "checkpoint": str(checkpoint),
            "step": int(checkpoint.name.removeprefix("global_step_")),
            "status": "finished" if cached is not None else "pending",
            **(
                {"compute_status": "cached", "return_code": 0}
                if cached is not None else {"command": command}
            ),
            "output_dir": str(output_dir),
            "log_path": str(output_dir / "eval.log"),
        })
        checkpoint_statuses[checkpoint.name] = results[-1]
    summary = {
        "action": "evaluate",
        "eval_run": str(output),
        "eval_type": args.eval_type,
        "benchmark": args.benchmark,
        "gpus": args.gpus,
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "status": "running",
        "selected_checkpoints": [checkpoint.name for checkpoint in checkpoints],
        "checkpoints": checkpoint_statuses,
    }
    _write_evaluation_status(result_path, summary, results)
    try:
        if any(result["status"] == "pending" for result in results):
            _compute_evaluations(args, results, summary, result_path)
        else:
            print("[evaluation] 使用已完成的汇总结果更新展示，无需加载模型或原始数据。", flush=True)
        if tracking == "online":
            _publish_evaluation_results(args, results, summary, result_path, run)
    except KeyboardInterrupt:
        for result in results:
            if (
                result["status"] in {"pending", "running"}
                or result.get("publication_status") == "pending"
            ):
                result["status"] = "stopped"
                if result.get("publication_status") == "pending":
                    result["publication_status"] = "stopped"
        summary["status"] = "stopped"
        summary["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        _write_evaluation_status(result_path, summary, results)
        raise
    failures = sum(result["status"] != "finished" for result in results)
    summary["status"] = "failed" if failures else "finished"
    summary["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    _write_evaluation_status(result_path, summary, results)
    if failures:
        raise RuntimeError(
            f"evaluation failed for {failures}/{len(checkpoints)} checkpoints; see {result_path}"
        )
    return summary


def run_evaluation(args):
    if args.eval_run is not None:
        output = args.eval_run.expanduser().resolve()
        if not output.is_dir() or output.parent.name != "train":
            raise ValueError("--eval-run requires an existing <series>/train/<method-dir>")
        checkpoints = [
            path.parent
            for path in output.glob("checkpoints/global_step_*/state.pt")
            if path.is_file() and re.fullmatch(r"global_step_[0-9]+", path.parent.name)
        ]
        if not checkpoints and args.eval_type == "benchmark":
            publication_path = (
                output.parent.parent / "eval" / output.name / args.benchmark / "swanlab.json"
            )
            if publication_path.is_file():
                publication = json.loads(publication_path.read_text(encoding="utf-8"))
                checkpoints = [
                    Path(json.loads(Path(path).read_text(encoding="utf-8"))["checkpoint"]["path"])
                    for path in publication["summaries"]
                ]
        if not checkpoints:
            raise ValueError(f"no retained checkpoints or published evaluation results found in {output}")
    else:
        checkpoints = args.checkpoint
    checkpoints = [path.expanduser().resolve() for path in checkpoints]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("checkpoint list contains duplicate directories")
    for checkpoint in checkpoints:
        if checkpoint.parent.name != "checkpoints" or checkpoint.parent.parent.parent.name != "train":
            raise ValueError("checkpoint must belong to <series>/train/<method-dir>/checkpoints")
        if re.fullmatch(r"global_step_[0-9]+", checkpoint.name) is None:
            raise ValueError("checkpoint must be a global_step_<step> directory")
        cached = (
            args.eval_type == "benchmark"
            and (
                checkpoint.parents[3] / "eval" / checkpoint.parents[1].name
                / args.benchmark / checkpoint.name / "summary.json"
            ).is_file()
        )
        if not cached and not (checkpoint / "state.pt").is_file():
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

    def interrupt_evaluation(signum, frame):
        raise KeyboardInterrupt(f"evaluation interrupted by {signal.Signals(signum).name}")

    with (output / ".evaluation-scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"another evaluation scheduler is running for {output}")
        previous_handler = signal.signal(signal.SIGTERM, interrupt_evaluation)
        try:
            return _run_evaluation_locked(args, checkpoints, output, run)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


def run_job(args):
    if args.evaluate_only:
        return run_evaluation(args)
    checkpoint = args.checkpoint
    checkpoint = checkpoint.expanduser().resolve() if checkpoint is not None else None
    if not args.resume:
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
            load_training_schedule(plan / "train.json")
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
    gpu_count = len(args.gpus.split(","))
    if gpu_count > 1:
        command += [
            "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpu_count}",
        ]
    command += [
        "-m", "latent_working_memory.v4.train", "--config", str(config_path),
        "--train-config", str(plan / "train.json"), "--device", "cuda",
    ]
    if args.resume:
        command += ["--resume", str(checkpoint)]
    job = {
        "action": "resume" if args.resume else "fork-train" if args.fork_train else "train",
        "gpus": args.gpus,
        "command": command,
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
    }
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "status.json"
    write_json(result_path, {**job, "status": "running"})
    try:
        execute(command, environment, output / "train.log")
        result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        result = {**result, "status": "finished"}
    except (Exception, KeyboardInterrupt) as error:
        write_json(
            result_path,
            {
                **job,
                "status": "stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                "return_code": (
                    error.returncode if isinstance(error, subprocess.CalledProcessError)
                    else 130 if isinstance(error, KeyboardInterrupt) else None
                ),
                "error": f"{type(error).__name__}: {error}",
                "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            },
        )
        raise
    write_json(
        result_path,
        {
            **job, "status": "finished", "return_code": 0,
            "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        },
    )
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run_job(args)
        if args.evaluate_only or not result["complete"]:
            return
        output = Path(result["checkpoint"]).parent.parent
        plan = output.parent.parent / "plan" / output.name
        execution = load_training_schedule(plan / "train.json")["execution"]
        if not execution["train_only"]:
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
                        "--test-data",
                        execution["test_data"],
                        "--max-test-samples",
                        (
                            "null" if execution["max_test_samples"] is None
                            else str(execution["max_test_samples"])
                        ),
                    ]
                )
            )
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    except KeyboardInterrupt as error:
        raise SystemExit(130) from error


if __name__ == "__main__":
    main()
