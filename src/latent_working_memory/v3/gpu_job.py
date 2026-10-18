"""GPU 实验入口：连续训练与评估，或从已有 checkpoint 单独补跑评估。"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

from latent_working_memory.v3 import job_plan
from latent_working_memory.v3.config import (
    DYNAMIC_METHODS,
    METHODS,
    WRITER_MODES,
    EvaluationConfig,
    load_common_config,
    load_evaluation,
    parse_optional_positive_integer,
)
from latent_working_memory.v3.evaluation_records import EvaluationHistory, evaluation_log_path
from latent_working_memory.v3.checkpoint import read_checkpoint
from latent_working_memory.v3.evaluate import (
    publish_cached_evaluations,
    validate_evaluation_publication,
)
from latent_working_memory.v3.training_status import read_training_status, update_training_status
from latent_working_memory.tracking_credentials import swanlab_api_key


def nonnegative_count(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def positive_count(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def save_frequency(value):
    if value == "after_each_epoch":
        return value
    return positive_count(value)


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
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--train-only",
        action="store_true",
        help="只启动新训练，保留训练中 dev 验证，跳过最终生成评估",
    )
    actions.add_argument(
        "--evaluate-only", action="store_true", help="评估一个或多个已有 checkpoint，不启动训练"
    )
    actions.add_argument(
        "--resume", action="store_true", help="恢复原阶段，并完成该方法的剩余训练与评估"
    )
    checkpoint_sources = parser.add_mutually_exclusive_group()
    checkpoint_sources.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        help="checkpoint 目录；--evaluate-only 可传多个，初始化与恢复只接受一个",
    )
    checkpoint_sources.add_argument(
        "--eval-run",
        type=Path,
        help="评估指定方法 run 目录中全部保留的 checkpoint；仅用于 --evaluate-only",
    )
    parser.add_argument("--mode", choices=job_plan.LEVELS, default="full")
    parser.add_argument("--method", choices=(*METHODS, "dynamic_pretrain"), help="本次训练的方法")
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
        help="QA 数据目录，训练、验证和最终评估分别读取 train/dev/test.jsonl",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        help="覆盖最终 test 数据目录；独立评估无 test.json 时必填，读取 test.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_count,
        help="新训练的公共随机种子；默认沿用配置顶层 seed，评估读取公共 config.json 的 seed",
    )
    parser.add_argument(
        "--model-path", help="覆盖预设模型位置，当前预设为 ~/models/Qwen3-4B-Instruct-2507"
    )
    parser.add_argument(
        "--gpus",
        type=physical_gpus,
        default="0,1",
        help="逗号分隔且不重复的物理 GPU 编号；训练使用所选卡，--evaluate-only 每卡一个计算进程",
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
        "--task-qa-count", type=positive_count, help="每段训练 task QA 总题数，默认 8；首段只用新题"
    )
    parser.add_argument(
        "--gate-qa-count", type=positive_count, help="训练和 dev 每次门控的历史 QA 数，默认 8"
    )
    parser.add_argument(
        "--task-new-ratio", type=probability, help="训练 task 新题比例，默认 0.5；新题数须为正整数"
    )
    parser.add_argument(
        "--lm-ratio",
        type=probability,
        help="AE＋LM 预训练中每条来源选择 LM 的概率，默认 0.5；AutoCompressors 始终使用 LM",
    )
    parser.add_argument(
        "--max-pretrain-data-tokens",
        type=parse_optional_positive_integer,
        default=argparse.SUPPRESS,
        help="AE/LM 训练正文 token 上限，预设 12288；null 跳过上限筛选，仅过滤 train",
    )
    parser.add_argument(
        "--max-qa-data-tokens",
        type=parse_optional_positive_integer,
        default=argparse.SUPPRESS,
        help="FactQA 训练正文 token 上限，预设 12288；null 跳过上限筛选，仅过滤 train",
    )
    parser.add_argument(
        "--lm-target-tokens",
        type=positive_count,
        help="预训练 LM 目标的真实 token 数，默认 512；续文候选不足时沿用任务回退规则",
    )
    parser.add_argument(
        "--icae-min-segments",
        type=positive_count,
        help="ICAE-multi 随机分块数量下限，默认 3；总容量 K 均分给各块",
    )
    parser.add_argument(
        "--icae-max-segments",
        type=positive_count,
        help="ICAE-multi 随机分块数量上限，默认 6，不超过 K",
    )
    parser.add_argument(
        "--ac-num-segments",
        type=positive_count,
        help="AutoCompressors 正文压缩块数量，默认 4；总容量 K 均分给各块",
    )
    parser.add_argument(
        "--bptt-steps",
        type=parse_optional_positive_integer,
        default=argparse.SUPPRESS,
        help="每个反传窗口的写入轮数，null 表示完整 BPTT；AutoCompressors 同时据此分组随机切分文本",
    )
    parser.add_argument(
        "--writer-mode",
        choices=WRITER_MODES,
        help="动态后训练的写入方式：local、tag、mask 或 dual_lora；默认沿用配置",
    )
    parser.add_argument(
        "--tag-tokens",
        type=positive_count,
        help="tag 写入每个动作的可训练控制 token 数，默认 3",
    )
    parser.add_argument(
        "--append-slots",
        type=positive_count,
        help="动态后训练每次追加的 slots 数，预设 32；首次写入为 K slots",
    )
    for category, stages in (("pretrain", "pretrain/lm"), ("qa", "qa/warmup/policy")):
        parser.add_argument(
            f"--max-{category}-train-samples",
            type=parse_optional_positive_integer,
            default=argparse.SUPPRESS,
            help=f"{stages} 阶段训练样本上限，null 表示不限；smoke 另受档位上限约束",
        )
    parser.add_argument(
        "--max-dev-samples",
        type=parse_optional_positive_integer,
        default=argparse.SUPPRESS,
        help="训练中 dev 验证的样本上限：正整数或 null；smoke 默认 4，显式 null 不限",
    )
    parser.add_argument(
        "--max-test-samples",
        type=parse_optional_positive_integer,
        default=argparse.SUPPRESS,
        help="最终 test 评估的样本上限：正整数或 null；smoke 默认 2，显式 null 不限",
    )
    parser.add_argument(
        "--max-steps",
        type=parse_optional_positive_integer,
        default=argparse.SUPPRESS,
        help="覆盖运行档位的训练步数上限，null 表示不限",
    )
    parser.add_argument(
        "--dev-every", type=positive_count, help="每隔多少 optimizer step 验证 dev"
    )
    parser.add_argument(
        "--save-every",
        type=save_frequency,
        help="保存间隔：正整数 optimizer step 数或 after_each_epoch；full 沿用配置，smoke 默认 1",
    )
    parser.add_argument(
        "--save-total-limit",
        type=positive_count,
        help="同一方法所有阶段合计保留的 checkpoint 数，默认 2",
    )
    parser.add_argument(
        "--max-new-tokens", type=positive_count, help="最终评估生成上限，初次默认 64"
    )
    for name in ("threshold-i", "threshold-d", "threshold-g", "eta"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--tracking", choices=("online", "disabled"), default="online")
    parser.add_argument("--swanlab-project", default="latent-working-memory-v3")
    parser.add_argument("--group", help="显式实验系列 group；默认使用本次运行标识")
    parser.add_argument(
        "--run-id",
        help="产物目录名和默认 group；默认上海时间 YYYYMMDD-HHMMSS，动态后训练未指定时继承预训练标识",
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/v3"))
    defaults = [(action, action.default) for action in parser._actions]
    for action, _ in defaults:
        action.default = argparse.SUPPRESS
    args = parser.parse_args(argv)
    supplied = set(vars(args))
    for action, default in defaults:
        action.default = default
        if default != argparse.SUPPRESS and not hasattr(args, action.dest):
            setattr(args, action.dest, default)
    if args.resume:
        overrides = supplied - {"resume", "checkpoint", "gpus"}
        if overrides:
            options = sorted(
                action.option_strings[0]
                for action, _ in defaults
                if action.dest in overrides
            )
            parser.error(f"--resume does not accept configuration overrides: {', '.join(options)}")
    if args.eval_run is not None and not args.evaluate_only:
        parser.error("--eval-run requires --evaluate-only")
    if args.test_data is not None and (args.train_only or args.resume):
        parser.error("--test-data is not supported with --train-only or --resume")
    if args.evaluate_only:
        if "seed" in supplied:
            parser.error("--seed is only supported when starting new training")
        if args.checkpoint is None and args.eval_run is None:
            parser.error("--evaluate-only requires --checkpoint or --eval-run")
        return args
    if args.checkpoint is not None:
        if len(args.checkpoint) != 1:
            parser.error("multiple checkpoints require --evaluate-only")
        args.checkpoint = args.checkpoint[0]
    if args.resume:
        if args.checkpoint is None:
            parser.error("--resume requires --checkpoint")
        return args
    if args.method is None:
        parser.error("training requires --method")
    if args.run_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("run-id must contain only letters, digits, '-' and '_'")
    if args.checkpoint is not None and args.method not in (*DYNAMIC_METHODS, "dynamic_pretrain"):
        parser.error("checkpoint initialization is only supported by dynamic methods")
    if args.writer_mode is not None and args.method not in DYNAMIC_METHODS:
        parser.error("--writer-mode is only supported by dynamic posttraining methods")
    if args.tag_tokens is not None and args.method not in DYNAMIC_METHODS:
        parser.error("--tag-tokens is only supported by dynamic tag posttraining")
    if args.append_slots is not None and args.method in {
        "icae_single",
        "icae_multi",
        "autocompressors",
        "dynamic_pretrain",
    }:
        parser.error("--append-slots is only supported by dynamic posttraining methods")
    if "bptt_steps" in vars(args) and args.method not in (
        "autocompressors",
        *DYNAMIC_METHODS,
    ):
        parser.error("--bptt-steps is only supported by AutoCompressors or dynamic methods")
    if args.ac_num_segments is not None and args.method != "autocompressors":
        parser.error("--ac-num-segments is only supported by AutoCompressors")
    return args


def training_command(jobs, gpu_count, max_steps, resume=None):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        "-m",
        "latent_working_memory.v3.train",
        "--config",
        str(jobs[0].config_path),
        "--train-config",
        str(jobs[0].train_config_path),
        "--stages",
        *(job.config.objective.stage for job in jobs),
        "--device",
        "cuda",
    ]
    if max_steps is not None:
        command.extend(("--max-steps", str(max_steps)))
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def evaluation_command(args, config_path, checkpoint, output, log_to_swanlab=None):
    command = [
        sys.executable,
        "-m",
        "latent_working_memory.v3.evaluate",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--device",
        "cuda",
    ]
    if log_to_swanlab is None:
        log_to_swanlab = args.tracking == "online"
    if log_to_swanlab:
        command.append("--log-to-swanlab")
    return command


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def save_training_stage(job, config):
    """将阶段的实际配置写回训练计划，保留其他阶段。"""
    training = json.loads(job.train_config_path.read_text(encoding="utf-8"))
    index = next(
        index
        for index, stage in enumerate(training["stages"])
        if stage["objective"]["stage"] == job.config.objective.stage
    )
    training["stages"][index] = {
        "objective": config["objective"],
        "training": {name: value for name, value in config["training"].items() if name != "seed"},
    }
    save_json(job.train_config_path, training)


def _save_training_status(directory, result, status, error=None):
    """调度器只更新执行字段，保留 worker 登记和评估追加的最新内容。"""
    result["status"] = status
    result.pop("error", None)
    if error is not None:
        result["error"] = str(error)
    with update_training_status(directory) as saved:
        saved["status"] = status
        saved["checkpoints"] = result["checkpoints"]
        for path in result["summaries"]:
            if path not in saved["summaries"]:
                saved["summaries"].append(path)
        result["summaries"][:] = saved["summaries"]
        saved.pop("error", None)
        if error is not None:
            saved["error"] = str(error)


def _write_evaluation_status(path, summary, results):
    """集中保存各 checkpoint 最近状态，进度只统计本次选择。"""
    summary["progress"] = {
        "total": len(results),
        **{
            status: sum(result["status"] == status for result in results)
            for status in ("pending", "running", "finished", "failed", "stopped")
        },
    }
    save_json(
        path,
        {key: value for key, value in summary.items() if key != "summaries"},
    )


def resolve_evaluation_config(args, config_path, level=None, initial_seed=None):
    """沿用已有配置；独立评估首次创建时须显式指定数据，其他字段使用默认值。"""
    if args.resume:
        return load_evaluation(config_path)
    if config_path.is_file():
        config = load_evaluation(config_path)
    else:
        dataset_dir = args.test_data
        if dataset_dir is None and not args.evaluate_only:
            dataset_dir = args.qa_data
        if dataset_dir is None:
            raise ValueError(f"{config_path} is missing; specify --test-data to create it")
        seed = initial_seed
        if seed is None:
            seed = load_common_config(config_path.parent / "config.json")["seed"]
        config = EvaluationConfig(
            dataset_dir=str(Path(dataset_dir).resolve()),
            seed=seed,
            max_test_samples=level["max_test_samples"] if level is not None else None,
        )
    updates = {}
    for argument, field in (
        ("test_data", "dataset_dir"),
        ("max_new_tokens", "max_new_tokens"),
    ):
        value = getattr(args, argument)
        if value is not None:
            if argument == "test_data":
                value = str(value.resolve())
            updates[field] = value
    if "max_test_samples" in vars(args):
        updates["max_test_samples"] = args.max_test_samples
    return replace(config, **updates)


def save_evaluation_config(config_path, config):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(config_path, {name: value for name, value in asdict(config).items() if name != "seed"})


def require_dataset_entries(directories):
    """检查规范数据入口，不扫描下载分片或校验资源完整性。"""
    files = {
        Path(directory) / name
        for directory in directories
        for name in ("preparation.json", "train.jsonl", "dev.jsonl", "test.jsonl")
    }
    for path in sorted(files):
        if not path.is_file():
            raise FileNotFoundError(f"required dataset entry is missing: {path}")


def execute(command, environment, log_path, append=False):
    print(f"$ {shlex.join(command)}", flush=True)
    with log_path.open("a" if append else "w", encoding="utf-8") as log:
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


def execution_environment(args):
    environment = dict(
        os.environ,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=args.gpus,
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    if args.tracking == "online":
        environment["SWANLAB_API_KEY"] = swanlab_api_key()
    return environment


def checkpoint_evaluation_identity(checkpoint):
    stage_directory = job_plan.checkpoint_stage_directory(checkpoint)
    saved = read_checkpoint(checkpoint)
    run = saved["run"]
    if run["config"]["objective"]["method"] == "dynamic":
        raise ValueError("shared dynamic pretraining has no FactQA memory policy")
    stage_step = saved["cursor"]["step"]
    if (
        int(checkpoint.name.removeprefix("global_step_")) != stage_step
        or Path(run["config"]["training"]["output_dir"]).resolve() != stage_directory
        or run["config"]["objective"]["stage"] != stage_directory.name
        or json.loads((stage_directory / "run.json").read_text(encoding="utf-8")) != run
    ):
        raise ValueError("checkpoint identity differs from its saved stage directory")
    return stage_directory, run["step_offset"] + stage_step


def evaluate_checkpoints(args, checkpoints, summaries):
    """自动 test 复用独立评估调度，只使用训练所选的第一张 GPU。"""
    method_directory = job_plan.checkpoint_stage_directory(checkpoints[0]).parent
    evaluation_args = argparse.Namespace(**{**vars(args), "gpus": args.gpus.split(",")[0]})
    try:
        _schedule_evaluation(evaluation_args, checkpoints, method_directory)
    finally:
        result_path = (
            method_directory.parent.parent / "eval" / method_directory.name / "test" / "status.json"
        )
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            for key in result["selected_checkpoints"]:
                checkpoint_status = result["checkpoints"][key]
                if checkpoint_status["compute_status"] in {"cached", "finished"}:
                    summary = str(Path(checkpoint_status["output_dir"]) / "summary.json")
                    if summary not in summaries:
                        summaries.append(summary)


def _terminate_evaluation_process(process):
    """回收本调度器启动的进程组，避免中断后留下 GPU worker。"""
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


def _compute_evaluations(args, tasks, results, summary, result_path, environment):
    """每张 GPU 固定一个槽位，仅并行计算，不在子进程发布 SwanLab。"""
    pending = Queue()
    for task in tasks:
        pending.put(task)
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
            prefix = (
                f"[evaluation] gpu={gpu} checkpoint={result['stage']}"
                f"/global_step_{result['stage_step']}"
            )
            error = None
            try:
                with state_lock:
                    if stopping.is_set():
                        return
                    result.update(
                        status="running", compute_status="running", gpu=gpu,
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
                            env=dict(environment, CUDA_VISIBLE_DEVICES=gpu),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
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
                status = "stopped" if stopping.is_set() else "failed" if error is not None else "finished"
                result.update(
                    status=status, compute_status=status,
                    return_code=(
                        error.returncode if isinstance(error, subprocess.CalledProcessError)
                        else 1 if error is not None else 0
                    ),
                    finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                )
                if status != "finished":
                    result["publication_status"] = "skipped"
                if error is not None:
                    result["error"] = f"{type(error).__name__}: {error}"
                _write_evaluation_status(result_path, summary, results)
                completed = sum(item["compute_status"] not in {"pending", "running"} for item in results)
                print(
                    f"{prefix} compute {status} completed={completed}/{len(results)} "
                    f"log={result['log_path']}",
                    flush=True,
                )

    threads = []
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for gpu in args.gpus.split(",")[:len(tasks)]:
            thread = threading.Thread(target=worker, args=(gpu,), name=f"v3-eval-gpu-{gpu}")
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


def _run_evaluation_locked(args, checkpoints, method_directory, started):
    identities = {path: checkpoint_evaluation_identity(path) for path in checkpoints}
    if len({identity[1] for identity in identities.values()}) != len(checkpoints):
        raise ValueError("different checkpoints cannot share the same global optimizer step")
    directory = method_directory.parent.parent
    checkpoints.sort(key=lambda path: identities[path][1])
    print(
        f"[evaluation] run={method_directory.name} checkpoints={len(checkpoints)} gpus={args.gpus}",
        flush=True,
    )
    plan_directory = directory / "plan" / method_directory.name
    config_path = plan_directory / "test.json"
    config = resolve_evaluation_config(args, config_path)
    require_dataset_entries([config.dataset_dir])
    training_dir = (
        validate_evaluation_publication(checkpoints) if args.tracking == "online" else None
    )
    evaluation_directory = directory / "eval" / method_directory.name / "test"
    outputs = {
        checkpoint: (
            evaluation_directory / identities[checkpoint][0].name / checkpoint.name
        )
        for checkpoint in checkpoints
    }
    results, tasks = [], []
    for checkpoint in checkpoints:
        output = outputs[checkpoint]
        stage, global_step = identities[checkpoint]
        log_path = evaluation_log_path(output)
        result = {
            "checkpoint": str(checkpoint),
            "stage": stage.name,
            "stage_step": int(checkpoint.name.removeprefix("global_step_")),
            "global_step": global_step,
            "status": "pending",
            "compute_status": "pending",
            "publication_status": "pending" if training_dir is not None else "disabled",
            "command": evaluation_command(
                args, output / "eval.json", checkpoint, output, log_to_swanlab=False
            ),
            "output_dir": str(output),
            "log_path": str(log_path),
            "gpu": None,
        }
        history = EvaluationHistory(output, asdict(config))
        cached = history.cached_summary(checkpoint)
        if cached is None:
            tasks.append(result)
        else:
            result.update(
                status="finished", compute_status="cached", return_code=0,
                started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            )
            print(
                f"[evaluation] source={stage.name}/{checkpoint.name} cache ready "
                f"samples={cached['trajectories']} summary={output / 'summary.json'}",
                flush=True,
            )
        results.append(result)
    environment = execution_environment(args)
    save_evaluation_config(config_path, config)
    result_path = evaluation_directory / "status.json"
    checkpoint_statuses = (
        json.loads(result_path.read_text(encoding="utf-8"))["checkpoints"]
        if result_path.is_file() else {}
    )
    selected = [f"{result['stage']}/global_step_{result['stage_step']}" for result in results]
    checkpoint_statuses.update(zip(selected, results))
    summary = {
        "action": "evaluate",
        "eval_run": str(method_directory),
        "eval_type": "test",
        "gpus": args.gpus,
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "status": "running",
        "selected_checkpoints": selected,
        "checkpoints": checkpoint_statuses,
        "summaries": [],
    }
    for result in results:
        Path(result["output_dir"]).mkdir(parents=True, exist_ok=True)
        if result["compute_status"] == "cached":
            with Path(result["log_path"]).open("a", encoding="utf-8") as log:
                log.write(f"[evaluation] reusing local result: {result['output_dir']}/summary.json\n")
    _write_evaluation_status(result_path, summary, results)
    print(
        f"[evaluation] computation started cached={len(results) - len(tasks)} pending={len(tasks)} "
        f"gpu_slots={min(len(args.gpus.split(',')), len(tasks))}",
        flush=True,
    )
    try:
        _compute_evaluations(args, tasks, results, summary, result_path, environment)
        successful = [result for result in results if result["compute_status"] in {"cached", "finished"}]
        summary["summaries"] = [str(Path(result["output_dir"]) / "summary.json") for result in successful]
        training_result_path = method_directory / "status.json"
        if training_result_path.is_file():
            with update_training_status(method_directory) as training_result:
                for path in summary["summaries"]:
                    if path not in training_result["summaries"]:
                        training_result["summaries"].append(path)
        if training_dir is not None and successful:
            print(f"[evaluation] CPU publication started checkpoints={len(successful)}", flush=True)
            failures = dict(publish_cached_evaluations(
                [Path(result["checkpoint"]) for result in successful],
                [Path(result["output_dir"]) for result in successful],
                config,
                training_dir,
            ))
            for result in successful:
                error = failures.get(Path(result["checkpoint"]))
                result["publication_status"] = "failed" if error is not None else "finished"
                if error is not None:
                    result.update(status="failed", error=f"publication failed: {type(error).__name__}: {error}")
                    print(f"[evaluation] {result['stage']}/{Path(result['checkpoint']).name} {result['error']}", flush=True)
                result["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
            _write_evaluation_status(result_path, summary, results)
    except BaseException as error:
        stopped = isinstance(error, KeyboardInterrupt)
        summary["summaries"] = [
            str(Path(result["output_dir"]) / "summary.json")
            for result in results if result["compute_status"] in {"cached", "finished"}
        ]
        for result in results:
            if result["status"] in {"pending", "running"}:
                result["status"] = "stopped" if stopped else "failed"
                result["compute_status"] = result["status"]
            if result["publication_status"] == "pending":
                result["publication_status"] = "stopped" if stopped else "failed"
            result["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        summary.update(
            status="stopped" if stopped else "failed",
            error=f"{type(error).__name__}: {error}",
            finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        )
        _write_evaluation_status(result_path, summary, results)
        raise
    failures = sum(result["status"] != "finished" for result in results)
    summary.update(
        status="failed" if failures else "finished",
        finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
    )
    _write_evaluation_status(result_path, summary, results)
    print(
        f"[evaluation] {summary['status']} computed={len(summary['summaries'])}/{len(checkpoints)} "
        f"elapsed={time.perf_counter() - started:.1f}s result={result_path}",
        flush=True,
    )
    if failures:
        raise RuntimeError(
            f"evaluation failed for {failures}/{len(checkpoints)} checkpoints; see {result_path}"
        )
    return summary


def run_evaluation(args):
    print("[evaluation] preparing checkpoint list", flush=True)
    if args.eval_run is not None:
        method_directory = args.eval_run.resolve()
        if not method_directory.is_dir():
            raise FileNotFoundError(method_directory)
        if method_directory.parent.name != "train":
            raise ValueError("--eval-run requires <series>/train/<method>")
        checkpoints = [
            path.parent.resolve()
            for path in method_directory.glob("*/checkpoints/global_step_*/state.pt")
            if path.is_file()
        ]
        if not checkpoints:
            raise ValueError(f"no retained checkpoints found in {method_directory}")
    else:
        checkpoints = [path.resolve() for path in args.checkpoint]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("checkpoint list contains duplicate directories")
    methods = {job_plan.checkpoint_stage_directory(path).parent for path in checkpoints}
    if len(methods) != 1:
        raise ValueError("batch evaluation requires checkpoints from the same method run")
    method_directory = next(iter(methods))
    return _schedule_evaluation(args, checkpoints, method_directory)


def _schedule_evaluation(args, checkpoints, method_directory):
    started = time.perf_counter()

    def interrupt_evaluation(signum, frame):
        raise KeyboardInterrupt(f"evaluation interrupted by {signal.Signals(signum).name}")

    with (method_directory / ".evaluation-scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"another evaluation scheduler is running for {method_directory}")
        previous_handler = signal.signal(signal.SIGTERM, interrupt_evaluation)
        try:
            return _run_evaluation_locked(args, checkpoints, method_directory, started)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


def collect_training_result(job, allow_pruned=False):
    output = Path(job.config.training.output_dir)
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    save_training_stage(job, run["config"])
    checkpoint = Path(result["checkpoint"])
    if not (checkpoint / "state.pt").is_file():
        if allow_pruned:
            return None
        raise FileNotFoundError(f"training did not save its reported checkpoint: {checkpoint}")
    return checkpoint


def collect_training_results(jobs, checkpoints, complete=True):
    """仅登记仍存在的阶段 checkpoint，失败时保留尚未写阶段报告的恢复点。"""
    retained = {}
    for job in jobs:
        paths = list(
            Path(job.config.training.output_dir).glob("checkpoints/global_step_*/state.pt")
        )
        if paths:
            retained[job.key] = max(
                (path.parent for path in paths),
                key=lambda path: int(path.name.removeprefix("global_step_")),
            )
    for key, path in list(checkpoints.items()):
        if not (Path(path) / "state.pt").is_file():
            del checkpoints[key]
    for index, job in enumerate(jobs):
        if complete:
            checkpoint = collect_training_result(
                job, allow_pruned=any(later.key in retained for later in jobs[index + 1 :])
            )
        else:
            run_path = Path(job.config.training.output_dir) / "run.json"
            if run_path.is_file():
                run = json.loads(run_path.read_text(encoding="utf-8"))
                save_training_stage(job, run["config"])
            checkpoint = retained.get(job.key)
        if checkpoint is None:
            checkpoints.pop(job.key, None)
        else:
            checkpoints[job.key] = str(checkpoint)


def run_resume(args):
    directory, max_steps, jobs = job_plan.build_resume_jobs(args)
    plan_directory = jobs[0].config_path.parent
    training_directory = Path(jobs[0].config.training.experiment_dir)
    result = read_training_status(training_directory)
    result["checkpoints"][jobs[0].key] = str(args.checkpoint.resolve())
    evaluation_path = plan_directory / "test.json"
    evaluation_config = (
        resolve_evaluation_config(args, evaluation_path)
        if any(job.evaluate for job in jobs)
        else None
    )
    if evaluation_config is not None:
        require_dataset_entries([evaluation_config.dataset_dir])
        checkpoint = args.checkpoint.resolve()
        output = (
            directory / "eval" / plan_directory.name / "test"
            / jobs[0].config.objective.stage / checkpoint.name
        )
        EvaluationHistory(output, asdict(evaluation_config)).validate_configuration()

    environment = execution_environment(args)
    for job in jobs:
        save_training_stage(job, job.config.to_dict())
    if evaluation_config is not None:
        save_evaluation_config(evaluation_path, evaluation_config)
    _save_training_status(training_directory, result, "running")
    try:
        execute(
            training_command(jobs, len(args.gpus.split(",")), max_steps, args.checkpoint),
            environment,
            training_directory / "train.log",
        )
        collect_training_results(jobs, result["checkpoints"])
        evaluation_checkpoints = [
            Path(result["checkpoints"][job.key])
            for job in jobs if job.evaluate and job.key in result["checkpoints"]
        ]
        if evaluation_checkpoints:
            evaluate_checkpoints(args, evaluation_checkpoints, result["summaries"])
        _save_training_status(training_directory, result, "finished")
    except Exception as error:
        collect_training_results(jobs, result["checkpoints"], complete=False)
        _save_training_status(training_directory, result, "failed", error)
        raise
    print(
        f"[finished] method={args.method} mode={args.mode} "
        f"checkpoints={len(result['checkpoints'])} evaluations={len(result['summaries'])} "
        f"plan={plan_directory}",
        flush=True,
    )
    return read_training_status(training_directory)


def run_job(args):
    if args.evaluate_only:
        return run_evaluation(args)
    if args.resume:
        return run_resume(args)
    directory, level, jobs = job_plan.build_jobs(args)
    first = jobs[0]
    plan_directory = first.config_path.parent
    training_directory = Path(first.config.training.experiment_dir)
    gpu_count = len(args.gpus.split(","))
    evaluation_directories = {
        job.key: directory / "eval" / plan_directory.name / "test" / job.config.objective.stage
        for job in jobs
        if job.evaluate
    }
    evaluation_path = plan_directory / "test.json"
    evaluation_config = (
        resolve_evaluation_config(args, evaluation_path, level, first.config.training.seed)
        if evaluation_directories
        else None
    )
    targets = {plan_directory, training_directory}
    targets.update(path.parent for path in evaluation_directories.values())
    for path in sorted(targets):
        if path.exists():
            raise ValueError(f"job output already exists: {path}; refusing to overwrite")
    environment = execution_environment(args)
    require_dataset_entries(
        [job.config.training.dataset_dir for job in jobs]
        + ([evaluation_config.dataset_dir] if evaluation_config is not None else [])
    )
    checkpoints, summaries = {}, []
    result = {
        "status": "pending",
        "checkpoints": checkpoints,
        "summaries": summaries,
    }

    plan_directory.mkdir(parents=True)
    training_directory.mkdir(parents=True)
    if evaluation_config is not None:
        save_evaluation_config(evaluation_path, evaluation_config)
    save_json(
        first.config_path,
        {"seed": first.config.training.seed, "model": asdict(first.config.model)},
    )
    save_json(
        first.train_config_path,
        {
            "execution": {
                "mode": args.mode,
                "max_steps": level["max_steps"],
                "train_only": args.train_only,
            },
            "stages": [
                {
                    "objective": asdict(job.config.objective),
                    "training": {
                        name: value
                        for name, value in asdict(job.config.training).items()
                        if name != "seed"
                    },
                }
                for job in jobs
            ]
        },
    )
    _save_training_status(training_directory, result, "running")
    print(
        f"[start] method={args.method} mode={args.mode} stages={len(jobs)} "
        f"gpus={args.gpus} plan={plan_directory}",
        flush=True,
    )
    try:
        execute(
            training_command(jobs, gpu_count, level["max_steps"]),
            environment,
            training_directory / "train.log",
        )
        collect_training_results(jobs, checkpoints)
        evaluation_checkpoints = [
            Path(checkpoints[job.key]) for job in jobs if job.evaluate and job.key in checkpoints
        ]
        if evaluation_checkpoints:
            evaluate_checkpoints(args, evaluation_checkpoints, summaries)
        _save_training_status(training_directory, result, "finished")
    except Exception as error:
        collect_training_results(jobs, checkpoints, complete=False)
        _save_training_status(training_directory, result, "failed", error)
        raise
    print(
        f"[finished] method={args.method} mode={args.mode} "
        f"checkpoints={len(checkpoints)} evaluations={len(summaries)} plan={plan_directory}",
        flush=True,
    )
    return read_training_status(training_directory)


def main(argv=None):
    args = parse_args(argv)
    run_job(args)


if __name__ == "__main__":
    main()
