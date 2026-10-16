"""GPU 实验入口：连续训练与评估，或从已有 checkpoint 单独补跑评估。"""

import argparse
from dataclasses import asdict, replace
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

from latent_working_memory.v3 import job_plan
from latent_working_memory.v3.config import (
    DYNAMIC_METHODS,
    METHODS,
    WRITER_MODES,
    EvaluationConfig,
    load_evaluation,
    parse_optional_positive_integer,
)
from latent_working_memory.v3.evaluation_records import EvaluationHistory
from latent_working_memory.v3.checkpoint import read_checkpoint
from latent_working_memory.v3.evaluate import (
    publish_cached_evaluations,
    validate_evaluation_publication,
)
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
        help="QA 训练数据目录；首次创建评估配置时默认共用此目录",
    )
    parser.add_argument("--test-data", type=Path, help="独立的最终 QA 测试数据目录")
    parser.add_argument(
        "--seed",
        type=nonnegative_count,
        help="统一随机种子；新训练默认沿用 training.seed，独立评估默认复用 eval.json 或训练 seed",
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
    args = parser.parse_args(argv)
    if args.eval_run is not None and not args.evaluate_only:
        parser.error("--eval-run requires --evaluate-only")
    if args.evaluate_only:
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
        *(str(job.config_path) for job in jobs),
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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def resolve_evaluation_config(args, config_path, default_seed, level=None):
    """沿用独立评估设置，只应用显式评估参数；旧产物须明确指定数据入口。"""
    if config_path.is_file():
        config = load_evaluation(config_path)
    else:
        if level is None and args.test_data is None:
            raise ValueError(f"{config_path} is missing; specify --test-data to create it")
        config = EvaluationConfig(
            dataset_dir=str((args.test_data or args.qa_data).resolve()),
            seed=default_seed,
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
    if args.seed is not None and not args.resume:
        updates["seed"] = args.seed
    if "max_test_samples" in vars(args):
        updates["max_test_samples"] = args.max_test_samples
    return replace(config, **updates)


def save_evaluation_config(config_path, config):
    save_json(config_path, asdict(config))
    plan_path = config_path.parent / "job.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["evaluation_config_path"] = str(config_path)
        save_json(plan_path, plan)


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


def evaluate_checkpoint(args, checkpoint, config_path, environment):
    """自动评估与补评估共用 checkpoint 目录、配置快照和结果登记。"""
    started = time.perf_counter()
    stage_directory, global_step = checkpoint_evaluation_identity(checkpoint)
    method_directory = stage_directory.parent
    directory = method_directory.parent.parent
    output = directory / "eval" / method_directory.name / stage_directory.name / checkpoint.name
    print(
        f"[evaluation] method={method_directory.name} stage={stage_directory.name} "
        f"checkpoint={checkpoint.name} global_step={global_step}",
        flush=True,
    )
    config = load_evaluation(config_path)
    history = EvaluationHistory(output, asdict(config))
    cached = history.cached_summary(checkpoint)
    if cached is not None:
        history.record(cached)
        print(
            f"[evaluation] reusing local result | samples={cached['trajectories']} "
            f"summary={output / 'summary.json'}",
            flush=True,
        )
    if cached is None or args.tracking == "online":
        execute(
            evaluation_command(args, output / "eval.json", checkpoint, output),
            environment,
            output / "eval.log",
            append=cached is not None,
        )
    else:
        print("[evaluation] cached local result ready; tracking disabled", flush=True)
    summary = str(output / "summary.json")
    plan_directory = directory / "plan" / method_directory.name
    result_path = plan_directory / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if summary not in result["summaries"]:
            result["summaries"].append(summary)
        save_json(result_path, result)
    print(
        f"[evaluation] checkpoint={checkpoint.name} finished "
        f"elapsed={time.perf_counter() - started:.1f}s summary={summary}",
        flush=True,
    )
    return summary


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


def _compute_evaluations(args, tasks, outputs, identities, environment, total):
    """每张 GPU 固定一个槽位，仅并行计算，不在子进程发布 SwanLab。"""
    pending = Queue()
    for task in tasks:
        pending.put(task)
    stopping = threading.Event()
    state_lock = threading.Lock()
    processes = set()
    successes, failures = set(), {}

    def worker(gpu):
        while not stopping.is_set():
            try:
                position, checkpoint = pending.get_nowait()
            except Empty:
                return
            output = outputs[checkpoint]
            stage, global_step = identities[checkpoint]
            prefix = (
                f"[evaluation] gpu={gpu} checkpoint={position}/{total} "
                f"source={stage.name}/{checkpoint.name}"
            )
            command = evaluation_command(
                args, output / "eval.json", checkpoint, output, log_to_swanlab=False
            )
            print(f"{prefix} compute started global_step={global_step}", flush=True)
            print(f"{prefix} $ {shlex.join(command)}", flush=True)
            try:
                with (output / "eval.log").open("w", encoding="utf-8") as log:
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
            except Exception as error:
                with state_lock:
                    failures[checkpoint] = error
                    completed = len(successes) + len(failures)
                print(
                    f"{prefix} compute failed completed={completed}/{len(tasks)} "
                    f"error={error} log={output / 'eval.log'}",
                    flush=True,
                )
            else:
                with state_lock:
                    successes.add(checkpoint)
                    completed = len(successes) + len(failures)
                print(
                    f"{prefix} compute finished completed={completed}/{len(tasks)} "
                    f"summary={output / 'summary.json'}",
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
        for signum in (signal.SIGTERM, signal.SIGINT):
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
    return successes, failures


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
    config_path = plan_directory / "eval.json"
    default_seed = args.seed
    if not config_path.is_file() and default_seed is None:
        seeds = {
            json.loads((stage / "run.json").read_text(encoding="utf-8"))["config"]["training"]["seed"]
            for stage, _ in identities.values()
        }
        if len(seeds) != 1:
            raise ValueError(
                "selected checkpoints have different training seeds; specify --seed "
                "to create their shared evaluation configuration"
            )
        default_seed = seeds.pop()
    config = resolve_evaluation_config(args, config_path, default_seed)
    require_dataset_entries([config.dataset_dir])
    training_dir = (
        validate_evaluation_publication(checkpoints) if args.tracking == "online" else None
    )
    outputs = {
        checkpoint: (
            directory / "eval" / method_directory.name / identities[checkpoint][0].name
            / checkpoint.name
        )
        for checkpoint in checkpoints
    }
    successes, tasks = set(), []
    for position, checkpoint in enumerate(checkpoints, 1):
        history = EvaluationHistory(outputs[checkpoint], asdict(config))
        cached = history.cached_summary(checkpoint)
        if cached is None:
            tasks.append((position, checkpoint))
        else:
            history.record(cached)
            successes.add(checkpoint)
            print(
                f"[evaluation] checkpoint={position}/{len(checkpoints)} "
                f"source={identities[checkpoint][0].name}/{checkpoint.name} cache ready "
                f"samples={cached['trajectories']} summary={outputs[checkpoint] / 'summary.json'}",
                flush=True,
            )
    environment = execution_environment(args)
    plan_directory.mkdir(parents=True, exist_ok=True)
    save_evaluation_config(config_path, config)
    print(
        f"[evaluation] computation started cached={len(successes)} pending={len(tasks)} "
        f"gpu_slots={min(len(args.gpus.split(',')), len(tasks))}",
        flush=True,
    )
    computed, compute_failures = _compute_evaluations(
        args, tasks, outputs, identities, environment, len(checkpoints)
    )
    successes.update(computed)
    successful_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint in successes]
    summaries = [str(outputs[checkpoint] / "summary.json") for checkpoint in successful_checkpoints]
    result_path = plan_directory / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for summary in summaries:
            if summary not in result["summaries"]:
                result["summaries"].append(summary)
        save_json(result_path, result)
    print(
        f"[evaluation] computation finished success={len(successes)}/{len(checkpoints)} "
        f"failed={len(compute_failures)}",
        flush=True,
    )
    publication_failures = []
    if training_dir is not None and successful_checkpoints:
        print(
            f"[evaluation] CPU publication started checkpoints={len(successful_checkpoints)}",
            flush=True,
        )
        publication_failures = publish_cached_evaluations(
            successful_checkpoints,
            [outputs[checkpoint] for checkpoint in successful_checkpoints],
            config,
            training_dir,
        )
        for checkpoint, error in publication_failures:
            print(
                f"[evaluation] publication failed source={checkpoint.parent.parent.name}"
                f"/{checkpoint.name} error={error}",
                flush=True,
            )
    status = "failed" if compute_failures or publication_failures else "finished"
    print(
        f"[evaluation] {status} computed={len(successes)}/{len(checkpoints)} "
        f"compute_failed={len(compute_failures)} publication_failed={len(publication_failures)} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    if status == "failed":
        raise RuntimeError(
            f"evaluation completed with failures: compute={len(compute_failures)}, "
            f"publication={len(publication_failures)}; local results and logs are retained"
        )
    return {"status": "finished", "summaries": summaries}


def run_evaluation(args):
    started = time.perf_counter()
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
    result = json.loads((output / "training-result.json").read_text(encoding="utf-8"))
    save_json(job.config_path, json.loads((output / "config.json").read_text(encoding="utf-8")))
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
            config_path = Path(job.config.training.output_dir) / "config.json"
            if config_path.is_file():
                save_json(job.config_path, json.loads(config_path.read_text(encoding="utf-8")))
            checkpoint = retained.get(job.key)
        if checkpoint is None:
            checkpoints.pop(job.key, None)
        else:
            checkpoints[job.key] = str(checkpoint)


def run_resume(args):
    directory, level, jobs = job_plan.build_resume_jobs(args)
    plan_directory = jobs[0].config_path.parent
    result_path = plan_directory / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["checkpoints"][jobs[0].key] = str(args.checkpoint.resolve())
    evaluation_path = plan_directory / "eval.json"
    evaluation_config = (
        resolve_evaluation_config(args, evaluation_path, jobs[0].config.training.seed)
        if any(job.evaluate for job in jobs)
        else None
    )
    if evaluation_config is not None:
        require_dataset_entries([evaluation_config.dataset_dir])
        checkpoint = args.checkpoint.resolve()
        output = directory / "eval" / plan_directory.name / jobs[0].config.objective.stage / checkpoint.name
        EvaluationHistory(output, asdict(evaluation_config)).validate_configuration()

    def save_result(status, error=None):
        result["status"] = status
        result.pop("error", None)
        if error is not None:
            result["error"] = str(error)
        save_json(result_path, result)

    environment = execution_environment(args)
    evaluation_environment = dict(environment, CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0])
    for job in jobs:
        save_json(job.config_path, job.config.to_dict())
    if evaluation_config is not None:
        save_evaluation_config(evaluation_path, evaluation_config)
    save_result("running")
    try:
        execute(
            training_command(jobs, len(args.gpus.split(",")), level["max_steps"], args.checkpoint),
            environment,
            plan_directory / "train.log",
        )
        collect_training_results(jobs, result["checkpoints"])
        for job in jobs:
            checkpoint = result["checkpoints"].get(job.key)
            if job.evaluate and checkpoint is not None:
                summary = evaluate_checkpoint(
                    args, Path(checkpoint), evaluation_path, evaluation_environment
                )
                if summary not in result["summaries"]:
                    result["summaries"].append(summary)
        save_result("finished")
    except Exception as error:
        collect_training_results(jobs, result["checkpoints"], complete=False)
        save_result("failed", error)
        raise
    print(
        f"[finished] method={args.method} mode={args.mode} "
        f"checkpoints={len(result['checkpoints'])} evaluations={len(result['summaries'])} "
        f"plan={plan_directory}",
        flush=True,
    )
    return result


def run_job(args):
    if args.evaluate_only:
        return run_evaluation(args)
    if args.resume:
        return run_resume(args)
    directory, level, jobs = job_plan.build_jobs(args)
    first = jobs[0]
    plan_directory = first.config_path.parent
    gpu_count = len(args.gpus.split(","))
    evaluation_directories = {
        job.key: directory / "eval" / plan_directory.name / job.config.objective.stage
        for job in jobs
        if job.evaluate
    }
    evaluation_path = plan_directory / "eval.json"
    evaluation_config = (
        resolve_evaluation_config(args, evaluation_path, first.config.training.seed, level)
        if evaluation_directories
        else None
    )
    training_run = {
        "stages": [job.key for job in jobs],
        "command": training_command(jobs, gpu_count, level["max_steps"]),
        "log_path": str(plan_directory / "train.log"),
    }
    plan = {
        "mode": args.mode,
        "method": args.method,
        "experiment_id": args.run_id,
        "directory": str(directory),
        "plan_directory": str(plan_directory),
        "gpus": args.gpus,
        "limits": level,
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
        "training_runs": [training_run],
        "evaluation_config_path": str(evaluation_path) if evaluation_config is not None else None,
    }
    targets = {plan_directory, Path(first.config.training.experiment_dir)}
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
        "mode": args.mode,
        "experiment_id": args.run_id,
        "checkpoints": checkpoints,
        "summaries": summaries,
    }

    def save_result(status, error=None):
        result["status"] = status
        if error is not None:
            result["error"] = str(error)
        save_json(plan_directory / "result.json", result)

    plan_directory.mkdir(parents=True)
    save_json(plan_directory / "job.json", plan)
    if evaluation_config is not None:
        save_json(evaluation_path, asdict(evaluation_config))
    for job in jobs:
        save_json(job.config_path, job.config.to_dict())
    save_result("running")
    print(
        f"[start] method={args.method} mode={args.mode} stages={len(jobs)} "
        f"gpus={args.gpus} plan={plan_directory}",
        flush=True,
    )
    evaluation_environment = dict(environment, CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0])
    try:
        execute(training_run["command"], environment, Path(training_run["log_path"]))
        collect_training_results(jobs, checkpoints)
        for job in jobs:
            checkpoint = checkpoints.get(job.key)
            if job.evaluate and checkpoint is not None:
                summary = evaluate_checkpoint(
                    args, Path(checkpoint), evaluation_path, evaluation_environment
                )
                summaries.append(summary)
        save_result("finished")
    except Exception as error:
        collect_training_results(jobs, checkpoints, complete=False)
        save_result("failed", error)
        raise
    print(
        f"[finished] method={args.method} mode={args.mode} "
        f"checkpoints={len(checkpoints)} evaluations={len(summaries)} plan={plan_directory}",
        flush=True,
    )
    return result


def main(argv=None):
    args = parse_args(argv)
    run_job(args)


if __name__ == "__main__":
    main()
