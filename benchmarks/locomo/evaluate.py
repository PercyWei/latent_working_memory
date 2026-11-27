"""LoCoMo 独立评估调度；参数解析和模型加载由调用方法提供。"""

from datetime import datetime
import fcntl
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from benchmarks.data import load_samples
from benchmarks.io import write_jsonl
from benchmarks.output import add_output_arguments, evaluation_directory
from benchmarks.progress import QuestionTimer
from benchmarks.protocol import data_identity
from latent_working_memory.prompts import QA_SYSTEM_PROMPT
from .engine import shard_conversations
from .records import read_progress, summarize, write_json


ROOT = Path(__file__).resolve().parents[2]


def add_evaluation_arguments(parser, output_root, checkpoint_layout=False):
    """向方法自己的 parser 添加公共调度参数。"""
    parser.add_argument("--benchmark-data", type=Path, default=ROOT / "data/benchmarks/locomo")
    add_output_arguments(parser, output_root, checkpoint_layout)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size-per-gpu", type=int, default=1,
                        help="每卡批处理数，默认 1")


def run_evaluation(args, method, settings, load_adapter, output_name, summarize_method=None,
                   checkpoint_name=None):
    """方法提供默认运行名称；worker 加载模型，主进程汇总指标。"""
    if args.batch_size_per_gpu < 1:
        raise ValueError("batch size must be positive")
    directory = evaluation_directory(args, "locomo", output_name, checkpoint_name)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if not all(device.isdigit() for device in devices) or len(devices) != len(set(devices)):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain distinct GPU indices")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ:
        raise ValueError("launch directly, without torchrun")
    samples = load_samples("locomo", args.benchmark_data)
    plan = {
        "method": method, "benchmark": "locomo", "settings": settings,
        "adapter": {"module": load_adapter.__module__, "factory": load_adapter.__name__},
        "data": data_identity(samples, args.benchmark_data, "locomo"),
        "protocol": {"system_prompt": QA_SYSTEM_PROMPT, "categories": [1, 2, 3, 4], "decoding": "greedy", "max_new_tokens": 50,
                     "scoring": "official LoCoMo F1; question-weighted mean",
                     "sharing": "one immutable context state per conversation; independent question branches"},
    }
    config = directory / "evaluation_config.json"
    if args.resume:
        if not config.is_file():
            raise FileNotFoundError(f"--resume requires {config}")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if json.loads(config.read_text()) != plan:
                raise ValueError("saved model, data or evaluation settings differ")
        else:
            write_json(config, plan)
        records = read_progress(directory, samples, repair=True)
        pending = [sample for sample in samples if sample.id not in records]
        shards = shard_conversations(pending, len(devices))
        progress = directory / "progress"
        progress.mkdir(exist_ok=True)
        # 每次启动使用新的文件编号，恢复时统一按题号汇集历史。
        existing = [int(p.stem.split("-")[1]) for p in progress.glob("worker-*.jsonl")]
        offset = max(existing, default=-1) + 1
        work = {"workers": [[] for _ in range(offset)] + [[s.id for s in shard] for shard in shards],
                "batch_size_per_gpu": args.batch_size_per_gpu}
        write_json(directory / "work.json", work)
        processes, logs = [], []
        timer = QuestionTimer(len(records))

        def report_progress(completed):
            elapsed = time.monotonic() - timer.started
            print(f"LoCoMo {completed}/{len(samples)} | {elapsed:.0f}s | {timer.format_rate(completed)}",
                  flush=True)

        print(f"LoCoMo {method} | {len(records)} completed | {len(pending)} pending | {directory}", flush=True)
        source = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        execution = {"git_commit": source, "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                     "devices": devices[:len(shards)], "batch_size_per_gpu": args.batch_size_per_gpu,
                     "status": "running"}
        write_json(directory / "execution.json", execution)
        last_report = time.monotonic()
        previous_handler = signal.getsignal(signal.SIGTERM)
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
        try:
            for index, shard in enumerate(shards):
                rank = offset + index
                log = (progress / f"worker-{rank}.log").open("w")
                logs.append(log)
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": devices[index]}
                # Worker 继承目录锁，主进程意外退出后仍阻止另一轮续评改写任务。
                process = subprocess.Popen([
                    sys.executable, "-u", "-m", "benchmarks.locomo.worker",
                    "--directory", str(directory), "--rank", str(rank),
                ], env=environment, stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),))
                processes.append(process)
            while any(process.poll() is None for process in processes):
                failures = [p.returncode for p in processes if p.poll() not in (None, 0)]
                if failures:
                    raise RuntimeError(f"LoCoMo worker failed; inspect {progress}")
                if time.monotonic() - last_report >= 30:
                    completed = read_progress(directory, samples)
                    report_progress(len(completed))
                    last_report = time.monotonic()
                time.sleep(1)
            if any(process.returncode != 0 for process in processes):
                raise RuntimeError(f"LoCoMo worker failed; inspect {progress}")
            records = read_progress(directory, samples)
            if len(records) != len(samples):
                raise ValueError("workers did not cover all questions")
            if pending:
                report_progress(len(records))
            rows = [records[sample.id] for sample in samples]
            summary = summarize(plan, rows, summarize_method)
            write_jsonl(directory / "details.jsonl", rows)
            write_json(directory / "summary.json", summary)
            execution["status"] = "completed"
            print(f"LoCoMo F1 {100 * summary['metrics']['score']:.2f}", flush=True)
            return summary
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()
            if execution["status"] == "running":
                execution["status"] = "failed"
            execution["wall_seconds"] = time.monotonic() - timer.started
            write_json(directory / "execution.json", execution)
