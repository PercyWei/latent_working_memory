"""公共 LoCoMo 评估入口；各方法注册加载和会话计算逻辑。"""

import argparse
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

from latent_working_memory.benchmarks.data import load_samples
from latent_working_memory.benchmarks.protocol import data_identity
from latent_working_memory.prompts import QA_SYSTEM_PROMPT
from .engine import conversations, shard_conversations
from .records import read_progress, summarize, write_json
from .registry import METHODS, method_module


ROOT = Path(__file__).resolve().parents[4]


def parse_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--method", choices=METHODS)
    selection, _ = selector.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, parents=[selector])
    parser.add_argument("--benchmark-data", type=Path, default=ROOT / "data/benchmarks/locomo")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/locomo")
    parser.add_argument("--output-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--max-conversations", type=int, help="仅评估前 N 段完整对话，默认全部")
    if selection.method is not None:
        method_module(selection.method).add_arguments(parser)
    args = parser.parse_args(argv)
    if args.method is None:
        parser.error("--method is required")
    if args.batch_size_per_gpu < 1 or (args.max_conversations is not None and args.max_conversations < 1):
        parser.error("batch size and conversation limit must be positive")
    if args.output_id is None:
        if args.resume:
            parser.error("--resume requires the original --output-id")
        current = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M")
        args.output_id = f"{args.method}_{current}"
    if not args.output_id.strip() or args.output_id in {".", ".."} or any(
        character in args.output_id for character in ("/", "\\")
    ):
        parser.error("--output-id must be a single nonempty directory name")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if not all(device.isdigit() for device in devices) or len(devices) != len(set(devices)):
        parser.error("CUDA_VISIBLE_DEVICES must contain distinct GPU indices")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ:
        parser.error("launch directly, without torchrun")
    return args, devices


def main(argv=None):
    args, devices = parse_args(argv)
    samples = load_samples("locomo", args.benchmark_data)
    if args.max_conversations is not None:
        samples = tuple(s for group in conversations(samples)[:args.max_conversations] for s in group)
    plan = {
        "method": args.method, "benchmark": "locomo",
        "settings": method_module(args.method).configuration(args),
        "data": data_identity(samples, args.benchmark_data, "locomo"),
        "max_conversations": args.max_conversations,
        "protocol": {"system_prompt": QA_SYSTEM_PROMPT, "categories": [1, 2, 3, 4], "decoding": "greedy", "max_new_tokens": 50,
                     "scoring": "official LoCoMo F1; question-weighted mean",
                     "sharing": "one immutable context state per conversation; independent question branches"},
    }
    run = args.output_root.expanduser().resolve() / "eval" / args.output_id
    directory = run / "locomo"
    config = directory / "evaluation_config.json"
    if args.resume:
        if not config.is_file():
            raise FileNotFoundError(f"--resume requires {config}")
    else:
        run.mkdir(parents=True, exist_ok=False)
        directory.mkdir()
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
        started = time.monotonic()
        print(f"LoCoMo {args.method} | {len(records)} completed | {len(pending)} pending | {directory}", flush=True)
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
                    sys.executable, "-u", "-m", "latent_working_memory.benchmarks.locomo.worker",
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
                    print(f"LoCoMo {len(completed)}/{len(samples)} | {time.monotonic() - started:.0f}s", flush=True)
                    last_report = time.monotonic()
                time.sleep(1)
            if any(process.returncode != 0 for process in processes):
                raise RuntimeError(f"LoCoMo worker failed; inspect {progress}")
            records = read_progress(directory, samples)
            if len(records) != len(samples):
                raise ValueError("workers did not cover all questions")
            rows = [records[sample.id] for sample in samples]
            summary = summarize(plan, rows)
            temporary = directory / "details.jsonl.tmp"
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
            temporary.replace(directory / "details.jsonl")
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
            execution["wall_seconds"] = time.monotonic() - started
            write_json(directory / "execution.json", execution)


if __name__ == "__main__":
    main()
