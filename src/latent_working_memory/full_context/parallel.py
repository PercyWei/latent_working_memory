"""在可见 GPU 上按会话或题目分片评估，按原始题序提交完整明细。"""

import json
import signal
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from multiprocessing import get_context
from multiprocessing.connection import wait
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .inference import evaluate_samples


class _Log:
    def __init__(self, console, stream):
        self.console, self.stream = console, stream

    @property
    def encoding(self):
        return self.console.encoding

    def isatty(self):
        return False

    def write(self, value):
        self.stream.write(value)
        return self.console.write(value)

    def flush(self):
        self.stream.flush()
        self.console.flush()


@contextmanager
def evaluation_log(path):
    """将本进程的标准输出和错误同步到终端及共享日志，逐行追加。"""
    with (
        path.open("a", buffering=1, encoding="utf-8") as stream,
        redirect_stdout(_Log(sys.stdout, stream)),
        redirect_stderr(_Log(sys.stderr, stream)),
    ):
        yield


def shard_samples(samples, benchmark, num_devices):
    """LoCoMo 以完整会话分片并均衡题数，RULER 按题目轮流分配。"""
    if benchmark == "locomo":
        conversations = {}
        for sample in samples:
            conversations.setdefault(sample.metadata["conversation_id"], []).append(sample)
        shards = [[] for _ in range(min(num_devices, len(conversations)))]
        for questions in sorted(conversations.values(), key=len, reverse=True):
            min(shards, key=len).extend(questions)
        return shards
    count = min(num_devices, len(samples))
    return [samples[rank::count] for rank in range(count)]


def _worker(rank, samples, snapshot, directory, execution):
    with evaluation_log(directory.parent / "eval.log"):
        try:
            device = torch.device("cuda", rank)
            resources = evaluate_samples(samples, snapshot, device, directory / f"{rank}.jsonl", execution)
            (directory / f"{rank}.json").write_text(json.dumps(resources), encoding="utf-8")
        except Exception as error:
            (directory / f"{rank}.error").write_text(
                f"{type(error).__name__}: {error}", encoding="utf-8"
            )
            traceback.print_exc()
            raise SystemExit(1) from error


def _stop_workers(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        if process.pid is not None:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()


def _run_workers(shards, snapshot, devices, directory, execution):
    context = get_context("spawn")
    processes = []
    stop_signal = None

    def request_stop(signum, _frame):
        # 允许正在执行的 start() 交回进程句柄，以便安全回收。
        nonlocal stop_signal
        stop_signal = signum

    with ExitStack() as cleanup:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(signum, request_stop)
            cleanup.callback(signal.signal, signum, previous)
        cleanup.callback(_stop_workers, processes)
        for rank, samples in enumerate(shards):
            if stop_signal is not None:
                break
            process = context.Process(
                target=_worker,
                args=(rank, samples, snapshot, directory, execution),
            )
            processes.append(process)
            process.start()
        pending = {process.sentinel: rank for rank, process in enumerate(processes)}
        while pending and stop_signal is None:
            for sentinel in wait(pending, timeout=0.5):
                if stop_signal is not None:
                    break
                rank = pending.pop(sentinel)
                process = processes[rank]
                process.join()
                if process.exitcode:
                    error_path = directory / f"{rank}.error"
                    reason = (
                        error_path.read_text(encoding="utf-8")
                        if error_path.exists()
                        else f"process exited with code {process.exitcode}"
                    )
                    raise RuntimeError(f"GPU {devices[rank]} evaluation failed: {reason}")
    if stop_signal == signal.SIGINT:
        raise KeyboardInterrupt
    if stop_signal is not None:
        raise SystemExit(128 + stop_signal)


def evaluate_parallel(samples, snapshot, devices, output, shards, execution):
    """各进程仅接收本卡的题面。控制进程校验并合并所有分片后原子发布明细。"""
    output = Path(output)
    started = time.perf_counter()
    with TemporaryDirectory(dir=output) as temporary:
        directory = Path(temporary)
        _run_workers(shards, snapshot, devices, directory, execution)
        resources = [
            json.loads((directory / f"{rank}.json").read_text(encoding="utf-8"))
            for rank in range(len(devices))
        ]
        by_id = {}
        for rank, assigned in enumerate(shards):
            pending = {sample.id for sample in assigned}
            with (directory / f"{rank}.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    sample_id = record["id"]
                    if sample_id in by_id:
                        raise ValueError(f"duplicate evaluation record: {sample_id}")
                    if sample_id not in pending:
                        raise ValueError(f"unexpected evaluation record from GPU {devices[rank]}: {sample_id}")
                    pending.remove(sample_id)
                    by_id[sample_id] = record
            if pending:
                raise ValueError(f"missing {len(pending)} evaluation records from GPU {devices[rank]}")
        records = [by_id[sample.id] for sample in samples]
        merged = directory / "details.jsonl"
        with merged.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        merged.replace(output / "details.jsonl")
    return records, {
        "evaluation_seconds": max(row["evaluation_seconds"] for row in resources),
        "wall_seconds": time.perf_counter() - started,
        "generation_seconds": sum(row["generation_seconds"] for row in resources),
        "peak_memory_allocated_bytes": max(row["peak_memory_allocated_bytes"] for row in resources),
        "peak_memory_reserved_bytes": max(row["peak_memory_reserved_bytes"] for row in resources),
        "scope": "evaluation_seconds is the longest worker evaluation loop after model loading. "
        "wall_seconds includes startup, model loading, generation and result merging. "
        "generation_seconds sums worker generation times, with shared prefill and batched "
        "generation apportioned across questions. Memory is the maximum single-worker peak.",
    }
