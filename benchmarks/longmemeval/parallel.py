"""每张可见 GPU 运行一个独立进程，按原始题序合并已落盘的回答。"""

from contextlib import ExitStack
import json
import os
import signal
import subprocess
import sys
import time

from benchmarks.io import write_jsonl


def generate_on_devices(directory, samples, rows, devices, batch_size, lock):
    progress = directory / "progress"
    progress.mkdir(exist_ok=True)
    predictions = directory / "predictions.jsonl"
    prefix_size = len(rows)
    unfinished = range(prefix_size, len(samples))
    pending = [i for i in unfinished if not (progress / f"{i}.json").is_file()]
    if pending and not devices:
        raise ValueError("LongMemEval generation requires a visible GPU")
    print(f"{directory.name} | {len(samples) - len(pending)}/{len(samples)} | {directory}", flush=True)
    workers = []
    with ExitStack() as logs:
        def stop(signum, frame):
            raise SystemExit(128 + signum)

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            for rank, device in enumerate(devices[:len(pending)]):
                indices = pending[rank::len(devices)]
                log_path = progress / f"gpu-{device}.log"
                log = logs.enter_context(log_path.open("a"))
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "benchmarks.longmemeval.worker",
                     "--directory", str(directory), "--batch-size", str(batch_size),
                     "--indices", *map(str, indices)],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
                    stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),),
                )
                workers.append((process, log_path))
            last_report = time.monotonic()
            while workers:
                running = False
                for process, path in workers:
                    status = process.poll()
                    if status is None:
                        running = True
                    elif status != 0:
                        lines = path.read_text(errors="replace").splitlines()
                        reason = lines[-1] if lines else f"exit code {status}"
                        raise RuntimeError(f"LongMemEval worker failed: {reason} (log: {path})")
                if not running:
                    break
                if time.monotonic() - last_report >= 30:
                    completed = prefix_size + sum((progress / f"{i}.json").is_file() for i in unfinished)
                    print(f"{directory.name} | {completed}/{len(samples)}", flush=True)
                    last_report = time.monotonic()
                time.sleep(1)
        finally:
            for process, _ in workers:
                if process.poll() is None:
                    process.terminate()
            for process, _ in workers:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            signal.signal(signal.SIGTERM, previous)
    for i in unfinished:
        path = progress / f"{i}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["question_id"] != samples[i].id or not isinstance(row["hypothesis"], str):
            raise ValueError(f"{path}: invalid prediction for {samples[i].id}")
        rows.append(row)
    if unfinished:
        write_jsonl(predictions, rows)
    # 如果上次在合并后中断，完整预测仍是有效的，续评只需清理分片。
    for path in progress.glob("[0-9]*.json"):
        path.unlink()
