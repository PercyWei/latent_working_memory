"""运行独立的 GPU worker，转发中断并保留各卡日志。"""

from contextlib import ExitStack
from datetime import datetime
import os
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


def resolve_generation_run(args, output_name):
    """确定运行名称及可见设备；仅评分的续跑允许没有 GPU。"""
    output_id = args.output_id
    if output_id is None:
        if args.resume:
            raise ValueError("--resume requires the original --output-id")
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M")
        output_id = f"{output_name}_{timestamp}"
    if not output_id.strip() or output_id in {".", ".."} or any(c in output_id for c in "/\\"):
        raise ValueError("--output-id must be a single nonempty directory name")
    if args.batch_size_per_gpu < 1:
        raise ValueError("--batch-size-per-gpu must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    devices = visible.split(",") if visible else []
    if not all(device.isdigit() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain distinct GPU indices")
    if not devices and not args.resume:
        raise ValueError("generation requires a visible GPU")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ:
        raise ValueError("launch directly, without torchrun")
    return output_id, devices


def run_workers(module, directory, devices, shards, batch_size, lock, report_progress):
    workers = []
    with ExitStack() as logs:
        def stop(signum, frame):
            raise SystemExit(128 + signum)

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            for device, indices in zip(devices, shards):
                log_path = directory / "progress" / f"gpu-{device}.log"
                log = logs.enter_context(log_path.open("a"))
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", module,
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
                        raise RuntimeError(f"{module} failed: {reason} (log: {path})")
                if not running:
                    break
                if time.monotonic() - last_report >= 30:
                    report_progress()
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
