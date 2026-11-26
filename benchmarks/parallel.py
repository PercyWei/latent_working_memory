"""运行独立的 GPU worker，转发中断并保留各卡日志。"""

from contextlib import ExitStack
import os
import signal
import subprocess
import sys
import time


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
