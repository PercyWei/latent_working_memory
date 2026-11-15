"""训练与评估共用的占用锁、可见 GPU 和子进程管理。"""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import subprocess


@contextmanager
def _lock(path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, mode | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"an experiment is already running: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _lock_directory(run_directory):
    directory = Path(run_directory).expanduser().resolve()
    return directory.parent / ".locks" / directory.name


def run_lock(run_directory, exclusive=False):
    return _lock(_lock_directory(run_directory) / "run.lock",
                 fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)


def evaluation_lock(run_directory, eval_id, task):
    return _lock(_lock_directory(run_directory) / f"eval-{eval_id}" / f"{task}.lock", fcntl.LOCK_EX)


def visible_devices():
    devices = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if any(not value.isdecimal() for value in devices) or len(set(map(int, devices))) != len(devices):
        raise ValueError("set CUDA_VISIBLE_DEVICES to distinct physical GPU indices")
    return [str(int(value)) for value in devices]


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
