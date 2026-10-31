"""等待整组 worker；失败或中断先清理全部子进程，再向上抛出。"""

import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys

from .state import write_json


def _stop(processes):
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # leader 可能已退出，其进程组也已清空。
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        finally:
            # leader 退出后，清掉可能仍在同一组中的后代。
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_workers(commands, log_path):
    """命令并发执行，输出同时打印并追加到日志；返回时没有遗留 worker。"""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path.with_suffix(".commands.json"),
        {
            "commands": commands,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    processes = []
    with path.open("ab", buffering=0) as log, selectors.DefaultSelector() as selector:
        try:
            for command in commands:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                processes.append(process)
                selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                for key, _ in selector.select(timeout=0.1):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if block:
                        log.write(block)
                        sys.stdout.buffer.write(block)
                        sys.stdout.buffer.flush()
                    else:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                for process in processes:
                    if process.poll() not in (None, 0):
                        raise subprocess.CalledProcessError(process.returncode, process.args)
            for process in processes:
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, process.args)
        finally:
            _stop(processes)
