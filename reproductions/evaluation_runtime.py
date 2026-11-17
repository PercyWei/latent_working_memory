"""三种 TTT 复现评估共用的进程清理、来源记录与终端进度。"""

from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def source_record():
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    status = git("status", "--porcelain=v1").splitlines()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(status), "git_status": status}


def stop_processes(processes, timeout=10):
    running = [process for process in processes if process.poll() is None]
    for process in running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while any(process.poll() is None for process in running) and time.monotonic() < deadline:
        time.sleep(0.1)
    for process in running:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def duration(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class EvaluationProgress:
    """只读增量统计各分片已提交的样本，不转发详细模型日志。"""

    def __init__(self, parts, started, history_completed=0, total=None):
        self.parts = parts
        self.started = started
        self.plans = [None] * len(parts)
        self.offsets = [0] * len(parts)
        self.completed = [0] * len(parts)
        self.tty = sys.stdout.isatty()
        self.last_print = float("-inf")
        self.history_completed = history_completed
        self.total = total
        self.refresh()
        self.initial_completed = sum(self.completed)

    def refresh(self):
        for index, part in enumerate(self.parts):
            plan_path = part / "plan.json"
            if self.plans[index] is None and plan_path.is_file():
                self.plans[index] = json.loads(plan_path.read_text())
            predictions = part / "predictions.jsonl"
            if not predictions.is_file():
                continue
            with predictions.open("rb") as stream:
                stream.seek(self.offsets[index])
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        break
                    self.completed[index] += 1
                    self.offsets[index] = stream.tell()

    def render(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_print < (5 if self.tty else 30):
            return
        self.refresh()
        elapsed = now - self.started
        current_completed = sum(self.completed)
        completed = self.history_completed + current_completed
        speed = (current_completed - self.initial_completed) / elapsed if elapsed else 0.0
        total = self.total
        if total is None and all(plan is not None for plan in self.plans):
            total = self.history_completed + sum(plan["assigned_samples"] for plan in self.plans)
        count = (f"{completed}/{total} ({completed / total:.1%})"
                 if total is not None else f"准备中，已完成 {completed}")
        text = (f"[evaluate] {count} | {speed:.2f} samples/s | "
                f"elapsed {duration(elapsed)}")
        if self.tty:
            print("\r" + text, end="\033[K", flush=True)
        else:
            print(text, flush=True)
        self.last_print = now

    def finish_line(self):
        if self.tty:
            print(flush=True)
