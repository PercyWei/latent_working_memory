"""三种 TTT 复现评估共用的进程清理、来源记录与终端进度。"""

from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import unicodedata
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


STAGES = {
    "running": "评估中",
    "preparing": "准备中", "loading": "加载模型", "generating": "生成中",
    "scoring": "评分中", "merging": "汇总中", "completed": "完成", "failed": "失败",
}
BENCHMARK_LABELS = {"ruler": "RULER", "squad_completion": "SQuAD", "swde": "SWDE"}


class WorkerProgress:
    """独立于评估产物的状态快照；只统计当前 worker 的本轮待评样本。"""

    def __init__(self, directory, physical_gpu):
        self.path = Path(directory) / "progress.json"
        self.state = {
            "physical_gpu": str(physical_gpu), "stage": "preparing",
            "benchmark": None, "length": None, "total": 0, "batch_size": None,
            "generated": 0, "completed": 0, "group_started": None,
            "generation_started": None, "error": None,
        }
        self.last_write = float("-inf")
        self._publish(True)

    def _publish(self, force=False):
        now = time.monotonic()
        if force or now - self.last_write >= 1:
            self.state["updated_at"] = timestamp()
            write_json(self.path, self.state)
            self.last_write = now

    def set_stage(self, stage, error=None):
        STAGES[stage]
        changed = self.state["stage"] != stage or self.state["error"] != error
        self.state.update(stage=stage, error=error)
        if stage == "generating" and self.state["generation_started"] is None:
            self.state["generation_started"] = time.monotonic()
        self._publish(changed)

    def start_group(self, benchmark, length, total, batch_size):
        self.state.update(
            stage="preparing", benchmark=benchmark, length=length, total=total,
            batch_size=batch_size, generated=0, completed=0,
            group_started=time.monotonic(), generation_started=None, error=None,
        )
        self._publish(True)

    def advance_generated(self, count):
        self.state["generated"] += count
        self._publish(self.state["generated"] == self.state["total"])

    def advance_completed(self, count):
        self.state["completed"] += count
        self._publish(self.state["completed"] == self.state["total"])


def terminal_line(text, columns):
    """避免多行原地刷新时长行自动折行，中文按终端列宽计算。"""
    result, width = [], 0
    for char in text:
        size = 0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in "WF" else 1)
        if width + size > columns - 2:
            return "".join(result) + "…"
        result.append(char)
        width += size
    return text


class EvaluationProgress:
    """共用展示：总体以评分 JSONL 为准，各卡生成进度来自本轮独立快照。"""

    def __init__(self, parts, started, history_completed, total, workers, method, checkpoint, inference_mode):
        self.parts = [Path(part) for part in parts]
        self.started = started
        self.offsets = [0] * len(parts)
        self.completed = [0] * len(parts)
        self.history_completed, self.total = history_completed, total
        self.workers = workers
        self.method, self.checkpoint, self.inference_mode = method, Path(checkpoint).name, inference_mode
        self.tty = sys.stdout.isatty()
        self.last_print = float("-inf")
        self.header_printed = False
        self.drawn_lines = 0
        self.stage, self.error = "running", None
        self.states, self.rates, self.worker_errors = {}, {}, {}
        self.last_signature = None
        self.last_errors = None

    def refresh(self):
        for index, part in enumerate(self.parts):
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

    def _read_workers(self, now):
        for worker in self.workers:
            gpu = str(worker["physical_gpu"])
            path = Path(worker["directory"]) / "progress.json"
            if not path.is_file():
                continue
            state = json.loads(path.read_text(encoding="utf-8"))
            self.states[gpu] = state
            group = state["group_started"]
            if gpu not in self.rates or self.rates[gpu]["group"] != group:
                self.rates[gpu] = {"group": group, "points": deque(), "generated": 0}
            rate = self.rates[gpu]
            points = rate["points"]
            if state["generation_started"] is not None and not points:
                points.append((state["generation_started"], 0))
            if state["generated"] != rate["generated"]:
                points.append((now, state["generated"]))
                rate["generated"] = state["generated"]
            while len(points) > 3 and points[1][0] < now - 30:
                points.popleft()

    def _rate(self, gpu, state, now):
        points = self.rates[gpu]["points"]
        if state["stage"] != "generating" or len(points) < 3:
            return None
        elapsed = now - points[0][0]
        count = state["generated"] - points[0][1]
        return count / elapsed if elapsed >= 5 and count > 0 else None

    def _worker_line(self, worker, now):
        gpu = str(worker["physical_gpu"])
        state = self.states.get(gpu)
        error = self.worker_errors.get(gpu)
        if state is None:
            return f"GPU {gpu} | {STAGES['failed' if error else 'preparing']}"
        stage = "failed" if error else state["stage"]
        fields = [f"GPU {gpu}"]
        if state["benchmark"] is not None:
            label = BENCHMARK_LABELS[state["benchmark"]]
            if state["length"] is not None:
                length = state["length"]
                label += f"/{length // 1024}K" if length % 1024 == 0 else f"/{length}"
            fields.append(label)
        fields.append(STAGES[stage])
        if state["benchmark"] is not None:
            fields.extend((
                f"生成 {state['generated']}/{state['total']}",
                f"完成 {state['completed']}/{state['total']}", f"B{state['batch_size']}",
            ))
            speed = self._rate(gpu, state, now) if stage != "failed" else None
            fields.append(f"{speed:.2f} 条/s" if speed is not None else "— 条/s")
            eta = (state["total"] - state["generated"]) / speed if speed else None
            fields.append(f"生成剩余 {duration(eta)}" if eta is not None else "生成剩余 —")
        return " | ".join(fields)

    def render(self, force=False):
        now = time.monotonic()
        self._read_workers(now)
        signature = (self.stage, self.error, tuple(
            (str(worker["physical_gpu"]), self.states.get(str(worker["physical_gpu"]), {}).get("stage"),
             self.states.get(str(worker["physical_gpu"]), {}).get("group_started"),
             self.states.get(str(worker["physical_gpu"]), {}).get("error"),
             self.worker_errors.get(str(worker["physical_gpu"]))) for worker in self.workers
        ))
        if not force and signature == self.last_signature and now - self.last_print < (5 if self.tty else 30):
            return
        self.refresh()
        if not self.header_printed:
            devices = ",".join(str(worker["physical_gpu"]) for worker in self.workers) or "—"
            print(f"{self.method} | {self.checkpoint} | {self.inference_mode} | GPU {devices}", flush=True)
            self.header_printed = True
        current = sum(self.completed)
        completed = self.history_completed + current
        percent = completed / self.total if self.total else 1
        lines = [
            f"总体：已完成 {completed}/{self.total}（{percent:.1%}） | 历史 {self.history_completed} + 本轮 {current}"
            f" | {STAGES[self.stage]} | 已耗时 {duration(now - self.started)}",
            *(self._worker_line(worker, now) for worker in self.workers),
        ]
        errors = []
        if self.error:
            errors.append(f"错误：{str(self.error).splitlines()[0]}")
        for worker in self.workers:
            gpu = str(worker["physical_gpu"])
            state = self.states.get(gpu, {})
            error = state.get("error") if state.get("stage") == "failed" else self.worker_errors.get(gpu)
            if error:
                errors.append(f"GPU {gpu} 错误：{str(error).splitlines()[0]} | 日志：{worker['log']}")
        if self.tty:
            if self.drawn_lines:
                print(f"\033[{self.drawn_lines}A", end="")
            columns = shutil.get_terminal_size().columns
            for line in lines:
                print("\r\033[2K" + terminal_line(line, columns))
            self.drawn_lines = len(lines)
            sys.stdout.flush()
        else:
            print("\n".join(lines), flush=True)
        if errors and tuple(errors) != self.last_errors:
            self.finish_line()
            print("\n".join(errors), flush=True)
            self.last_errors = tuple(errors)
        self.last_print, self.last_signature = now, signature

    def set_stage(self, stage, error=None):
        STAGES[stage]
        self.stage, self.error = stage, error
        self.render(force=True)

    def fail_worker(self, physical_gpu, error):
        self.worker_errors[str(physical_gpu)] = error
        self.set_stage("failed", error)

    def finish_line(self):
        # 每次绘制已以换行结束；解除原地刷新后，日志和汇总可正常输出。
        self.drawn_lines = 0
