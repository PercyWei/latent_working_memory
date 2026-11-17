"""主进程统一展示评估进度，完成数来自已提交的样本记录。"""

import json
from pathlib import Path
import shutil
import sys
import time
import unicodedata

from ..io import write_json


STAGES = {
    "pending": "等待启动", "preparing": "准备数据", "loading": "加载模型",
    "running": "评估中", "evaluating": "评估中", "merging": "汇总中",
    "completed": "完成", "finished": "完成", "failed": "失败", "stopped": "已停止",
}
TASKS = {"test": "Test", "locomo": "LoCoMo", "ruler": "RULER"}


def _duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _terminal_line(text, columns):
    """按终端列宽裁剪，避免中文或长行折行破坏原地刷新。"""
    result, width = [], 0
    for char in text:
        size = 0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in "WF" else 1)
        if width + size > columns - 2:
            return "".join(result) + "…"
        result.append(char)
        width += size
    return text


class WorkerProgress:
    """只发布当前计算阶段；样本数量沿用正式 JSONL 的提交记录。"""

    def __init__(self, output_dir, rank):
        self.path = Path(output_dir) / "progress" / f"worker_{rank}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = None
        self.state = None
        self.set_stage("preparing")

    def set_stage(self, stage, batch_size=None, length=None):
        STAGES[stage]
        if stage == "evaluating" and self.started is None:
            self.started = time.monotonic()
        state = {
            "stage": stage, "batch_size": batch_size, "length": length,
            "evaluation_started": self.started,
        }
        if state != self.state:
            write_json(self.path, state)
            self.state = state


class EvaluationProgress:
    """历史结果与本轮新增记录分开计数，单卡和多卡共用展示。"""

    def __init__(self, output_dir, task, step, work, workers):
        self.output_dir = Path(output_dir)
        self.task, self.step, self.workers = TASKS[task], step, workers
        self.total, self.history = work["total_samples"], work["completed_samples"]
        self.started = time.monotonic()
        self.offsets, self.completed = {}, {}
        for worker in workers:
            rank = worker["rank"]
            path = self.output_dir / "progress" / f"worker_{rank}.jsonl"
            self.offsets[rank] = path.stat().st_size if path.exists() else 0
            self.completed[rank] = 0
            path.with_suffix(".json").unlink(missing_ok=True)
        self.stage = "running"
        self.tty = sys.stdout.isatty()
        self.last_print, self.last_signature = float("-inf"), None
        self.header_printed, self.drawn_lines = False, 0
        self.reported_errors = set()

    def _refresh(self):
        for rank in self.completed:
            path = self.output_dir / "progress" / f"worker_{rank}.jsonl"
            if not path.exists():
                continue
            with path.open("rb") as stream:
                stream.seek(self.offsets[rank])
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        break
                    self.completed[rank] += 1
                    self.offsets[rank] = stream.tell()

    def _worker_line(self, worker, state, now):
        status = worker["status"]
        stage = (status if status in {"finished", "failed", "stopped"}
                 else state.get("stage", "preparing" if status == "running" else status))
        completed, total = self.completed[worker["rank"]], worker["sample_count"]
        fields = [f"GPU {worker['gpu']}", STAGES[stage], f"本轮 {completed}/{total}"]
        if state.get("length") is not None:
            length = state["length"]
            fields.append(f"{length // 1024}K" if length % 1024 == 0 else f"长度 {length}")
        if state.get("batch_size") is not None:
            fields.append(f"B{state['batch_size']}")
        if stage == "evaluating":
            elapsed = now - state["evaluation_started"]
            speed = completed / elapsed if completed >= 2 and elapsed >= 5 else None
            fields.append(f"{speed:.2f} 样本/s" if speed else "— 样本/s")
            fields.append(f"剩余约 {_duration((total - completed) / speed)}" if speed else "剩余约 —")
        return " | ".join(fields)

    def render(self, force=False):
        now = time.monotonic()
        workers, states = [dict(worker) for worker in self.workers], []
        for worker in workers:
            path = self.output_dir / "progress" / f"worker_{worker['rank']}.json"
            states.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})
        signature = (self.stage, tuple(
            (worker["status"], worker.get("error"), state.get("stage"), state.get("length"))
            for worker, state in zip(workers, states, strict=True)
        ))
        if not force and signature == self.last_signature and now - self.last_print < (5 if self.tty else 30):
            return
        self._refresh()
        if not self.header_printed:
            gpus = ",".join(worker["gpu"] for worker in workers) or "—"
            print(f"v3.1.1 | {self.task} | step={self.step} | GPU {gpus}\n日志：{self.output_dir / 'eval.log'}",
                  flush=True)
            self.header_printed = True
        current = sum(self.completed.values())
        total_completed = self.history + current
        fraction = total_completed / self.total if self.total else 1.0
        lines = [
            f"总体：{total_completed}/{self.total}（{fraction:.1%}） | 历史 {self.history} + 本轮 {current}"
            f" | {STAGES[self.stage]} | 耗时 {_duration(now - self.started)}",
            *(self._worker_line(worker, state, now)
              for worker, state in zip(workers, states, strict=True)),
        ]
        if self.tty:
            if self.drawn_lines:
                print(f"\033[{self.drawn_lines}A", end="")
            columns = shutil.get_terminal_size().columns
            for line in lines:
                print("\r\033[2K" + _terminal_line(line, columns))
            self.drawn_lines = len(lines)
            sys.stdout.flush()
        else:
            print("\n".join(lines), flush=True)
        for worker in workers:
            error = worker.get("error")
            key = (worker["rank"], error)
            if error and key not in self.reported_errors:
                self.finish_line()
                print(f"GPU {worker['gpu']} 失败：{error.splitlines()[0]} | 日志：{self.output_dir / 'eval.log'}",
                      flush=True)
                self.reported_errors.add(key)
        self.last_print, self.last_signature = now, signature

    def set_stage(self, stage):
        STAGES[stage]
        self.stage = stage
        self.render(force=True)

    def finish_line(self):
        self.drawn_lines = 0
