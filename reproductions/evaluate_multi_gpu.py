"""Shard LaCT or In-Place TTT evaluation across physical GPUs 0 and 1."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_record():
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    status = git("status", "--porcelain=v1").splitlines()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(status), "git_status": status}


def stop_processes(processes):
    running = [process for process in processes if process.poll() is None]
    for process in running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
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

    def __init__(self, parts, started):
        self.parts = parts
        self.started = started
        self.plans = [None] * len(parts)
        self.offsets = [0] * len(parts)
        self.completed = [0] * len(parts)
        self.tty = sys.stdout.isatty()
        self.last_print = float("-inf")
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
        completed = sum(self.completed)
        speed = (completed - self.initial_completed) / elapsed if elapsed else 0.0
        total = None
        if all(plan is not None for plan in self.plans):
            total = sum(plan["assigned_samples"] for plan in self.plans)
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


def execute(args, worker_args, interruption):
    def check_interruption():
        if interruption["signum"] is not None:
            raise SystemExit(128 + interruption["signum"])

    method_root = ROOT / "reproductions" / args.method
    source = source_record()
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    identity = {
        "method": args.method, "checkpoint": str(args.checkpoint),
        "checkpoint_config": json.loads((args.checkpoint / "config.json").read_text()),
        "devices": devices, "num_shards": len(devices),
        "worker_arguments": worker_args, "python": sys.executable,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    plan_path, execution_path = args.output_dir / "plan.json", args.output_dir / "execution.json"
    if args.resume:
        saved = json.loads(plan_path.read_text())
        if {k: v for k, v in saved.items() if k not in ("created_at", "source")} != identity:
            raise ValueError("Multi-GPU settings differ from the saved plan")
        attempts = json.loads(execution_path.read_text())["attempts"]
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "shards").mkdir()
        (args.output_dir / "logs").mkdir()
        write_json(plan_path, {"created_at": timestamp(), "source": source, **identity})
        attempts = []
    (args.output_dir / "summary.json").unlink(missing_ok=True)
    attempt = {"started_at": timestamp(), "status": "running", "source": source, "workers": []}
    attempts.append(attempt)
    write_json(execution_path, {"attempts": attempts})
    processes, logs = [], []
    merge_process = None
    started = time.monotonic()
    parts = [args.output_dir / "shards" / f"shard-{rank}" for rank in range(len(devices))]
    try:
        progress = EvaluationProgress(parts, started)
        print(f"[evaluate] {args.method} GPUs={','.join(devices)} | logs: {args.output_dir / 'logs'}", flush=True)
        progress.render(force=True)
        for rank in range(len(devices)):
            part = args.output_dir / "shards" / f"shard-{rank}"
            if args.resume and part.exists() and not (part / "plan.json").is_file():
                raise ValueError(f"Shard setup is incomplete; inspect and remove this uninitialized directory before resuming: {part}")
        for rank, device in enumerate(devices):
            check_interruption()
            part = args.output_dir / "shards" / f"shard-{rank}"
            command = [
                "bash", str(method_root / "eval.sh"), "--checkpoint", str(args.checkpoint),
                "--output-dir", str(part), "--num-shards", str(len(devices)),
                "--shard-index", str(rank), *worker_args,
            ]
            if args.resume and part.exists():
                command.append("--resume")
            log_path = args.output_dir / "logs" / f"shard-{rank}.log"
            log = log_path.open("a" if args.resume else "x")
            logs.append(log)
            process = subprocess.Popen(
                command, cwd=ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": device, "WORLD_SIZE": "1"},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            processes.append(process)
            attempt["workers"].append({
                "shard_index": rank, "physical_gpu": device, "pid": process.pid,
                "directory": str(part), "log": str(log_path), "command": command,
                "status": "running",
            })
            check_interruption()
            write_json(execution_path, {"attempts": attempts})
        while True:
            check_interruption()
            changed = False
            for process, worker in zip(processes, attempt["workers"], strict=True):
                code = process.poll()
                if code is not None and worker["status"] == "running":
                    worker.update(
                        exit_code=code, finished_at=timestamp(),
                        status="completed" if code == 0 else "failed",
                    )
                    changed = True
            if changed:
                write_json(execution_path, {"attempts": attempts})
            progress.render()
            if any(worker["status"] == "failed" for worker in attempt["workers"]):
                raise RuntimeError("An evaluation shard failed; see logs and execution.json")
            if all(worker["status"] == "completed" for worker in attempt["workers"]):
                break
            time.sleep(0.2)
        progress.render(force=True)
        progress.finish_line()
        print("[evaluate] 样本评估完成，正在合并结果...", flush=True)
        # The merge is derived from intact raw shards and is rebuilt on every attempt.
        merged = args.output_dir / "merged"
        if merged.exists():
            shutil.rmtree(merged)
        merge_command = [
            sys.executable, str(method_root / "scripts/merge_evaluation.py"),
            "--shard-dirs", *[worker["directory"] for worker in attempt["workers"]],
            "--output-dir", str(merged),
        ]
        with (args.output_dir / "logs/merge.log").open("a" if args.resume else "x") as log:
            merge_process = subprocess.Popen(
                merge_command, cwd=ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": devices[0], "WORLD_SIZE": "1"},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            attempt["merge"] = {"command": merge_command, "pid": merge_process.pid}
            write_json(execution_path, {"attempts": attempts})
            while merge_process.poll() is None:
                check_interruption()
                time.sleep(0.2)
            check_interruption()
            if merge_process.returncode != 0:
                raise RuntimeError("Shard merge failed; see logs/merge.log")
        summary = json.loads((merged / "summary.json").read_text())
        summary.update(
            devices=devices, parallel_execution_path=str(execution_path),
            merged_summary_path=str(merged / "summary.json"),
            pool_wall_seconds=time.monotonic() - started,
        )
        write_json(args.output_dir / "summary.json", summary)
        attempt["status"] = "completed"
        print(f"[evaluate] 完成 | elapsed {duration(time.monotonic() - started)} | summary: {args.output_dir / 'summary.json'}", flush=True)
    except BaseException as error:
        attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
        if sys.stdout.isatty():
            print(flush=True)
        print(f"[evaluate] 评估未完成：{error} | logs: {args.output_dir / 'logs'}", flush=True)
        raise
    finally:
        stop_processes(processes + ([] if merge_process is None else [merge_process]))
        for process, worker in zip(processes, attempt["workers"], strict=True):
            if worker["status"] == "running":
                worker.update(status="terminated", exit_code=process.returncode, finished_at=timestamp())
        for log in logs:
            log.close()
        attempt.update(finished_at=timestamp(), pool_wall_seconds=time.monotonic() - started)
        write_json(execution_path, {"attempts": attempts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("lact", "in_place_ttt"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args, worker_args = parser.parse_known_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0,1", "1,0"):
        parser.error("Select both physical GPUs: CUDA_VISIBLE_DEVICES=0,1 or 1,0")
    if os.environ.get("WORLD_SIZE", "1") != "1" or any(
        name in os.environ for name in ("RANK", "LOCAL_RANK")
    ):
        parser.error("Use one standalone launcher, without torchrun")
    if any(value.split("=", 1)[0] in ("--num-shards", "--shard-index") for value in worker_args):
        parser.error("Multi-GPU evaluation assigns row shards automatically")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    interruption = {"signum": None}
    def terminate(signum, frame):
        interruption["signum"] = signum
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    execute(args, worker_args, interruption)


if __name__ == "__main__":
    main()
