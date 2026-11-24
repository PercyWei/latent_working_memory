"""按既定计划为每张可见物理卡串行调度 checkpoint 边界诊断。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shlex
import signal
import subprocess
import sys
import threading
from zoneinfo import ZoneInfo

from ..execution import stop_process
from ..io import write_json


def _now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _read_plan(path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    gpus = plan["gpus"]
    if (not gpus or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
            or len(set(gpus)) != len(gpus)):
        raise ValueError("plan.gpus must be distinct nonnegative physical GPU indices")
    if type(plan["batch_size"]) is not int or plan["batch_size"] < 1:
        raise ValueError("plan.batch_size must be a positive integer")
    jobs = plan["jobs"]
    if not jobs:
        raise ValueError("plan.jobs must not be empty")
    destinations = []
    checkpoints = []
    for job in jobs:
        checkpoint = Path(job["checkpoint"])
        destination = Path(job["output_dir"])
        if not checkpoint.is_absolute() or not checkpoint.is_file():
            raise ValueError(f"checkpoint must be an existing absolute file: {checkpoint}")
        if not destination.is_absolute():
            raise ValueError(f"output_dir must be absolute: {destination}")
        if not isinstance(job["run_id"], str) or not job["run_id"]:
            raise ValueError("each job requires a nonempty run_id")
        destinations.append(destination.resolve())
        checkpoints.append(checkpoint.resolve())
    if len(set(destinations)) != len(jobs) or len(set(checkpoints)) != len(jobs):
        raise ValueError("jobs must have distinct checkpoints and output directories")
    return plan


def _completed(job):
    summary_path = Path(job["output_dir"]) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary["status"] != "complete"
            or Path(summary["checkpoint"]).resolve() != Path(job["checkpoint"]).resolve()
            or set(summary["split_summaries"]) != {"train", "dev"}):
        raise ValueError(f"incomplete or mismatched diagnostic summary: {summary_path}")
    return str(summary_path)


def _initial_state(plan, status_path):
    if status_path.exists():
        state = json.loads(status_path.read_text(encoding="utf-8"))
        if state["plan"] != plan:
            raise ValueError("saved status belongs to a different plan")
        for job, result in zip(plan["jobs"], state["jobs"], strict=True):
            if result["status"] == "finished":
                _completed(job)
            elif result["status"] == "running":
                try:
                    os.kill(result["pid"], 0)
                except ProcessLookupError:
                    result["status"] = "pending"
                else:
                    raise RuntimeError(f"previous diagnostic process is still alive: {result['pid']}")
            elif result["status"] == "stopped":
                result["status"] = "pending"
        return state
    return {
        "plan": plan, "created_at": _now(), "status": "pending",
        "jobs": [{"status": "pending", "attempts": 0} for _ in plan["jobs"]],
    }


def run_plan(plan_path):
    plan_path = Path(plan_path).expanduser().resolve()
    plan = _read_plan(plan_path)
    status_path = plan_path.parent / "status.json"
    logs = plan_path.parent / "logs"
    logs.mkdir(exist_ok=True)
    with (plan_path.parent / "controller.lock").open("a") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = _initial_state(plan, status_path)
        stopping, lock, jobs = threading.Event(), threading.Lock(), Queue()
        state.update(status="running", controller_pid=os.getpid(), resumed_at=_now())
        for index, result in enumerate(state["jobs"]):
            if result["status"] == "pending":
                jobs.put(index)

        def save():
            state["updated_at"] = _now()
            state["progress"] = {
                name: sum(job["status"] == name for job in state["jobs"])
                for name in ("pending", "running", "finished", "failed", "stopped")
            }
            write_json(status_path, state)

        def worker(gpu):
            while not stopping.is_set():
                try:
                    index = jobs.get_nowait()
                except Empty:
                    return
                job, result = plan["jobs"][index], state["jobs"][index]
                command = [
                    sys.executable, "-m",
                    "latent_working_memory.v3_1_1.evaluation.boundary_diagnostic",
                    "--checkpoint", job["checkpoint"], "--output-dir", job["output_dir"],
                    "--batch-size", str(plan["batch_size"]),
                ]
                process = None
                with lock:
                    result["attempts"] += 1
                    log_path = logs / f"job_{index:03d}_attempt_{result['attempts']:02d}.log"
                try:
                    with log_path.open("x", encoding="utf-8") as log:
                        log.write(f"CUDA_VISIBLE_DEVICES={gpu} " + shlex.join(command) + "\n")
                        log.flush()
                        with lock:
                            process = subprocess.Popen(
                                command, cwd=Path.cwd(), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                                     "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4"},
                            )
                            result.pop("error", None)
                            result.update(
                                status="running", gpu=gpu, started_at=_now(), pid=process.pid,
                                command=command, log=str(log_path),
                            )
                            save()
                            print(f"started gpu={gpu} job={index} {job['checkpoint']}", flush=True)
                        while process.poll() is None:
                            if stopping.wait(0.5):
                                stop_process(process)
                                break
                        with lock:
                            result["return_code"] = process.returncode
                        if stopping.is_set():
                            with lock:
                                result["status"] = "stopped"
                        elif process.returncode:
                            raise RuntimeError(f"diagnostic process exited {process.returncode}")
                        else:
                            summary_path = _completed(job)
                            with lock:
                                result.update(status="finished", summary=summary_path)
                except Exception as error:
                    if process is not None:
                        stop_process(process)
                    with lock:
                        result.update(status="failed", error=f"{type(error).__name__}: {error}")
                finally:
                    with lock:
                        result["finished_at"] = _now()
                        save()
                        print(f"{result['status']} gpu={gpu} job={index}", flush=True)
                    jobs.task_done()

        previous = {signum: signal.signal(signum, lambda *_: stopping.set())
                    for signum in (signal.SIGINT, signal.SIGTERM)}
        try:
            save()
            with ThreadPoolExecutor(max_workers=len(plan["gpus"])) as executor:
                futures = [executor.submit(worker, gpu) for gpu in plan["gpus"]]
                for future in futures:
                    future.result()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            state["status"] = (
                "finished" if all(job["status"] == "finished" for job in state["jobs"])
                else "stopped" if stopping.is_set() else "failed"
            )
            state["finished_at"] = _now()
            save()
        return 0 if state["status"] == "finished" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    return run_plan(args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
