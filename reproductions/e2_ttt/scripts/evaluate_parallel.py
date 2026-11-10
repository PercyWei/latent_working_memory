"""Schedule eight disjoint native E²-TTT evaluation jobs on one GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
METHOD_ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (512, 1024, 2048, 4096, 8192, 16384)
RULER_TASKS = ("niah_single_1", "niah_single_2")
RECALL_COUNTS = {"squad_completion": 2984, "swde": 1111}
JOBS = [
    {"id": f"ruler-{length}", "benchmark": "ruler", "length": length}
    for length in reversed(LENGTHS)
] + [{"id": name, "benchmark": name} for name in RECALL_COUNTS]


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_record():
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    status = git("status", "--porcelain=v1").splitlines()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(status), "git_status": status}


def run_job(args, job):
    part = args.output_dir / "parts" / job["id"]
    command = [
        "bash", str(METHOD_ROOT / "eval.sh"), "--checkpoint", str(args.checkpoint),
        "--output-dir", str(part), "--benchmarks", job["benchmark"],
        "--ruler-data", str(args.ruler_data), "--squad-data", str(args.squad_data),
        "--swde-data", str(args.swde_data),
    ]
    if job["benchmark"] == "ruler":
        command.extend(("--lengths", str(job["length"])))
    if args.resume and part.exists():
        command.append("--resume")
    state_path = args.output_dir / "jobs" / f"{job['id']}.json"
    state = {"job": job["id"], "status": "running", "started_at": timestamp(), "command": command}
    write_json(state_path, state)
    started = time.monotonic()
    try:
        log_path = args.output_dir / "jobs" / f"{job['id']}.log"
        with log_path.open("a" if args.resume else "x") as log:
            process = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        state.update(exit_code=process.returncode, status="completed" if process.returncode == 0 else "failed")
    except Exception as error:
        state.update(exit_code=None, status="failed", error=repr(error))
    state.update(finished_at=timestamp(), elapsed_seconds=time.monotonic() - started)
    write_json(state_path, state)
    return state


def merge_results(args):
    metrics = {"ruler": {}}
    manifest = []
    count = 0
    for job in JOBS:
        relative = Path("parts") / job["id"]
        part = json.loads((args.output_dir / relative / "summary.json").read_text())
        if part["completed"] is not True or part["checkpoint"] != str(args.checkpoint):
            raise ValueError(f"Incomplete or mismatched native result: {job['id']}")
        benchmark = job["benchmark"]
        if set(part["metrics"]) != {benchmark}:
            raise ValueError(f"Unexpected native metrics: {job['id']}")
        if benchmark == "ruler":
            length = str(job["length"])
            if set(part["metrics"]["ruler"]) != {length}:
                raise ValueError(f"Unexpected RULER length: {job['id']}")
            values = part["metrics"]["ruler"][length]
            if set(values) != set(RULER_TASKS) or any(row["samples"] != 500 for row in values.values()):
                raise ValueError(f"Unexpected RULER sample counts: {job['id']}")
            metrics["ruler"][length] = values
            count += sum(row["samples"] for row in values.values())
        else:
            value = part["metrics"][benchmark]
            if value["samples"] != RECALL_COUNTS[benchmark]:
                raise ValueError(f"Unexpected recall sample count: {benchmark}")
            metrics[benchmark] = value
            count += value["samples"]
        manifest.append({
            "job": job["id"], "directory": str(relative),
            "summary": str(relative / "summary.json"),
            "predictions": str(relative / "predictions.jsonl"),
            "plan": str(relative / "plan.json"), "execution": str(relative / "execution.json"),
        })
    if count != 10095:
        raise ValueError(f"Expected 10095 samples, got {count}")
    write_json(args.output_dir / "manifest.json", {"parts": manifest})
    write_json(args.output_dir / "summary.json", {
        "checkpoint": str(args.checkpoint), "completed": True, "samples": count,
        "metrics": metrics, "manifest": "manifest.json",
    })


def execute(args):
    source = source_record()
    identity = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_config": json.loads((args.checkpoint / "config.json").read_text()),
        "data": {name: str(getattr(args, f"{name}_data")) for name in ("ruler", "squad", "swde")},
        "jobs": JOBS, "workers": args.workers, "batch_size_per_job": 1,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "python": sys.executable, "omp_num_threads": "1", "mkl_num_threads": "1",
    }
    plan_path = args.output_dir / "plan.json"
    if args.resume:
        saved = json.loads(plan_path.read_text())
        if {key: value for key, value in saved.items() if key not in ("created_at", "source")} != identity:
            raise ValueError("Launcher settings differ from the saved plan")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "parts").mkdir()
        (args.output_dir / "jobs").mkdir()
        write_json(plan_path, {"created_at": timestamp(), "source": source, **identity})
    execution_path = args.output_dir / "execution.json"
    attempts = json.loads(execution_path.read_text())["attempts"] if args.resume and execution_path.exists() else []
    attempt = {"started_at": timestamp(), "status": "running", "source": source, "jobs": {}}
    attempts.append(attempt)
    write_json(execution_path, {"attempts": attempts})
    (args.output_dir / "summary.json").unlink(missing_ok=True)
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(run_job, args, job): job for job in JOBS}
            for future in as_completed(pending):
                state = future.result()
                attempt["jobs"][state["job"]] = state
                write_json(execution_path, {"attempts": attempts})
        if any(state["status"] != "completed" for state in attempt["jobs"].values()):
            raise RuntimeError("Native evaluation jobs failed; see jobs/*.log and execution.json")
        merge_results(args)
        attempt["status"] = "completed"
    except Exception as error:
        attempt.update(status="failed", error=repr(error))
        raise
    finally:
        attempt.update(finished_at=timestamp(), pool_wall_seconds=time.monotonic() - started)
        write_json(execution_path, {"attempts": attempts})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    for name, default in (("ruler", "ruler-v1/reproductions/e2-ttt-1.3b"), ("squad", "based-squad"), ("swde", "based-swde")):
        parser.add_argument(f"--{name}-data", type=Path, default=PROJECT_ROOT / "data/benchmarks" / default)
    args = parser.parse_args()
    if not 1 <= args.workers <= len(JOBS):
        parser.error("--workers must be between 1 and 8")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1") or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one process launcher and one physical GPU: CUDA_VISIBLE_DEVICES=0 or 1")
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    for name in ("checkpoint", "output_dir", "ruler_data", "squad_data", "swde_data"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    execute(parse_args())
