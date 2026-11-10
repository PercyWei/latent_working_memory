"""Schedule disjoint native E²-TTT evaluation jobs over selected physical GPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import sys
from threading import Event, RLock
import time
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
METHOD_ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (512, 1024, 2048, 4096, 8192, 16384)
RULER_TASKS = ("niah_single_1", "niah_single_2")
RECALL_COUNTS = {"squad_completion": 2984, "swde": 1111}


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_record(args):
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    status = git("status", "--porcelain=v1").splitlines()
    return {
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(status), "git_status": status,
        "entrypoint": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "inference_mode": args.inference_mode,
        "implementation_path": "reproductions/e2_ttt/" + (
            "optimized" if args.inference_mode == "optimized" else "upstream/e2_ttt"
        ),
    }


def terminate_children(children):
    for process in children:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for process in children:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_job(args, job, gpu, slot, stop, processes, lock, record):
    part = args.output_dir / "parts" / job["id"]
    command = [
        "bash", str(METHOD_ROOT / "eval.sh"), "--checkpoint", str(args.checkpoint),
        "--output-dir", str(part), "--benchmarks", job["benchmark"],
        "--ruler-data", str(args.ruler_data), "--squad-data", str(args.squad_data),
        "--swde-data", str(args.swde_data),
        "--num-shards", str(job["num_shards"]),
        "--shard-index", str(job["shard_index"]),
        "--inference-mode", args.inference_mode, "--batch-size", str(args.batch_size),
    ]
    if job["benchmark"] == "ruler":
        command.extend(("--lengths", str(job["length"])))
    if args.resume and part.exists():
        command.append("--resume")
    state_path = args.output_dir / "jobs" / f"{job['id']}.json"
    state = {
        "job": job["id"], "status": "running", "started_at": timestamp(),
        "physical_gpu": gpu, "slot": slot, "command": command,
    }
    write_json(state_path, state)
    record(state)
    started = time.monotonic()
    process = None
    try:
        log_path = args.output_dir / "jobs" / f"{job['id']}.log"
        with log_path.open("a" if args.resume else "x") as log:
            with lock:
                if stop.is_set():
                    state.update(exit_code=None, status="cancelled")
                    return state
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "WORLD_SIZE": "1"},
                    start_new_session=True,
                )
                processes[process.pid] = process
            process.wait()
        state.update(exit_code=process.returncode, status="completed" if process.returncode == 0 else "failed")
    except Exception as error:
        state.update(exit_code=None, status="failed", error=repr(error))
    finally:
        if process is not None and process.poll() != 0:
            terminate_children([process])
        with lock:
            if process is not None:
                processes.pop(process.pid, None)
        state.update(finished_at=timestamp(), elapsed_seconds=time.monotonic() - started)
        write_json(state_path, state)
        record(state)
    return state


def merge_results(args, jobs):
    # Native metrics are per-document scores in [0,1], aggregated by their mean.
    # Restore original document order before summing; never average shard percentages.
    specs = {
        f"{name}_{length}": {"metric": str(length), "samples": 500}
        for length in args.lengths if "ruler" in args.benchmarks
        for name in RULER_TASKS
    }
    specs.update({name: {"metric": "contains", "samples": count}
                  for name, count in RECALL_COUNTS.items() if name in args.benchmarks})
    scores = {task: {} for task in specs}
    manifest, reference_plan = [], None
    common_fields = (
        "checkpoint", "checkpoint_config", "num_shards", "sharding",
        "ruler_samples_per_task_length", "ruler_generation_tokens",
        "recall_total_token_limit", "recall_input_token_limit", "recall_generation_tokens",
        "recall_truncation", "ruler_truncation", "prompt_format", "inference_mode", "batch_size", "batching",
        "num_fewshot", "dtype", "bootstrap_iters", "seeds", "versions",
    )
    for job in jobs:
        relative = Path("parts") / job["id"]
        folder = args.output_dir / relative
        part = json.loads((folder / "summary.json").read_text())
        plan = json.loads((folder / "plan.json").read_text())
        execution = json.loads((folder / "execution.json").read_text())
        if part["completed"] is not True or part["checkpoint"] != str(args.checkpoint) or execution["exit_code"] != 0:
            raise ValueError(f"Incomplete or mismatched native result: {job['id']}")
        if reference_plan is None:
            reference_plan = plan
        if any(plan[field] != reference_plan[field] for field in common_fields):
            raise ValueError(f"Native evaluation protocols differ: {job['id']}")
        if plan["inference_mode"] != args.inference_mode or plan["batch_size"] != args.batch_size:
            raise ValueError(f"Native inference mode or batch size differs: {job['id']}")
        shards, rank = job["num_shards"], job["shard_index"]
        if (
            plan["num_shards"] != shards or plan["shard_index"] != rank
            or part["num_shards"] != shards or part["shard_index"] != rank
            or plan["cuda_visible_devices"] != args.devices[rank]
        ):
            raise ValueError(f"Native shard assignment differs: {job['id']}")
        benchmark = job["benchmark"]
        source = str(args.ruler_data) if benchmark == "ruler" else str(
            getattr(args, "squad_data" if benchmark == "squad_completion" else "swde_data")
            / "data/validation-00000-of-00001.parquet"
        )
        if plan["benchmarks"] != [benchmark] or plan["data"] != {benchmark: source}:
            raise ValueError(f"Native benchmark or source differs: {job['id']}")
        if set(part["metrics"]) != {benchmark}:
            raise ValueError(f"Unexpected native metrics: {job['id']}")
        if benchmark == "ruler":
            length = str(job["length"])
            if plan["ruler_lengths"] != [job["length"]] or plan["ruler_tasks"] != list(RULER_TASKS):
                raise ValueError(f"Unexpected native RULER plan: {job['id']}")
            if set(part["metrics"]["ruler"]) != {length}:
                raise ValueError(f"Unexpected RULER length: {job['id']}")
            values = part["metrics"]["ruler"][length]
            assigned = len(range(rank, 500, shards))
            if set(values) != set(RULER_TASKS) or any(row["samples"] != assigned for row in values.values()):
                raise ValueError(f"Unexpected RULER sample counts: {job['id']}")
            expected_tasks = {f"{name}_{length}" for name in RULER_TASKS}
        else:
            value = part["metrics"][benchmark]
            if value["samples"] != len(range(rank, RECALL_COUNTS[benchmark], shards)):
                raise ValueError(f"Unexpected recall sample count: {benchmark}")
            expected_tasks = {benchmark}
        seen = {task: set() for task in expected_tasks}
        with (folder / "predictions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                task, source_id = record["task"], record["source_doc_id"]
                if task not in expected_tasks:
                    raise ValueError(f"Unexpected task in native predictions: {job['id']}: {task}")
                spec = specs[task]
                if (
                    type(source_id) is not int or not 0 <= source_id < spec["samples"]
                    or source_id % shards != rank or record["shard_index"] != rank
                    or record["doc"]["source_doc_id"] != source_id
                    or record["filter"] != "none" or record["metrics"] != [spec["metric"]]
                ):
                    raise ValueError(f"Invalid native source document identity: {job['id']}")
                if source_id in scores[task]:
                    raise ValueError(f"Duplicate native prediction: {task}/{source_id}")
                score = record[spec["metric"]]
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Invalid native per-document score: {task}/{source_id}")
                scores[task][source_id] = score
                seen[task].add(source_id)
        for task in expected_tasks:
            if seen[task] != set(range(rank, specs[task]["samples"], shards)):
                raise ValueError(f"Incomplete native shard predictions: {job['id']}: {task}")
        manifest.append({
            "job": job["id"], "directory": str(relative),
            "physical_gpu": args.devices[rank], "num_shards": shards, "shard_index": rank,
            "summary": str(relative / "summary.json"),
            "predictions": str(relative / "predictions.jsonl"),
            "plan": str(relative / "plan.json"), "execution": str(relative / "execution.json"),
        })
    means = {}
    for task, spec in specs.items():
        if scores[task].keys() != set(range(spec["samples"])):
            raise ValueError(f"Predictions do not cover the complete source: {task}")
        means[task] = sum(scores[task][index] for index in range(spec["samples"])) / spec["samples"]
    metrics = {}
    if "ruler" in args.benchmarks:
        metrics["ruler"] = {
            str(length): {
                name: {"samples": 500, "accuracy_percent": 100 * means[f"{name}_{length}"]}
                for name in RULER_TASKS
            }
            for length in args.lengths
        }
    for name in RECALL_COUNTS:
        if name in args.benchmarks:
            metrics[name] = {"samples": RECALL_COUNTS[name], "accuracy_percent": 100 * means[name]}
    count, expected_count = sum(map(len, scores.values())), sum(spec["samples"] for spec in specs.values())
    if count != expected_count:
        raise ValueError(f"Expected {expected_count} samples, got {count}")
    write_json(args.output_dir / "manifest.json", {"parts": manifest})
    write_json(args.output_dir / "summary.json", {
        "checkpoint": str(args.checkpoint), "completed": True, "samples": count,
        "inference_mode": args.inference_mode, "batch_size": args.batch_size,
        "metrics": metrics, "manifest": "manifest.json",
    })


def execute(args):
    source = source_record(args)
    base_jobs = [
        {"id": f"ruler-{length}", "benchmark": "ruler", "length": length}
        for length in sorted(args.lengths, reverse=True)
        if "ruler" in args.benchmarks
    ] + [{"id": name, "benchmark": name} for name in RECALL_COUNTS if name in args.benchmarks]
    jobs = [
        {**job, "id": f"{job['id']}-shard-{rank}",
         "num_shards": len(args.devices), "shard_index": rank}
        for job in base_jobs for rank in range(len(args.devices))
    ]
    gpu_assignments = {job["id"]: args.devices[job["shard_index"]] for job in jobs}
    queues = {gpu: Queue() for gpu in args.devices}
    for job in jobs:
        queues[gpu_assignments[job["id"]]].put(job)
    slots = [
        (gpu, slot)
        for gpu in args.devices
        for slot in range(min(args.workers, queues[gpu].qsize()))
    ]
    identity = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_config": json.loads((args.checkpoint / "config.json").read_text()),
        "data": {name: str(getattr(args, f"{name}_data")) for name in ("ruler", "squad", "swde")},
        "jobs": jobs, "gpu_assignments": gpu_assignments,
        "workers_per_gpu": args.workers, "slots": [list(slot) for slot in slots],
        "inference_mode": args.inference_mode,
        "batch_size_per_job": args.batch_size,
        "physical_gpus": args.devices,
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
    attempt = {
        "started_at": timestamp(), "status": "running", "source": source,
        "physical_gpus": args.devices, "gpu_assignments": gpu_assignments, "jobs": {},
    }
    attempts.append(attempt)
    write_json(execution_path, {"attempts": attempts})
    (args.output_dir / "summary.json").unlink(missing_ok=True)
    started = time.monotonic()
    lock, stop, processes = RLock(), Event(), {}

    def record(state):
        with lock:
            attempt["jobs"][state["job"]] = dict(state)
            write_json(execution_path, {"attempts": attempts})

    def cancel_children():
        stop.set()
        with lock:
            active = list(processes.values())
        terminate_children(active)

    def run_slot(gpu, slot):
        while not stop.is_set():
            try:
                job = queues[gpu].get_nowait()
            except Empty:
                return
            state = run_job(args, job, gpu, slot, stop, processes, lock, record)
            if state["status"] != "completed":
                cancel_children()
                return

    def terminate(signum, frame):
        cancel_children()
        raise SystemExit(128 + signum)

    previous_handlers = {
        signum: signal.signal(signum, terminate) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            pending = [pool.submit(run_slot, gpu, slot) for gpu, slot in slots]
            try:
                for future in as_completed(pending):
                    future.result()
            except BaseException:
                cancel_children()
                raise
        if (
            set(attempt["jobs"]) != {job["id"] for job in jobs}
            or any(state["status"] != "completed" for state in attempt["jobs"].values())
        ):
            raise RuntimeError("Native evaluation jobs failed; see jobs/*.log and execution.json")
        merge_results(args, jobs)
        attempt["status"] = "completed"
    except BaseException as error:
        cancel_children()
        attempt.update(status="failed", error=repr(error))
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        attempt.update(finished_at=timestamp(), pool_wall_seconds=time.monotonic() - started)
        write_json(execution_path, {"attempts": attempts})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-mode", choices=("original", "optimized"), default="original")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="Independent native workers per physical GPU")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmarks", nargs="+", choices=("ruler", "squad_completion", "swde"), default=["ruler", "squad_completion", "swde"])
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    for name, default in (("ruler", "ruler-v1/reproductions/e2-ttt-1.3b"), ("squad", "based-squad"), ("swde", "based-swde")):
        parser.add_argument(f"--{name}-data", type=Path, default=PROJECT_ROOT / "data/benchmarks" / default)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.inference_mode == "original" and args.batch_size != 1):
        parser.error("--batch-size must be positive; original inference requires --batch-size 1")
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if (
        visible not in ("0", "1", "0,1", "1,0")
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
        or "LOCAL_RANK" in os.environ
        or "RANK" in os.environ
    ):
        parser.error("Use one launcher; CUDA_VISIBLE_DEVICES must be 0, 1, 0,1 or 1,0")
    if len(args.benchmarks) != len(set(args.benchmarks)):
        parser.error("--benchmarks must not contain duplicates")
    if len(args.lengths) != len(set(args.lengths)) or any(length <= 128 for length in args.lengths):
        parser.error("--lengths must contain distinct integers greater than 128")
    args.devices = visible.split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    for name in ("checkpoint", "output_dir", "ruler_data", "squad_data", "swde_data"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    execute(parse_args())
