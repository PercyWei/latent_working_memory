"""Measure independent batch-one evaluation workers sharing physical GPU 0."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
RULER = {
    "lact": ("lact-760m", ("niah_single_1", "niah_multiquery")),
    "e2_ttt": ("e2-ttt-1.3b", ("niah_single_1", "niah_single_2")),
    "in_place_ttt": ("in-place-ttt-qwen3-4b", ("niah_single_1", "niah_multiquery")),
}


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_record():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = git("status", "--porcelain=v1").splitlines()
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": status, "git_dirty": bool(status)}


def samples_for(args):
    if args.dataset == "ruler":
        directory, tasks = RULER[args.method]
        return [{
            "id": f"ruler/{args.length}/{tasks[i % 2]}/{1 + i // 2}",
            "source": str(ROOT / "data/benchmarks/ruler-v1/reproductions" / directory / str(args.length) / tasks[i % 2] / "test.jsonl"),
            "row_index": 1 + i // 2,
        } for i in range(args.num_samples)]
    count, directory = (2984, "based-squad") if args.dataset == "squad_completion" else (1111, "based-swde")
    return [{
        "id": f"{args.dataset}/{round(i * (count - 1) / (args.num_samples - 1))}",
        "source": str(ROOT / "data/benchmarks" / directory / "data/validation-00000-of-00001.parquet"),
        "row_index": round(i * (count - 1) / (args.num_samples - 1)),
    } for i in range(args.num_samples)]


def device_sample():
    output = subprocess.run([
        "nvidia-smi", "--id=0", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True).stdout.strip()
    memory, utilization = map(int, output.split(","))
    return {"monotonic": time.monotonic(), "memory_used_mib": memory, "utilization_percent": utilization}


def run_trial(plan, env, python, folder):
    folder.mkdir()
    plan = {**plan, "output_dir": str(folder)}
    write_json(folder / "plan.json", plan)
    processes, logs, telemetry = [], [], []
    started, measurement_start = time.monotonic(), None
    error, status = None, "starting"
    try:
        for rank in range(plan["workers"]):
            log = (folder / f"worker-{rank}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen([
                str(python), "-u", str(ROOT / "reproductions/tuning/worker.py"),
                "--plan", str(folder / "plan.json"), "--rank", str(rank),
            ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        deadline = started + 1800
        next_sample = started
        while True:
            now = time.monotonic()
            if now >= next_sample:
                telemetry.append({**device_sample(), "phase": status})
                next_sample = now + 0.2
            failures = list(folder.glob("error-*.json"))
            if failures:
                error = [json.loads(path.read_text()) for path in failures]
                break
            if any(process.poll() not in (None, 0) for process in processes):
                error = "Worker exited without a successful result; inspect worker logs"
                break
            if measurement_start is None and all((folder / f"ready-{rank}.json").exists() for rank in range(plan["workers"])):
                measurement_start = time.monotonic()
                write_json(folder / "go.json", {"created_at": timestamp()})
                status = "measuring"
                print(json.dumps({"workers": plan["workers"], "status": status}), flush=True)
            if all((folder / f"worker-{rank}.json").exists() for rank in range(plan["workers"])):
                status = "completed"
                break
            if now > deadline:
                error = "Trial exceeded its 30-minute limit"
                break
            time.sleep(0.02)
        finished = time.monotonic()
        results = [json.loads(path.read_text()) for path in sorted(folder.glob("worker-*.json"))]
        measured = [sample for row in results for sample in row["samples"]]
        summary = {
            "workers": plan["workers"], "batch_size_per_worker": 1,
            "status": status if error is None else "failed", "error": error,
            "elapsed_including_load_warmup_seconds": finished - started,
            "measurement_wall_seconds": None if measurement_start is None else finished - measurement_start,
            "samples": measured,
            "peak_device_memory_mib": max(row["memory_used_mib"] for row in telemetry),
            "peak_worker_allocated_bytes": max((row["peak_cuda_allocated_bytes"] for row in results), default=None),
        }
        if error is None:
            if len(measured) != len(plan["samples"]):
                raise ValueError("Measured sample count differs from the planned corpus")
            summary["samples_per_second"] = len(measured) / summary["measurement_wall_seconds"]
            summary["generated_tokens_per_second"] = sum(row["generated_tokens"] for row in measured) / summary["measurement_wall_seconds"]
        write_json(folder / "telemetry.json", telemetry)
        write_json(folder / "summary.json", summary)
        return summary
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(RULER), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ruler", "squad_completion", "swde"), default="ruler")
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--inference-mode", choices=("original", "incremental", "optimized"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.num_samples < 2 or min(args.workers) < 1 or max(args.workers) > args.num_samples or len(set(args.workers)) != len(args.workers):
        parser.error("Use distinct positive workers and at least two samples, with samples >= workers")
    if args.dataset != "ruler" and args.method != "e2_ttt":
        parser.error("Recall benchmarks apply only to E2-TTT")
    inference_mode = args.inference_mode or ("incremental" if args.method == "lact" else "original")
    allowed_modes = {"lact": ("original", "incremental"), "e2_ttt": ("original",), "in_place_ttt": ("original", "optimized")}
    if inference_mode not in allowed_modes[args.method]:
        parser.error("Inference mode does not apply to this method")
    args.checkpoint, args.output_dir = args.checkpoint.expanduser().resolve(), args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    method_root = ROOT / "reproductions" / args.method
    paths = [method_root / "scripts", method_root / "upstream", method_root]
    paths += [ROOT / "src"] if args.method == "lact" else []
    if args.method == "lact":
        paths.append(method_root / "upstream/lact_llm")
    if args.method == "e2_ttt":
        paths.append(method_root / "upstream/lm-evaluation-harness")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": os.pathsep.join(map(str, paths))}
    if args.method == "in_place_ttt":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    python = Path.home() / "rivermind-data/envs" / f"ttt-{args.method}" / "bin/python"
    plan = {
        "created_at": timestamp(), "source": source_record(), "method": args.method,
        "checkpoint": str(args.checkpoint), "checkpoint_config": json.loads((args.checkpoint / "config.json").read_text()),
        "dataset": args.dataset, "length": args.length, "samples": samples_for(args),
        "inference_mode": inference_mode,
        "mlp_chunk_size": 0, "workers_to_test": args.workers,
        "cuda_visible_devices": "0", "batch_size_per_worker": 1,
        "worker_python": str(python), "omp_num_threads": 1, "mkl_num_threads": 1,
    }
    write_json(args.output_dir / "plan.json", plan)
    summaries, references = [], None
    for workers in args.workers:
        trial = {**plan, "workers": workers, "assignments": [list(range(rank, len(plan["samples"]), workers)) for rank in range(workers)]}
        print(json.dumps({"workers": workers, "status": "starting", "method": args.method, "length": args.length}), flush=True)
        result = run_trial(trial, env, python, args.output_dir / f"w{workers}")
        if result["status"] == "completed":
            values = {row["id"]: row["token_ids"] for row in result["samples"]}
            if workers == 1:
                references = values
            result["output_ids_equal_to_first_trial"] = None if references is None else values == references
        summaries.append(result)
        write_json(args.output_dir / "summary.json", summaries)
        print(json.dumps({key: value for key, value in result.items() if key != "samples"}), flush=True)


if __name__ == "__main__":
    main()
