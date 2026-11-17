"""E²-TTT 单／双卡统一调度；每个 benchmark 独立保存结果并按全局 ID 续评。"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from batching import execution_parameters
from evaluation_runtime import EvaluationProgress, stop_processes
from evaluation_state import (
    BENCHMARKS, LENGTHS, PROJECT_ROOT, benchmark_protocol, build_protocol, collect_records,
    evaluation_lock, merge_generation_caches, refresh_benchmark, refresh_run_summary,
    reject_live_workers, runtime_versions, sample_index, source_record, timestamp,
    worker_directories, write_json,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts/reproductions/e2_ttt")
    parser.add_argument("--output-id")
    parser.add_argument("--inference-mode", choices=("original", "optimized"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-size-by-length", nargs="+", default=[], metavar="LENGTH=BATCH")
    parser.add_argument("--benchmarks", choices=BENCHMARKS, nargs="+")
    parser.add_argument("--lengths", choices=LENGTHS, type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.resume and args.output_id is None:
        parser.error("--resume requires the original --output-id")
    if args.output_id is None:
        args.output_id = args.checkpoint.name + "_" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M")
    if (not args.output_id.strip() or args.output_id in (".", "..")
            or "/" in args.output_id or "\\" in args.output_id):
        parser.error("--output-id must be a single nonempty directory name")
    args.run_dir = args.output_root / "eval" / args.output_id
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if not devices or len(devices) != len(set(devices)) or not set(devices) <= {"0", "1"}:
        parser.error("Use CUDA_VISIBLE_DEVICES=0, 1, 0,1 or 1,0")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        parser.error("Use one launcher process")
    args.devices = devices
    return args


def resolve_protocol(args, saved):
    defaults = saved if saved is not None else {
        "inference_mode": "original", "benchmarks": list(BENCHMARKS), "ruler_lengths": list(LENGTHS),
    }
    for field, saved_field in (("inference_mode", "inference_mode"), ("benchmarks", "benchmarks"), ("lengths", "ruler_lengths")):
        if getattr(args, field) is None:
            setattr(args, field, defaults[saved_field])
    if not args.benchmarks or len(args.benchmarks) != len(set(args.benchmarks)):
        raise ValueError("Benchmarks must be nonempty and unique")
    original_benchmarks = saved["benchmarks"] if saved is not None else args.benchmarks
    if not set(args.benchmarks) <= set(original_benchmarks):
        raise ValueError("Resume benchmarks must be a subset of the original evaluation plan")
    if "ruler" in original_benchmarks and (not args.lengths or len(args.lengths) != len(set(args.lengths))):
        raise ValueError("RULER lengths must be nonempty and unique")
    args.batch_parameters = execution_parameters(
        args.batch_size, args.batch_size_by_length,
        args.lengths if "ruler" in original_benchmarks else [], args.benchmarks, args.inference_mode,
    )
    protocol = build_protocol(args.checkpoint, args.inference_mode, original_benchmarks, args.lengths, runtime_versions())
    if saved is not None and protocol != saved:
        different = [key for key in protocol.keys() | saved.keys() if protocol.get(key) != saved.get(key)]
        raise ValueError(f"Frozen E²-TTT evaluation settings differ: {sorted(different)}")
    return protocol


def execute(args):
    root = args.run_dir
    saved = json.loads((root / "evaluation_plan.json").read_text())["protocol"] if args.resume else None
    protocol = resolve_protocol(args, saved)
    source = source_record(__file__, args.inference_mode)
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    with evaluation_lock(root):
        reject_live_workers(root)
        for name in protocol["benchmarks"]:
            child = root / name
            child_protocol = benchmark_protocol(protocol, name)
            if saved is None:
                child.mkdir()
                write_json(child / "evaluation_plan.json", {
                    "created_at": timestamp(), "protocol": child_protocol, "source": source,
                })
                refresh_benchmark(child, child_protocol)
            elif json.loads((child / "evaluation_plan.json").read_text())["protocol"] != child_protocol:
                raise ValueError(f"Benchmark protocol differs: {name}")
        if saved is None:
            write_json(root / "evaluation_plan.json", {
                "created_at": timestamp(), "protocol": protocol, "source": source,
            })
        print(f"E²-TTT output-id={args.output_id}; output-root={args.output_root}", flush=True)
        return execute_attempt(args, protocol, source)


def execute_attempt(args, protocol, source):
    root = args.run_dir
    children = {name: benchmark_protocol(protocol, name) for name in args.benchmarks}
    pending, historical_workers, completed = {}, {}, {}
    for name, child_protocol in children.items():
        records, _ = collect_records(root / name, child_protocol)
        pending[name] = [key for key in sample_index(child_protocol) if key not in records]
        completed[name] = len(records)
        historical_workers[name] = worker_directories(root / name)
    attempts_root = root / "attempts"
    attempts_root.mkdir(exist_ok=True)
    directory = attempts_root / f"attempt-{len(list(attempts_root.glob('attempt-*'))):04d}"
    directory.mkdir()
    (directory / "workers").mkdir()
    assignments = [
        {name: ids[index::len(args.devices)] for name, ids in pending.items() if ids[index::len(args.devices)]}
        for index in range(len(args.devices))
    ]
    write_json(directory / "plan.json", {
        "created_at": timestamp(), "evaluation_plan": str(root / "evaluation_plan.json"),
        "benchmarks": args.benchmarks, "devices": args.devices, **args.batch_parameters,
        "completed_samples_before_attempt": completed,
        "pending_samples": {name: len(ids) for name, ids in pending.items()}, "source": source,
    })
    state = {"attempt": directory.name, "started_at": timestamp(), "status": "running",
             "benchmarks": args.benchmarks, "devices": args.devices, **args.batch_parameters, "workers": []}
    launcher_path = root / "launcher_execution.json"
    history = json.loads(launcher_path.read_text()) if launcher_path.exists() else {"attempts": []}
    history["attempts"].append(state)

    def save():
        write_json(directory / "execution.json", state)
        write_json(launcher_path, history)

    def refresh(names):
        for name in names:
            refresh_benchmark(root / name, children[name])
        return refresh_run_summary(root, protocol)

    save()
    refresh(args.benchmarks)
    processes, logs, observed = [], [], set()
    started = time.monotonic()
    parts = [
        root / name / "attempts" / directory.name / "workers" / f"worker-{index}"
        for index, assigned in enumerate(assignments) for name in assigned
    ]
    workers = [
        {"physical_gpu": gpu, "directory": directory / "workers" / f"worker-{index}",
         "log": directory / "workers" / f"worker-{index}" / "evaluation.log"}
        for index, (gpu, assigned) in enumerate(zip(args.devices, assignments, strict=True)) if assigned
    ]
    progress = EvaluationProgress(
        parts, started, history_completed=sum(completed.values()),
        total=sum(completed[name] + len(ids) for name, ids in pending.items()),
        workers=workers, method="E²-TTT", checkpoint=args.checkpoint.name,
        inference_mode=args.inference_mode,
    )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        progress.render(force=True)
        for index, (gpu, ids_by_benchmark) in enumerate(zip(args.devices, assignments, strict=True)):
            if not ids_by_benchmark:
                continue
            folder = directory / "workers" / f"worker-{index}"
            folder.mkdir()
            outputs = {name: root / name / "attempts" / directory.name / "workers" / f"worker-{index}"
                       for name in ids_by_benchmark}
            write_json(folder / "plan.json", {
                "protocol": protocol, "outputs": {name: str(path) for name, path in outputs.items()},
                **args.batch_parameters, "physical_gpu": gpu, "source": source,
            })
            assignment = directory / f"worker-{index}-sample-ids.json"
            write_json(assignment, ids_by_benchmark)
            cache_sources = {}
            for name, ids in ids_by_benchmark.items():
                child_folder = outputs[name]
                child_folder.mkdir(parents=True)
                write_json(child_folder / "plan.json", {
                    "protocol": children[name], "sample_ids": ids, **args.batch_parameters,
                    "physical_gpu": gpu, "source": source,
                })
                write_json(child_folder / "execution.json", {"status": "pending", "completed_samples": 0})
                cache_sources[name] = merge_generation_caches(
                    historical_workers[name], child_folder / "generation_cache.sqlite")
            worker = {"directory": str(folder.relative_to(root)), "physical_gpu": gpu,
                      "samples": {name: len(ids) for name, ids in ids_by_benchmark.items()},
                      "status": "starting", "cache_sources": cache_sources}
            state["workers"].append(worker)
            save()
            command = [
                sys.executable, "-u", str(METHOD_ROOT / "evaluation/evaluate.py"),
                "--protocol-file", str(root / "evaluation_plan.json"),
                "--sample-ids-file", str(assignment), "--worker-dir", str(folder),
                "--batch-size", str(args.batch_size),
            ]
            worker["command"] = command
            log = (folder / "evaluation.log").open("x")
            logs.append(log)
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "WORLD_SIZE": "1",
                           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
            environment["PYTHONPATH"] = ":".join((str(METHOD_ROOT), str(PROJECT_ROOT / "reproductions"),
                                                str(METHOD_ROOT / "source/e2_ttt"),
                                                str(METHOD_ROOT / "source/e2_ttt/lm-evaluation-harness"),
                                                os.environ.get("PYTHONPATH", "")))
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            worker.update(pid=process.pid, status="running")
            save()
        active = list(zip(processes, state["workers"], strict=True))
        while active:
            for process, worker in active[:]:
                execution = root / worker["directory"] / "execution.json"
                if execution.exists():
                    for name, child in json.loads(execution.read_text())["benchmarks"].items():
                        event = (worker["directory"], name)
                        if child["status"] == "completed" and event not in observed:
                            observed.add(event)
                            refresh([name])
                code = process.poll()
                if code is None:
                    continue
                worker.update(exit_code=code, status="completed" if code == 0 else "failed")
                active.remove((process, worker))
                save()
                if code:
                    progress.fail_worker(worker["physical_gpu"], f"Worker exit code {code}")
                    raise RuntimeError(f"E²-TTT worker failed: {worker['directory']}; see its evaluation.log")
            if active:
                progress.render()
                time.sleep(0.2)
        progress.set_stage("merging")
        progress.finish_line()
        for name in args.benchmarks:
            summary = refresh_benchmark(root / name, children[name])
            if not summary["completed"]:
                raise ValueError(f"Predictions do not cover the planned benchmark: {name}")
        summary = refresh_run_summary(root, protocol)
        state["status"] = "completed"
        progress.set_stage("completed")
        progress.finish_line()
        print(f"E²-TTT selected benchmarks complete; whole run completed={summary['completed']}; {root}", flush=True)
        return summary
    except BaseException as error:
        progress.set_stage("failed", error=repr(error))
        progress.finish_line()
        state.update(status="failed", error=repr(error))
        raise
    finally:
        stop_processes(processes, timeout=5)
        for process, worker in zip(processes, state["workers"]):
            if worker["status"] == "running":
                worker.update(exit_code=process.returncode, status="interrupted")
        for log in logs:
            log.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        try:
            refresh(args.benchmarks)
        finally:
            state.update(finished_at=timestamp(), seconds=time.monotonic() - started)
            save()


if __name__ == "__main__":
    execute(parse_args())
