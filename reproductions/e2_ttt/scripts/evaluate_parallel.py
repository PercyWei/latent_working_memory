"""E²-TTT 单／双卡统一调度；按全局样本 ID 续评，可更换 GPU 与 batch。"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from evaluation_state import (
    BENCHMARKS, LENGTHS, PROJECT_ROOT, build_protocol, collect_records,
    evaluation_lock, merge_generation_caches, merge_results, reject_live_workers,
    runtime_versions, sample_index, source_record, timestamp, worker_directories, write_json,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-mode", choices=("original", "optimized"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--benchmarks", choices=BENCHMARKS, nargs="+")
    parser.add_argument("--lengths", choices=LENGTHS, type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("checkpoint", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if not devices or len(devices) != len(set(devices)) or not set(devices) <= {"0", "1"}:
        parser.error("Use CUDA_VISIBLE_DEVICES=0, 1, 0,1 or 1,0")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        parser.error("Use one launcher process")
    args.devices = devices
    return args


def stop_workers(processes):
    active = [process for process in processes if process.poll() is None]
    for process in active:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for process in active:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def resolve_protocol(args, saved):
    defaults = saved if saved is not None else {
        "inference_mode": "original", "benchmarks": list(BENCHMARKS), "ruler_lengths": list(LENGTHS),
    }
    for field, saved_field in (("inference_mode", "inference_mode"), ("benchmarks", "benchmarks"), ("lengths", "ruler_lengths")):
        if getattr(args, field) is None:
            setattr(args, field, defaults[saved_field])
    if not args.benchmarks or len(args.benchmarks) != len(set(args.benchmarks)):
        raise ValueError("Benchmarks must be nonempty and unique")
    if "ruler" in args.benchmarks and (not args.lengths or len(args.lengths) != len(set(args.lengths))):
        raise ValueError("RULER lengths must be nonempty and unique")
    if args.batch_size < 1 or (args.inference_mode == "original" and args.batch_size != 1):
        raise ValueError("Batch size must be positive; original inference requires batch size 1")
    protocol = build_protocol(args.checkpoint, args.inference_mode, args.benchmarks, args.lengths, runtime_versions())
    if saved is not None and protocol != saved:
        different = [key for key in protocol.keys() | saved.keys() if protocol.get(key) != saved.get(key)]
        raise ValueError(f"Frozen E²-TTT evaluation settings differ: {sorted(different)}")
    return protocol


def execute(args):
    if args.resume:
        saved = json.loads((args.output_dir / "evaluation_plan.json").read_text())["protocol"]
    else:
        saved = None
    protocol = resolve_protocol(args, saved)
    source = source_record(__file__, args.inference_mode)
    if not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    with evaluation_lock(args.output_dir):
        reject_live_workers(args.output_dir)
        if saved is None:
            write_json(args.output_dir / "evaluation_plan.json", {
                "created_at": timestamp(), "protocol": protocol, "source": source,
            })
        return execute_attempt(args, protocol, source)


def execute_attempt(args, protocol, source):
    root = args.output_dir
    records, sources = collect_records(root, protocol)
    expected = sample_index(protocol)
    pending = [key for key in expected if key not in records]
    historical_workers = worker_directories(root)
    attempts_root = root / "attempts"
    attempts_root.mkdir(exist_ok=True)
    number = len(list(attempts_root.glob("attempt-*")))
    directory = attempts_root / f"attempt-{number:04d}"
    directory.mkdir()
    (directory / "workers").mkdir()
    assignments = [pending[index::len(args.devices)] for index in range(len(args.devices))]
    write_json(directory / "plan.json", {
        "created_at": timestamp(), "evaluation_plan": str(root / "evaluation_plan.json"),
        "devices": args.devices, "batch_size": args.batch_size,
        "completed_samples_before_attempt": len(records), "pending_samples": len(pending), "source": source,
    })
    state = {"attempt": directory.name, "started_at": timestamp(), "status": "running",
             "devices": args.devices, "batch_size": args.batch_size, "workers": []}
    launcher_path = root / "launcher_execution.json"
    history = json.loads(launcher_path.read_text()) if launcher_path.exists() else {"attempts": []}
    history["attempts"].append(state)

    def save():
        write_json(directory / "execution.json", state)
        write_json(launcher_path, history)

    save()
    write_json(root / "summary.json", {
        "checkpoint": protocol["checkpoint"], "inference_mode": protocol["inference_mode"],
        "completed": False, "samples": len(records), "expected_samples": len(expected),
    })
    processes, logs = [], []
    started = time.monotonic()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for index, (gpu, ids) in enumerate(zip(args.devices, assignments, strict=True)):
            if not ids:
                continue
            folder = directory / "workers" / f"worker-{index}"
            folder.mkdir()
            write_json(folder / "plan.json", {
                "protocol": protocol, "sample_ids": ids, "batch_size": args.batch_size,
                "physical_gpu": gpu, "source": source,
            })
            assignment = directory / f"worker-{index}-sample-ids.json"
            write_json(assignment, ids)
            cache_sources = merge_generation_caches(historical_workers, folder / "generation_cache.sqlite")
            worker = {"directory": str(folder.relative_to(root)), "physical_gpu": gpu,
                      "samples": len(ids), "status": "starting", "cache_sources": cache_sources}
            state["workers"].append(worker)
            save()
            command = [
                sys.executable, "-u", str(METHOD_ROOT / "scripts/evaluate.py"),
                "--protocol-file", str(root / "evaluation_plan.json"),
                "--sample-ids-file", str(assignment), "--output-dir", str(folder),
                "--batch-size", str(args.batch_size),
            ]
            worker["command"] = command
            log = (folder / "evaluation.log").open("x")
            logs.append(log)
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "WORLD_SIZE": "1",
                           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
            environment["PYTHONPATH"] = ":".join((str(METHOD_ROOT), str(METHOD_ROOT / "upstream"),
                                                str(METHOD_ROOT / "upstream/lm-evaluation-harness"),
                                                os.environ.get("PYTHONPATH", "")))
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            worker.update(pid=process.pid, status="running")
            save()
        print(f"E²-TTT: {len(records)}/{len(expected)} complete; {len(pending)} pending; "
              f"GPUs={','.join(args.devices)} batch={args.batch_size}", flush=True)
        active = list(zip(processes, state["workers"], strict=True))
        while active:
            for process, worker in active[:]:
                code = process.poll()
                if code is None:
                    continue
                worker.update(exit_code=code, status="completed" if code == 0 else "failed")
                active.remove((process, worker))
                save()
                if code:
                    raise RuntimeError(f"E²-TTT worker failed: {worker['directory']}; see its evaluation.log")
            if active:
                time.sleep(0.2)
        records, sources = collect_records(root, protocol)
        summary = merge_results(root, protocol, records, sources)
        state["status"] = "completed"
        print(f"E²-TTT complete: {summary['samples']} samples; {root / 'summary.json'}", flush=True)
        return summary
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        stop_workers(processes)
        for process, worker in zip(processes, state["workers"]):
            if worker["status"] == "running":
                worker.update(exit_code=process.returncode, status="interrupted")
        for log in logs:
            log.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        state.update(finished_at=timestamp(), seconds=time.monotonic() - started)
        save()


if __name__ == "__main__":
    execute(parse_args())
