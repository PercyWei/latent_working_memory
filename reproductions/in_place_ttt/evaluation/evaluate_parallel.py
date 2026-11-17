"""In-Place TTT 的单卡、多卡评估及跨设备、batch 设置续评。"""

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from evaluation_runtime import (
    EvaluationProgress,
    duration,
    source_record,
    stop_processes,
    timestamp,
    write_json,
)
from in_place_ttt.evaluation import records as state
from in_place_ttt.evaluation.batching import execution_parameters


ROOT = Path(__file__).resolve().parents[3]


@contextmanager
def evaluation_lock(directory):
    with (directory / ".evaluation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another In-Place TTT evaluation holds the output lock: {directory}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def reject_live_workers(directory):
    path = directory / "launcher_execution.json"
    if not path.exists():
        return
    for attempt in json.loads(path.read_text())["attempts"]:
        for worker in attempt["workers"]:
            if worker["status"] != "running":
                continue
            try:
                os.kill(worker["pid"], 0)
            except ProcessLookupError:
                continue
            raise RuntimeError(
                f"A recorded In-Place TTT worker is still alive: PID {worker['pid']} ({path})"
            )


def execute(args, interruption):
    def check_interruption():
        if interruption["signum"] is not None:
            raise SystemExit(128 + interruption["signum"])

    directory = args.output_dir
    run_directory = directory.parent
    if args.resume:
        if not directory.is_dir():
            raise ValueError(f"--resume requires the existing RULER directory: {directory}")
    else:
        run_directory.mkdir(parents=True, exist_ok=False)
        directory.mkdir()

    with evaluation_lock(run_directory):
        reject_live_workers(directory)
        plan_path = directory / "evaluation_plan.json"
        saved = json.loads(plan_path.read_text())["protocol"] if args.resume else None
        defaults = saved if saved is not None else {
            "inference_mode": "original", "mlp_chunk_size": 0, "lengths": list(state.LENGTHS),
        }
        for field in ("inference_mode", "mlp_chunk_size", "lengths"):
            if getattr(args, field) is None:
                setattr(args, field, defaults[field])
        if args.mlp_chunk_size < 0:
            raise ValueError("--mlp-chunk-size must be nonnegative; 0 disables MLP chunking")
        if args.inference_mode == "original" and args.mlp_chunk_size != 0:
            raise ValueError("original inference requires --mlp-chunk-size 0")
        if not args.lengths or len(set(args.lengths)) != len(args.lengths):
            raise ValueError("--lengths must be nonempty and unique")
        protocol = state.build_protocol(
            args.checkpoint, args.lengths, args.inference_mode, args.mlp_chunk_size
        )
        if saved is not None:
            # 模块位置属于来源说明；续评保留原计划，科学设置仍逐项核对。
            fields = (saved.keys() | protocol.keys()) - {"model_wrapper", "generation_path"}
            changed = [
                field for field in sorted(fields)
                if field not in saved or field not in protocol or saved[field] != protocol[field]
            ]
            if changed:
                raise ValueError(f"Frozen In-Place TTT evaluation settings differ: {', '.join(changed)}")
            protocol = saved
        batch_parameters = execution_parameters(
            args.batch_size, args.batch_size_by_length, args.lengths, args.inference_mode
        )
        expected = state.load_samples(protocol)
        records, sources = state.collect_records(directory, protocol, expected)
        pending = [sample_id for sample_id in expected if sample_id not in records]
        check_interruption()

        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        parameters = {
            "devices": devices, "num_shards": len(devices), **batch_parameters,
            "python": sys.executable,
            "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
            "pytorch_cuda_alloc_conf": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        }
        numbers = [int(path.name.split("-")[1]) for path in directory.glob("attempts/attempt-*")]
        number = max(numbers, default=-1) + 1
        attempt_dir = directory / "attempts" / f"attempt-{number:04d}"
        for name in ("logs", "assignments", "shards"):
            (attempt_dir / name).mkdir(parents=True, exist_ok=False)
        source = source_record()
        if not args.resume:
            write_json(plan_path, {"created_at": timestamp(), "source": source, "protocol": protocol})
        write_json(attempt_dir / "plan.json", {
            "protocol": protocol, "created_at": timestamp(), "source": source,
            "execution_parameters": parameters,
            "history_completed": len(records), "assigned_samples": len(pending),
        })
        execution_path = directory / "launcher_execution.json"
        attempts = json.loads(execution_path.read_text())["attempts"] if args.resume else []
        attempt = {
            "execution_attempt": number, "started_at": timestamp(), "status": "running",
            "source": source, "plan_path": str(attempt_dir / "plan.json"),
            "execution_parameters": parameters, "history_completed": len(records),
            "assigned_samples": len(pending), "workers": [],
        }
        attempts.append(attempt)

        def save_execution():
            write_json(attempt_dir / "execution.json", attempt)
            write_json(execution_path, {"attempts": attempts})

        save_execution()
        (directory / "summary.json").unlink(missing_ok=True)
        processes, logs = [], []
        started = time.monotonic()
        assignments = [(rank, device, pending[rank::len(devices)])
                       for rank, device in enumerate(devices)]
        assignments = [item for item in assignments if item[2]]
        parts = [attempt_dir / "shards" / f"shard-{rank}" for rank, _, _ in assignments]
        progress_workers = [
            {
                "physical_gpu": device,
                "directory": attempt_dir / "shards" / f"shard-{rank}",
                "log": attempt_dir / "logs" / f"shard-{rank}.log",
            }
            for rank, device, _ in assignments
        ]
        progress = EvaluationProgress(
            parts, started, history_completed=len(records), total=len(expected),
            workers=progress_workers, method="In-Place TTT",
            checkpoint=args.checkpoint.name, inference_mode=args.inference_mode,
        )
        print(f"[evaluate] in_place_ttt GPUs={','.join(devices)} | logs: {attempt_dir / 'logs'}", flush=True)
        progress.render(force=True)
        try:
            for rank, device, assigned in assignments:
                check_interruption()
                part = attempt_dir / "shards" / f"shard-{rank}"
                ids_file = attempt_dir / "assignments" / f"shard-{rank}.json"
                write_json(ids_file, assigned)
                command = [
                    sys.executable, "-u", "-m", "in_place_ttt.evaluation.evaluate",
                    "--protocol-file", str(plan_path), "--output-dir", str(part),
                    "--sample-ids-file", str(ids_file), "--execution-attempt", str(number),
                    "--shard-index", str(rank), "--runtime-file", str(directory / "runtime.json"),
                    "--batch-size", str(args.batch_size),
                ]
                if args.batch_size_by_length:
                    command.extend(["--batch-size-by-length", *args.batch_size_by_length])
                log_path = attempt_dir / "logs" / f"shard-{rank}.log"
                log = log_path.open("x")
                logs.append(log)
                process = subprocess.Popen(
                    command, cwd=ROOT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": device, "WORLD_SIZE": "1"},
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                processes.append(process)
                attempt["workers"].append({
                    "shard_index": rank, "physical_gpu": device, "pid": process.pid,
                    "directory": str(part), "log": str(log_path), "command": command, "status": "running",
                })
                save_execution()
            while True:
                check_interruption()
                changed = False
                for process, worker in zip(processes, attempt["workers"], strict=True):
                    code = process.poll()
                    if code is not None and worker["status"] == "running":
                        worker.update(exit_code=code, finished_at=timestamp(),
                                      status="completed" if code == 0 else "failed")
                        if code != 0:
                            progress.fail_worker(
                                worker["physical_gpu"],
                                f"Worker exited with code {code}; see {worker['log']}",
                            )
                        changed = True
                if changed:
                    save_execution()
                progress.render()
                if any(worker["status"] == "failed" for worker in attempt["workers"]):
                    raise RuntimeError("An evaluation worker failed; see the current attempt logs")
                if all(worker["status"] == "completed" for worker in attempt["workers"]):
                    break
                time.sleep(0.2)
            progress.set_stage("merging")
            progress.finish_line()
            check_interruption()
            print("[evaluate] 样本评估完成，正在合并结果...", flush=True)
            records, sources = state.collect_records(directory, protocol, expected)
            merged = attempt_dir / "merged"
            summary = state.merge_records(directory, protocol, expected, records, sources, merged)
            check_interruption()
            summary.update(
                devices=devices, execution_path=str(execution_path),
                merged_summary_path=str(merged / "summary.json"),
                pool_wall_seconds=time.monotonic() - started,
            )
            write_json(directory / "summary.json", summary)
            attempt.update(status="completed", merged_summary_path=str(merged / "summary.json"))
            progress.set_stage("completed")
            progress.finish_line()
            print(f"[evaluate] 完成 | elapsed {duration(time.monotonic() - started)} | summary: {directory / 'summary.json'}", flush=True)
        except BaseException as error:
            attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
            progress.set_stage("failed", error=f"{type(error).__name__}: {error}")
            progress.finish_line()
            print(f"[evaluate] 评估未完成：{error} | logs: {attempt_dir / 'logs'}", flush=True)
            raise
        finally:
            stop_processes(processes)
            for process, worker in zip(processes, attempt["workers"], strict=True):
                if worker["status"] == "running":
                    worker.update(status="terminated", exit_code=process.returncode, finished_at=timestamp())
            for log in logs:
                log.close()
            attempt.update(finished_at=timestamp(), pool_wall_seconds=time.monotonic() - started)
            save_execution()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/reproductions/in_place_ttt")
    parser.add_argument("--output-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inference-mode", choices=("original", "optimized"))
    parser.add_argument("--mlp-chunk-size", type=int)
    parser.add_argument("--lengths", type=int, nargs="+", choices=state.LENGTHS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-size-by-length", nargs="+", default=[], metavar="LENGTH=BATCH")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1", "0,1", "1,0"):
        parser.error("Select physical GPUs 0 and/or 1 with CUDA_VISIBLE_DEVICES")
    if os.environ.get("WORLD_SIZE", "1") != "1" or any(
        name in os.environ for name in ("RANK", "LOCAL_RANK")
    ):
        parser.error("Use one standalone launcher, without torchrun")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.resume and args.output_id is None:
        parser.error("--resume requires the original --output-id")
    if args.output_id is None:
        current = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M")
        args.output_id = f"{args.checkpoint.name}_{current}"
    if not args.output_id.strip() or args.output_id in {".", ".."} or any(
        separator in args.output_id for separator in ("/", "\\")
    ):
        parser.error("--output-id must be one nonempty directory name")
    args.output_dir = args.output_root / "eval" / args.output_id / "ruler"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[evaluate] output-id={args.output_id} | ruler: {args.output_dir}", flush=True)
    interruption = {"signum": None}

    def terminate(signum, frame):
        interruption["signum"] = signum

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    execute(args, interruption)


if __name__ == "__main__":
    main()
