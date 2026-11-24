"""LaCT 单卡与多卡评估的独立进程调度。"""

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


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "reproductions"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evaluation_runtime import (  # noqa: E402
    EvaluationProgress, duration, source_record, stop_processes, timestamp, write_json,
)
from merge_evaluation import merge_records  # noqa: E402
from records import BATCHING, build_protocol, load_samples, sample_index  # noqa: E402
from scheduling import (  # noqa: E402
    add_lact_arguments, check_protocol, collect_records, evaluation_lock,
    execution_parameters, existing_protocol, reject_live_workers,
    resolve_protocol_arguments, worker_arguments,
)


def execute_lact(args, interruption):
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
        saved = existing_protocol(directory) if args.resume else None
        resolve_protocol_arguments(args, saved)
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        parameters = execution_parameters(args, devices)
        samples = load_samples({"lengths": args.lengths})
        protocol = build_protocol(args.checkpoint, args.lengths, args.inference_mode, samples)
        if saved is not None:
            check_protocol(saved, protocol, directory)
        expected = sample_index(samples)
        records, sources = collect_records(directory, protocol, expected)
        pending = [sample_id for sample_id in expected if sample_id not in records]
        numbers = [int(path.name.split("-")[1]) for path in directory.glob("attempts/attempt-*")]
        number = max(numbers, default=-1) + 1
        attempt_dir = directory / "attempts" / f"attempt-{number:04d}"
        for name in ("logs", "assignments", "shards"):
            (attempt_dir / name).mkdir(parents=True, exist_ok=False)
        source = source_record()
        round_plan = {
            **protocol, **parameters, "created_at": timestamp(), "source": source,
            "assigned_samples": len(pending), "history_completed": len(records),
            "worker_arguments": worker_arguments(args), "python": sys.executable,
        }
        write_json(attempt_dir / "plan.json", round_plan)
        global_plan = directory / "evaluation_plan.json"
        if not global_plan.is_file():
            write_json(global_plan, {
                **protocol, "created_at": timestamp(), "source": source,
                "num_shards": None, "shard_index": None, "batch_size": None,
                "batching": BATCHING, "assigned_samples": len(expected), "source_shards": sources,
            })
        execution_path = directory / "launcher_execution.json"
        attempts = json.loads(execution_path.read_text())["attempts"] if execution_path.is_file() else []
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
        assignments = [(rank, device, pending[rank::len(devices)]) for rank, device in enumerate(devices)]
        assignments = [item for item in assignments if item[2]]
        parts = [attempt_dir / "shards" / f"shard-{rank}" for rank, _, _ in assignments]
        workers = [{
            "physical_gpu": device,
            "directory": attempt_dir / "shards" / f"shard-{rank}",
            "log": attempt_dir / "logs" / f"shard-{rank}.log",
        } for rank, device, _ in assignments]
        progress = EvaluationProgress(
            parts, started, history_completed=len(records), total=len(expected),
            workers=workers, method="LaCT", checkpoint=args.checkpoint,
            inference_mode=args.inference_mode,
        )
        progress.finish_line()
        print(f"[evaluate] lact GPUs={','.join(devices)} | logs: {attempt_dir / 'logs'}", flush=True)
        progress.render(force=True)
        try:
            for rank, device, assigned in assignments:
                check_interruption()
                part = attempt_dir / "shards" / f"shard-{rank}"
                ids_file = attempt_dir / "assignments" / f"shard-{rank}.json"
                write_json(ids_file, assigned)
                command = [
                    sys.executable, "-u", str(ROOT / "reproductions/lact/evaluation/evaluate.py"),
                    "--checkpoint", str(args.checkpoint), "--output-dir", str(part),
                    "--num-shards", str(len(devices)), "--shard-index", str(rank),
                    "--sample-ids-file", str(ids_file), *worker_arguments(args),
                ]
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
                        worker.update(exit_code=code, finished_at=timestamp(), status="completed" if code == 0 else "failed")
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
            progress.render(force=True)
            progress.set_stage("merging")
            progress.finish_line()
            check_interruption()
            print("[evaluate] 样本评估完成，正在合并结果...", flush=True)
            merged = attempt_dir / "merged"
            progress.finish_line()
            plan, summary = merge_records(directory, protocol, merged)
            write_json(global_plan, plan)
            summary.update(
                plan_path=str(global_plan), execution_path=str(execution_path),
                devices=devices, merged_summary_path=str(merged / "summary.json"),
                pool_wall_seconds=time.monotonic() - started,
            )
            attempt.update(status="completed", merged_summary_path=str(merged / "summary.json"))
            write_json(directory / "summary.json", summary)
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
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/reproductions/lact")
    parser.add_argument("--output-id")
    parser.add_argument("--resume", action="store_true")
    add_lact_arguments(parser)
    args, worker_args = parser.parse_known_args()
    allowed_devices = ("0", "1", "0,1", "1,0")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in allowed_devices:
        parser.error("Select physical GPUs 0 and/or 1 with CUDA_VISIBLE_DEVICES")
    if os.environ.get("WORLD_SIZE", "1") != "1" or any(
        name in os.environ for name in ("RANK", "LOCAL_RANK")
    ):
        parser.error("Use one standalone launcher, without torchrun")
    if any(value.split("=", 1)[0] in ("--num-shards", "--shard-index") for value in worker_args):
        parser.error("Multi-GPU evaluation assigns row shards automatically")
    if worker_args:
        parser.error(f"Unknown LaCT evaluation arguments: {' '.join(worker_args)}")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.resume and args.output_id is None:
        parser.error("--resume requires the original --output-id")
    if args.output_id is None:
        current = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M")
        args.output_id = f"{args.checkpoint.name}_{current}"
    if not args.output_id.strip() or args.output_id in {".", ".."} or any(
        separator in args.output_id for separator in ("/", "\\")
    ):
        parser.error("--output-id must be one nonempty directory name")
    args.output_root = args.output_root.expanduser().resolve()
    args.output_dir = args.output_root / "eval" / args.output_id / "ruler"
    print(f"[evaluate] output-id={args.output_id} | ruler: {args.output_dir}", flush=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    interruption = {"signum": None}
    def terminate(signum, frame):
        interruption["signum"] = signum
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    execute_lact(args, interruption)


if __name__ == "__main__":
    main()
