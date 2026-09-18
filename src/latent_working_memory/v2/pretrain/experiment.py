"""两卡一组调度独立实验，保存配置快照，支持完整训练和断点续训。"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from latent_working_memory.v2.pretrain.train import read_experiment


def write_status(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def free_memory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
    )
    return {int(line.split(",")[0]): int(line.split(",")[1]) / 1024 for line in output.splitlines()}


def prepare_runs(args):
    records = []
    names = [f"{config.parent.name}_{args.run_date}" for config in args.experiments]
    if len(set(names)) != len(names):
        raise ValueError("experiments must have unique names")
    status_path = args.output_dir / "status.json"
    if args.resume:
        previous = json.loads(status_path.read_text())
        if previous["group"] != args.group or [r["name"] for r in previous["runs"]] != names:
            raise ValueError("resume requires the same experiment list, date and group")
        for record in previous["runs"]:
            if record["state"] == "running":
                try:
                    os.kill(record["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError(f"{record['name']} still has an active process")
        return previous["runs"]
    if status_path.exists():
        raise ValueError("existing series requires --resume")
    for config, name in zip(args.experiments, names, strict=True):
        model, selection, training = read_experiment(config.resolve())
        if args.model_path is not None:
            model = replace(model, model_name_or_path=str(args.model_path.resolve()))
        selection = replace(selection, dataset_dir=str(Path(selection.dataset_dir).resolve()))
        plan = args.output_dir / "plan" / name
        plan.mkdir(parents=True, exist_ok=True)
        for filename, value in (
            ("model.json", asdict(model)),
            ("selection.json", asdict(selection)),
            (
                "experiment.json",
                {
                    "model": "model.json",
                    "selection": "selection.json",
                    "training": training.to_dict(),
                },
            ),
        ):
            write_status(plan / filename, value)
        output = args.output_dir / "train" / name
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"new training requires an empty output directory: {output}")
        command = [
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "latent_working_memory.v2.pretrain.train",
            "--experiment",
            str(plan / "experiment.json"),
            "--output-dir",
            str(output),
            "--swanlab-project",
            "latent-working-memory-v2",
            "--swanlab-mode",
            "online",
            "--swanlab-group",
            args.group,
            "--swanlab-tag",
            "study:reconstruction",
        ]
        records.append(
            {
                "name": name,
                "source_experiment": str(config.resolve()),
                "command": command,
                "output": str(output),
                "state": "queued",
            }
        )
    return records


def resume_command(record):
    """Reuse the saved configuration and latest checkpoint, including after final evaluation fails."""
    output = Path(record["output"])
    result = output / "training-result.json"
    if result.exists() and json.loads(result.read_text())["complete"]:
        record["state"] = "complete"
        return None
    command = list(record["command"])
    if output.exists() and any(output.iterdir()):
        checkpoints = sorted((output / "checkpoints").glob("step-*.pt"))
        if not checkpoints:
            raise ValueError(f"cannot resume {output}: no checkpoint")
        command += ["--resume", str(checkpoints[-1])]
    return command


def run_queue(args, gpu_pairs, records):
    commands = {r["name"]: resume_command(r) for r in records}
    for record in records:
        if commands[record["name"]] is not None:
            record["state"] = "queued"
    status = {
        "group": args.group,
        "gpu_pairs": gpu_pairs,
        "runs": records,
        "scheduler_pid": os.getpid(),
        "state": "planned",
    }
    status_path = args.output_dir / "status.json"
    write_status(status_path, status)
    if args.plan_only:
        return
    active = {}
    failed = False
    try:
        while True:
            for pair, (process, record, stream) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                result = Path(record["output"]) / "training-result.json"
                complete = (
                    code == 0 and result.exists() and json.loads(result.read_text())["complete"]
                )
                record.update(
                    state="complete" if complete else "failed",
                    exit_code=code,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                failed |= not complete
                del active[pair]
            if not failed and any(r["state"] == "queued" for r in records):
                memory = free_memory()
                for pair in gpu_pairs:
                    pair = tuple(pair)
                    if pair in active or any(memory[gpu] < args.min_free_gib for gpu in pair):
                        continue
                    record = next((r for r in records if r["state"] == "queued"), None)
                    if record is None:
                        break
                    log = args.output_dir / f"{record['name']}.log"
                    stream = log.open("a")
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=",".join(map(str, pair)),
                        OMP_NUM_THREADS="4",
                        TOKENIZERS_PARALLELISM="true",
                        RAYON_NUM_THREADS="4",
                        HF_HUB_OFFLINE="1",
                        PYTORCH_ALLOC_CONF="expandable_segments:True",
                    )
                    try:
                        process = subprocess.Popen(
                            commands[record["name"]],
                            env=env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    except BaseException:
                        stream.close()
                        raise
                    record.update(
                        state="running",
                        gpus=list(pair),
                        pid=process.pid,
                        log=str(log),
                        started_at=datetime.now(timezone.utc).isoformat(),
                    )
                    active[pair] = (process, record, stream)
            status["updated_at"] = datetime.now(timezone.utc).isoformat()
            status["state"] = (
                "failed"
                if failed
                else ("complete" if all(r["state"] == "complete" for r in records) else "running")
            )
            write_status(status_path, status)
            if not active and (failed or status["state"] == "complete"):
                break
            time.sleep(10)
    finally:
        # Each torchrun has its own process group. Never signal unrelated GPU tasks.
        if active:
            for process, record, stream in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                stream.close()
                record.update(state="interrupted", exit_code=process.returncode)
            status["state"] = "interrupted"
            write_status(status_path, status)
    if failed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Run v2 experiments on exclusive pairs of GPUs")
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--group", required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--model-path", type=Path, help="本地模型目录，写入运行配置快照")
    parser.add_argument("--min-free-gib", type=float, default=75)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    if not gpus or len(gpus) % 2 or len(gpus) != len(set(gpus)) or set(gpus) - {4, 5, 6, 7}:
        parser.error("use pairs of distinct physical GPUs from 4,5,6,7")
    gpu_pairs = [gpus[i : i + 2] for i in range(0, len(gpus), 2)]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        with (args.output_dir / "scheduler.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_queue(args, gpu_pairs, prepare_runs(args))
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
