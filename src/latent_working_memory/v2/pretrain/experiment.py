"""按空闲 GPU 执行独立实验；每张卡一个进程，保存命令、队列状态和退出结果。"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_status(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def free_memory():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
    )
    return {int(line.split(",")[0]): int(line.split(",")[1]) / 1024 for line in text.splitlines()}


def main():
    parser = argparse.ArgumentParser(description="Run independent v2 experiments on exclusive GPUs")
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--group", required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--min-free-gib", type=float, default=70)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)) or set(gpus) - {4, 5, 6, 7}:
        parser.error("this experiment series uses distinct physical GPUs 4-7")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scheduler.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = args.output_dir / "status.json"
        if status_path.exists() and not args.resume:
            parser.error("existing series requires --resume")
        records = []
        for config in args.experiments:
            name = f"{config.parent.name}_{args.run_date}"
            output = args.output_dir / "train" / name
            command = [
                sys.executable,
                "-u",
                "-m",
                "latent_working_memory.v2.pretrain.train",
                "--experiment",
                str(config.resolve()),
                "--output-dir",
                str(output.resolve()),
                "--swanlab-project",
                "latent-working-memory-v2",
                "--swanlab-mode",
                "online",
                "--swanlab-group",
                args.group,
                "--swanlab-tag",
                "study:reconstruction",
            ]
            result = output / "training-result.json"
            completed = result.exists() and json.loads(result.read_text())["complete"]
            if args.resume and not completed and output.exists():
                checkpoints = sorted((output / "checkpoints").glob("step-*.pt"))
                if not checkpoints:
                    raise ValueError(f"cannot resume {output}: no checkpoint")
                command += ["--resume", str(checkpoints[-1].resolve())]
            records.append(
                {
                    "name": name,
                    "command": command,
                    "output": str(output.resolve()),
                    "state": "complete" if completed else "queued",
                }
            )
        if len({record["name"] for record in records}) != len(records):
            parser.error("experiments must have unique names")
        status = {"group": args.group, "gpus": gpus, "runs": records, "scheduler_pid": os.getpid()}
        active = {}
        failed = False
        while True:
            for gpu, (process, record, stream) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                result_path = Path(record["output"]) / "training-result.json"
                complete = (
                    code == 0
                    and result_path.exists()
                    and json.loads(result_path.read_text())["complete"]
                )
                record.update(
                    state="complete" if complete else "failed",
                    exit_code=code,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                failed |= not complete
                del active[gpu]
            if not failed:
                memory = free_memory()
                for gpu in gpus:
                    if gpu in active or memory[gpu] < args.min_free_gib:
                        continue
                    record = next((r for r in records if r["state"] == "queued"), None)
                    if record is None:
                        break
                    log = args.output_dir / f"{record['name']}.log"
                    stream = log.open("a")
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=str(gpu),
                        OMP_NUM_THREADS="4",
                        TOKENIZERS_PARALLELISM="false",
                        HF_HUB_OFFLINE="1",
                        PYTORCH_ALLOC_CONF="expandable_segments:True",
                    )
                    process = subprocess.Popen(
                        record["command"], env=env, stdout=stream, stderr=subprocess.STDOUT
                    )
                    record.update(
                        state="running",
                        gpu=gpu,
                        pid=process.pid,
                        log=str(log.resolve()),
                        started_at=datetime.now(timezone.utc).isoformat(),
                    )
                    active[gpu] = (process, record, stream)
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
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
