"""训练与恢复的控制进程，沿用 v4.1 的启动和记录流程。"""

from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

import torch

from .cli import config_overrides, parse_run_args as parse_args
from ..config import load_config
from ..io import write_json
from ..execution import run_lock, stop_process, visible_devices
from .plans import load_training_plan, save_training_plan, validate_training_plan
from .checkpoint import checkpoint_file, read_checkpoint


def _training_output_dir(config):
    run_id = config.training.run.run_id
    if run_id is None:
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M")
        prefix = ("policy-post" if config.uses_boundary_policy else
                  "compressor-post" if config.stage == "post" else "pretrain")
        run_id = f"{prefix}_k{config.model.structure.memory_slots}_{timestamp}"
    return Path(config.training.run.output_root).expanduser().resolve() / run_id / "train"


def _execute(command, environment, log_path):
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    process, previous = None, {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        with log_path.open("a", encoding="utf-8") as log:
            invocation = "$ " + shlex.join(command) + "\n"
            print(invocation, end="", flush=True)
            log.write(invocation)
            log.flush()
            process = subprocess.Popen(
                command, env=environment, start_new_session=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    except BaseException:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if process is not None:
            stop_process(process)
        raise
    finally:
        if process is not None:
            process.stdout.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _run_training(args, config, output_dir):
    resume = None
    devices = visible_devices()
    world_size = len(devices)
    if args.resume:
        resume = checkpoint_file(args.checkpoint)
        state = read_checkpoint(resume)
        output_dir = Path(state["config"]["training"]["run"]["output_dir"]).expanduser().resolve()
        plan_dir = output_dir.parent / "plan"
        config = load_training_plan(plan_dir)
        validate_training_plan(config, state)
        step = state["cursor"]["step"]
        if resume != output_dir / "checkpoints" / f"global_step_{step}" / "state.pt":
            raise ValueError("checkpoint path differs from its saved training directory")
        completed = [
            int(path.parent.name.rsplit("_", 1)[1])
            for path in (output_dir / "checkpoints").glob("global_step_*/state.pt")
        ]
        if max(completed) > step:
            raise ValueError("resume the latest complete checkpoint in this run")
        if len(state["rng"]) != world_size or len(state["boundary_rng"]) != world_size:
            raise ValueError("resume requires the original number of training processes")
        complete = step == state["config"]["training"]["loop"]["max_steps"]
        del state
        if complete:
            result = {"step": step, "checkpoint": str(resume), "output_dir": str(output_dir)}
            write_json(output_dir / "summary.json", result)
            status_path = output_dir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status.update(status="finished", return_code=0)
            status.pop("error", None)
            status.setdefault(
                "finished_at", datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            )
            write_json(status_path, status)
            return result
    if not torch.cuda.is_available():
        raise RuntimeError("v3.1.1 training requires a CUDA GPU")
    if not args.resume:
        if args.overwrite:
            if not Path(config.training.data.dataset_dir).is_dir():
                raise FileNotFoundError(f"training data directory does not exist: {config.training.data.dataset_dir}")
            if output_dir.parent.exists():
                shutil.rmtree(output_dir.parent)
        elif output_dir.parent.exists() and any(output_dir.parent.iterdir()):
            raise ValueError(f"run directory is not empty: {output_dir.parent}; use --resume or --overwrite")
        config = replace(config, training=replace(
            config.training, run=replace(config.training.run, output_dir=str(output_dir)),
        ))
        plan_dir = save_training_plan(config)
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--local-addr=127.0.0.1",
        "--nnodes=1", f"--nproc-per-node={world_size}", "--module", "--",
        "latent_working_memory.v3_1_1.training.train",
    ]
    environment = dict(
        os.environ, CUDA_VISIBLE_DEVICES=",".join(devices),
        PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
    )
    command.extend(("--plan-dir", str(plan_dir)))
    if resume is not None:
        command.extend(("--resume", "--checkpoint", str(resume)))
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    job = {
        "action": "resume" if resume is not None else "train",
        "device": "cuda", "gpus": ",".join(devices),
        "command": command,
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
    }
    write_json(status_path, {**job, "status": "running"})
    try:
        _execute(command, environment, output_dir / "train.log")
        result = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    except BaseException as error:
        stopped = isinstance(error, (KeyboardInterrupt, SystemExit))
        code = (error.returncode if isinstance(error, subprocess.CalledProcessError)
                else error.code if isinstance(error, SystemExit)
                else 130 if isinstance(error, KeyboardInterrupt) else 1)
        write_json(status_path, {
            **job, "status": "stopped" if stopped else "failed", "return_code": code,
            "error": f"{type(error).__name__}: {error}",
            "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        })
        raise
    write_json(status_path, {
        **job, "status": "finished", "return_code": 0,
        "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
    })
    return result


def run_job(args):
    if args.resume:
        checkpoint = checkpoint_file(args.checkpoint)
        if checkpoint.name != "state.pt" or re.fullmatch(r"global_step_[1-9][0-9]*", checkpoint.parent.name) is None:
            raise ValueError("resume requires a canonical global_step_<step>/state.pt checkpoint")
        output_dir = checkpoint.parent.parent.parent
        config = None
    else:
        config = load_config(args.config, config_overrides(args))
        output_dir = _training_output_dir(config)
        config = replace(config, training=replace(
            config.training, run=replace(config.training.run, run_id=output_dir.parent.name),
        ))
    with run_lock(output_dir.parent, exclusive=True):
        return _run_training(args, config, output_dir)


def main(argv=None):
    args = parse_args(argv)
    try:
        return run_job(args)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode if error.returncode > 0 else 128 - error.returncode) from error
    except KeyboardInterrupt as error:
        raise SystemExit(130) from error


if __name__ == "__main__":
    main()
