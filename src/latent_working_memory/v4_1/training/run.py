"""完整运行、训练恢复与独立评估。"""

import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

from ..cli import DEFAULT_CONFIG, config_overrides, parse_run_args as parse_args
from ..config import load_config
from ..evaluation.records import write_json
from ..evaluation.run import run_evaluations
from ..plans import (
    load_training_plan, prepare_evaluation_plan, save_training_plan, validate_training_plan,
)
from .checkpoint import checkpoint_file, read_checkpoint
from .train import _training_output_dir


def _visible_devices():
    devices = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if any(not value.isdecimal() for value in devices) or len(set(map(int, devices))) != len(devices):
        raise ValueError("set CUDA_VISIBLE_DEVICES to distinct physical GPU indices")
    return [str(int(value)) for value in devices]


def _stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


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
            _stop_process(process)
        raise
    finally:
        if process is not None:
            process.stdout.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def run_job(args):
    if args.evaluate_only:
        return run_evaluations(args)
    devices = _visible_devices()
    resume = None
    complete = False
    if args.resume:
        resume = checkpoint_file(args.checkpoint)
        state = read_checkpoint(resume)
        output_dir = Path(state["config"]["training"]["output_dir"]).expanduser().resolve()
        plan_dir = output_dir.parent / "plan"
        config = load_training_plan(plan_dir)
        validate_training_plan(config, state)
        step = state["cursor"]["step"]
        if resume != output_dir / "checkpoints" / f"global_step_{step}" / "state.pt":
            raise ValueError("checkpoint path differs from its saved training directory")
        if len(state["rng"]) != len(devices) or len(state["query_rng"]) != len(devices):
            raise ValueError("resume requires the original number of training processes")
        completed_steps = [int(path.parent.name.rsplit("_", 1)[1])
                           for path in (output_dir / "checkpoints").glob("global_step_*/state.pt")]
        if max(completed_steps) > step:
            raise ValueError("resume the latest complete checkpoint in this run")
        complete = step == state["config"]["training"]["max_steps"]
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
    else:
        config = load_config(getattr(args, "config", DEFAULT_CONFIG), config_overrides(args))
        now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
        config = replace(config, training=replace(
            config.training,
            run_id=config.training.run_id or now,
            eval_id=config.training.eval_id or "auto",
            test_data=str(Path(config.training.test_data or config.training.dataset_dir).expanduser().resolve()),
        ))
        output_dir = _training_output_dir(config, None)
        if output_dir.parent.exists() and any(output_dir.parent.iterdir()):
            raise ValueError(f"run directory is not empty: {output_dir.parent}; use --resume")
        config = replace(config, training=replace(config.training, output_dir=str(output_dir)))
        plan_dir = save_training_plan(config)
        if not config.training.train_only:
            prepare_evaluation_plan(
                output_dir.parent, config, config.training.eval_id, "test",
                {"data": config.training.test_data,
                 "max_samples": config.training.max_test_samples,
                 "max_new_tokens": config.training.max_new_tokens},
            )

    if not complete:
        command = [
            sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc-per-node={len(devices)}", "--module", "latent_working_memory.v4_1.training.train",
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
            "gpus": environment["CUDA_VISIBLE_DEVICES"], "command": command,
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

    if not config.training.train_only:
        print("训练完成，开始最终 test 评估。", flush=True)
        run_evaluations(argparse.Namespace(
            checkpoint=[result["checkpoint"]], eval_run=None, eval_type="test", benchmark=None,
            eval_id=config.training.eval_id, output_root=None,
        ))
    return result


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
