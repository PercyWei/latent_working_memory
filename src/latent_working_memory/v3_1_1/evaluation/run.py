"""逐 checkpoint 执行多卡样本分片，汇总结果并复用已完成样本。"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
from zoneinfo import ZoneInfo

import torch

from ..config import config_from_dict
from ..execution import evaluation_lock, run_lock, stop_process, visible_devices
from .plans import prepare_evaluation_plan
from .progress import EvaluationProgress
from ..training.checkpoint import read_checkpoint
from .evaluate import (
    finalize_sample_work, load_cached_evaluation, prepare_sample_work,
)
from ..io import write_json
from .shards import IncompleteEvaluationError


def _checkpoints(directory, paths=None):
    directory = Path(directory).expanduser().resolve()
    paths = (list(directory.glob("checkpoints/global_step_*/state.pt"))
             if paths is None else [Path(path) for path in paths])
    if not paths:
        raise ValueError(f"no complete checkpoints found in {directory}")
    if len(set(paths)) != len(paths):
        raise ValueError("checkpoint list contains duplicate paths")
    selected, config = [], None
    for path in paths:
        state = read_checkpoint(path)
        current = config_from_dict(state["config"])
        step = state["cursor"]["step"]
        training_directory = Path(current.training.run.output_dir).expanduser().resolve()
        if path.parent.name != f"global_step_{step}":
            raise ValueError(f"checkpoint directory and saved step differ: {path}")
        if path != training_directory / "checkpoints" / f"global_step_{step}" / "state.pt":
            raise ValueError("checkpoint path differs from its saved training directory")
        if training_directory != directory:
            raise ValueError("selected checkpoints must belong to the same training run")
        if config is not None and current != config:
            raise ValueError("selected checkpoints have different training configurations")
        directory, config = training_directory, current
        selected.append((path, step))
        del state
    selected.sort(key=lambda item: item[1])
    if len({step for _, step in selected}) != len(selected):
        raise ValueError("different checkpoints cannot share the same optimizer step")
    return selected, config, directory


def _write_status(path, summary, results):
    """保留各 checkpoint 最近状态，整体进度仅统计本次选择。"""
    summary["progress"] = {
        "total": len(results),
        **{name: sum(result["status"] == name for result in results)
           for name in ("pending", "running", "finished", "failed", "stopped")},
    }
    write_json(path, summary)


def _compute(results, summary, status_path, settings, task_module):
    stopping, lock, processes = threading.Event(), threading.Lock(), set()
    devices = None
    options = {name: settings[name] for name in (
        "max_samples", "max_new_tokens", "lengths", "eval_id",
    )}

    def worker(result, shard, log):
        process, error = None, None
        prefix = f"[evaluate step={result['step']} gpu={shard['gpu']} rank={shard['rank']}] "
        try:
            with lock:
                if stopping.is_set():
                    return
                shard.update(status="running", started_at=_now())
                _write_status(status_path, summary, results)
                invocation = "$ " + shlex.join(shard["command"]) + "\n"
                log.write(prefix + invocation)
                log.flush()
                process = subprocess.Popen(
                    shard["command"], env=dict(os.environ, CUDA_VISIBLE_DEVICES=shard["gpu"],
                                               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false"),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    encoding="utf-8", bufsize=1, start_new_session=True,
                )
                processes.add(process)
            try:
                for line in process.stdout:
                    with lock:
                        log.write(prefix + line)
                        log.flush()
                code = process.wait()
            finally:
                stop_process(process)
                process.stdout.close()
                with lock:
                    processes.remove(process)
            if code:
                raise subprocess.CalledProcessError(code, shard["command"])
        except Exception as caught:
            error = caught
            with lock:
                log.write(f"{prefix}failed: {error}\n")
                log.flush()
        with lock:
            shard.update(
                status="stopped" if stopping.is_set() else "failed" if error else "finished",
                return_code=(error.returncode if isinstance(error, subprocess.CalledProcessError)
                             else 1 if error else 0),
                finished_at=_now(),
            )
            if error:
                shard["error"] = (f"process exited with code {error.returncode}"
                                  if isinstance(error, subprocess.CalledProcessError)
                                  else f"{type(error).__name__}: {error}")
            _write_status(status_path, summary, results)

    def stop_workers():
        stopping.set()
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        with lock:
            running = list(processes)
        for process in running:
            stop_process(process)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"evaluation interrupted by {signal.Signals(signum).name}")

    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in handlers:
        signal.signal(signum, interrupted)
    executor, progress = None, None
    try:
        for result in results:
            if result["status"] != "pending":
                continue
            progress = None
            result.update(status="running", started_at=_now())
            _write_status(status_path, summary, results)
            try:
                slots = (len(devices) if devices is not None else max(1, len([
                    value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()
                ])))
                work = prepare_sample_work(
                    result["checkpoint"], settings["data"], result["output_dir"],
                    task_module, **options, num_workers=slots,
                )
                result["samples"] = {
                    "total_samples": work["total_samples"],
                    "cached_samples": work["completed_samples"],
                    "scheduled_samples": work["pending_samples"],
                }
                shards = []
                if work["workers"]:
                    if devices is None:
                        if not torch.cuda.is_available():
                            raise RuntimeError("CUDA is required to run v3.1.1 evaluation")
                        devices = visible_devices()
                        summary["gpus"] = ",".join(devices)
                    for rank, sample_ids in enumerate(work["workers"]):
                        shards.append({
                            "rank": rank, "gpu": devices[rank], "sample_count": len(sample_ids),
                            "status": "pending",
                            "command": [*result["command"], "--worker-rank", str(rank)],
                        })
                result["workers"] = shards
                _write_status(status_path, summary, results)
                progress = EvaluationProgress(
                    result["output_dir"], task_module.NAME, result["step"], work, shards,
                )
                progress.render(force=True)
                if shards:
                    with Path(result["log_path"]).open("a", encoding="utf-8") as log:
                        executor = ThreadPoolExecutor(max_workers=len(shards))
                        try:
                            pending = {executor.submit(worker, result, shard, log) for shard in shards}
                            while pending:
                                finished, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                                for future in finished:
                                    future.result()
                                with lock:
                                    progress.render()
                        except BaseException:
                            stop_workers()
                            executor.shutdown(wait=True, cancel_futures=True)
                            executor = None
                            raise
                        executor.shutdown(wait=True)
                        executor = None
                errors = [f"worker {shard['rank']}: {shard['error']}" for shard in shards
                          if shard["status"] != "finished"]
                progress.set_stage("merging")
                try:
                    finalize_sample_work(
                        result["checkpoint"], settings["data"], result["output_dir"],
                        task_module, **options,
                    )
                    result["samples"].update(completed_samples=work["total_samples"], pending_samples=0)
                except IncompleteEvaluationError as error:
                    result["samples"].update(
                        completed_samples=error.completed_samples,
                        pending_samples=error.total_samples - error.completed_samples,
                    )
                    errors.append(f"merge: {type(error).__name__}: {error}")
                except Exception as error:
                    errors.append(f"merge: {type(error).__name__}: {error}")
                if errors:
                    code = next((shard["return_code"] for shard in shards if shard["return_code"]), 1)
                    result.update(status="failed", return_code=code, error="; ".join(errors))
                else:
                    result.update(status="finished", return_code=0)
            except Exception as error:
                if stopping.is_set():
                    raise
                result.update(status="failed", return_code=1, error=f"{type(error).__name__}: {error}")
            result["finished_at"] = _now()
            _write_status(status_path, summary, results)
            if progress is not None:
                progress.set_stage("completed" if result["status"] == "finished" else "failed")
                progress.finish_line()
            if result["status"] == "failed":
                print(f"[evaluate step={result['step']}] failed: {result['error']} | log: {result['log_path']}",
                      flush=True)
    except BaseException as error:
        stop_workers()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            executor = None
        for result in results:
            for shard in result.get("workers", []):
                if shard["status"] in {"pending", "running"}:
                    shard.update(status="stopped", finished_at=_now())
        if progress is not None:
            progress.set_stage("stopped" if isinstance(error, KeyboardInterrupt) else "failed")
            progress.finish_line()
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def _now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def run_evaluations(args, task_module):
    """按任务入口规范化后的参数执行 checkpoint 评估。"""
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.resume and not args.eval_id:
        raise ValueError("evaluation --resume requires an explicit --eval-id")
    task = task_module.NAME
    source_run_directory = Path(args.eval_run).expanduser().resolve()
    training_directory = source_run_directory / "train"
    eval_id = (args.eval_id if args.eval_id is not None
               else datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S"))
    if not eval_id.strip() or Path(eval_id).name != eval_id or eval_id in {".", ".."} or "\\" in eval_id:
        raise ValueError("eval-id must be a nonempty directory name")
    run_directory = source_run_directory
    if args.output_root:
        run_directory = Path(args.output_root).expanduser().resolve() / run_directory.name
    task_directory = run_directory / "eval" / eval_id / task
    with ExitStack() as stack:
        # 源 checkpoint 与重定向后的结果目录均保持占用，训练覆盖不能与评估并行。
        for directory in sorted({source_run_directory, run_directory}):
            stack.enter_context(run_lock(directory))
        stack.enter_context(evaluation_lock(run_directory, eval_id, task))
        saved = None
        if args.resume:
            plan_path = run_directory / "plan" / f"eval-{eval_id}" / f"{task}.json"
            if not plan_path.is_file():
                raise ValueError(f"evaluation plan does not exist: {plan_path}")
            saved = json.loads(plan_path.read_text(encoding="utf-8"))
            paths = saved["checkpoints"]
        else:
            paths = None
        if args.checkpoint is not None:
            if any(re.fullmatch(r"global_step_[0-9]+", name) is None for name in args.checkpoint):
                raise ValueError("--checkpoint must contain global_step_<step> names")
            paths = [training_directory / "checkpoints" / name / "state.pt"
                     for name in args.checkpoint]
        selected, config, _ = _checkpoints(training_directory, paths)
        options = task_module.plan_options(args, saved)
        settings = prepare_evaluation_plan(
            run_directory, config, eval_id, task,
            {"checkpoints": [str(path) for path, _ in selected],
             **options},
            overwrite=args.overwrite, resume=args.resume,
        )
        execution, worker_arguments = task_module.execution_settings(args, settings)
        task_directory.mkdir(parents=True, exist_ok=True)
        status_path = task_directory / "status.json"
        history = (json.loads(status_path.read_text(encoding="utf-8"))["checkpoints"]
                   if status_path.exists() else {})
        results = []
        for checkpoint, step in selected:
            output = task_directory / f"global_step_{step}"
            cached = load_cached_evaluation(
                checkpoint, settings["data"], output, task_module,
                max_new_tokens=settings["max_new_tokens"], max_samples=settings["max_samples"],
                lengths=settings["lengths"], eval_id=eval_id,
            )
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-m", "latent_working_memory.v3_1_1.evaluation.worker",
                "--checkpoint", str(checkpoint), "--data", settings["data"],
                "--task", settings["task"], "--output-dir", str(output), "--eval-id", eval_id,
                *worker_arguments,
            ]
            result = {
                "checkpoint": str(checkpoint), "step": step, "output_dir": str(output),
                "log_path": str(output / "eval.log"), "command": command,
                **execution,
                "status": "finished" if cached is not None else "pending",
                **({"compute_status": "cached", "return_code": 0} if cached is not None else {}),
            }
            results.append(result)
            history[output.name] = result
            if cached is not None:
                print(f"[evaluate step={step}] cached {task}", flush=True)
        pending = any(result["status"] == "pending" for result in results)
        summary = {
            "action": "evaluate", "eval_run": str(source_run_directory), "eval_id": eval_id,
            "task": task, "gpus": "", "started_at": _now(), "status": "running",
            "device": "cuda",
            **execution,
            "selected_checkpoints": [f"global_step_{step}" for _, step in selected],
            "checkpoints": history,
        }
        _write_status(status_path, summary, results)
        try:
            if pending:
                _compute(results, summary, status_path, settings, task_module)
        except BaseException as error:
            for result in results:
                if result["status"] in {"pending", "running"}:
                    result.update(status="stopped", finished_at=_now())
            summary.update(status="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                           error=f"{type(error).__name__}: {error}", finished_at=_now())
            _write_status(status_path, summary, results)
            raise
        failures = sum(result["status"] != "finished" for result in results)
        summary.update(status="failed" if failures else "finished", finished_at=_now())
        _write_status(status_path, summary, results)
        if failures:
            raise RuntimeError(f"evaluation failed for {failures}/{len(results)} checkpoints; see {status_path}")
        print(f"evaluation complete: {len(selected)} checkpoints, results={task_directory}", flush=True)
    return task_directory
