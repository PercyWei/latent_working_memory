"""逐 checkpoint 执行多卡样本分片，汇总结果并复用已完成样本。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from ..cli import DEFAULT_EVAL_BATCH_SIZE, ignore_resume_options, optional_positive_int, positive_int
from ..config import config_from_dict
from ..execution import evaluation_lock, run_lock
from ..plans import prepare_evaluation_plan
from ..training.checkpoint import read_checkpoint
from .evaluate import (
    IncompleteEvaluationError, finalize_sample_work, load_cached_evaluation, prepare_sample_work,
)
from .records import read_evaluation_record, write_json


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
        training_directory = Path(current.training.output_dir).expanduser().resolve()
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


def _visible_devices():
    values = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    devices = [value.strip() for value in values]
    if any(not value.isdecimal() for value in devices) or len(set(map(int, devices))) != len(devices):
        raise ValueError("set CUDA_VISIBLE_DEVICES to distinct physical GPU indices for evaluation")
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


def _write_status(path, summary, results):
    """保留各 checkpoint 最近状态，整体进度仅统计本次选择。"""
    summary["progress"] = {
        "total": len(results),
        **{name: sum(result["status"] == name for result in results)
           for name in ("pending", "running", "finished", "failed", "stopped")},
    }
    write_json(path, summary)


def _compute(results, summary, status_path, settings):
    stopping, lock, processes = threading.Event(), threading.Lock(), set()
    devices = None
    options = {name: settings[name] for name in (
        "task", "max_samples", "max_new_tokens", "lengths", "eval_id",
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
                print(prefix + invocation, end="", flush=True)
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
                        print(prefix + line, end="", flush=True)
                        log.write(prefix + line)
                        log.flush()
                code = process.wait()
            finally:
                _stop_process(process)
                process.stdout.close()
                with lock:
                    processes.remove(process)
            if code:
                raise subprocess.CalledProcessError(code, shard["command"])
        except Exception as caught:
            error = caught
            print(f"{prefix}failed: {error}", flush=True)
        with lock:
            shard.update(
                status="stopped" if stopping.is_set() else "failed" if error else "finished",
                return_code=(error.returncode if isinstance(error, subprocess.CalledProcessError)
                             else 1 if error else 0),
                finished_at=_now(),
            )
            if error:
                shard["error"] = f"{type(error).__name__}: {error}"
            _write_status(status_path, summary, results)

    def stop_workers():
        stopping.set()
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        with lock:
            running = list(processes)
        for process in running:
            _stop_process(process)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"evaluation interrupted by {signal.Signals(signum).name}")

    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in handlers:
        signal.signal(signum, interrupted)
    executor = None
    try:
        for result in results:
            if result["status"] != "pending":
                continue
            result.update(status="running", started_at=_now())
            _write_status(status_path, summary, results)
            try:
                slots = (len(devices) if devices is not None else max(1, len([
                    value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()
                ])))
                work = prepare_sample_work(
                    result["checkpoint"], settings["data"], result["output_dir"],
                    **options, num_workers=slots,
                )
                result["samples"] = {
                    "total_samples": work["total_samples"],
                    "cached_samples": work["completed_samples"],
                    "scheduled_samples": work["pending_samples"],
                }
                shards = []
                if work["workers"]:
                    if devices is None:
                        devices = _visible_devices()
                        summary["gpus"] = ",".join(devices)
                    for rank, sample_ids in enumerate(work["workers"]):
                        shards.append({
                            "rank": rank, "gpu": devices[rank], "sample_count": len(sample_ids),
                            "status": "pending",
                            "command": [*result["command"], "--worker-rank", str(rank)],
                        })
                result["workers"] = shards
                _write_status(status_path, summary, results)
                if shards:
                    with Path(result["log_path"]).open("a", encoding="utf-8") as log:
                        executor = ThreadPoolExecutor(max_workers=len(shards))
                        try:
                            futures = [executor.submit(worker, result, shard, log) for shard in shards]
                            for future in as_completed(futures):
                                future.result()
                        except BaseException:
                            stop_workers()
                            executor.shutdown(wait=True, cancel_futures=True)
                            executor = None
                            raise
                        executor.shutdown(wait=True)
                        executor = None
                errors = [f"worker {shard['rank']}: {shard['error']}" for shard in shards
                          if shard["status"] != "finished"]
                # 即使某 worker 失败，仍提交已完整覆盖的 RULER 长度；全局结果须样本齐全。
                try:
                    finalize_sample_work(result["checkpoint"], settings["data"], result["output_dir"], **options)
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
                print(f"[evaluate step={result['step']}] failed: {error}", flush=True)
            result["finished_at"] = _now()
            _write_status(status_path, summary, results)
    except BaseException:
        stop_workers()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            executor = None
        for result in results:
            for shard in result.get("workers", []):
                if shard["status"] in {"pending", "running"}:
                    shard.update(status="stopped", finished_at=_now())
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def _now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _evaluation_arguments(args):
    """公开任务与数据参数映射到统一的计划字段。"""
    values = vars(args).copy()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.resume:
        values = {name: value for name, value in values.items() if name in {
            "resume", "overwrite", "evaluate_only", "eval_run", "eval_id", "eval_task",
            "checkpoint", "output_root", "eval_batch_size_per_gpu",
        }}
        if not hasattr(args, "eval_task"):
            raise ValueError("evaluation --resume requires an explicit --eval-task")
    if args.eval_run is None:
        raise ValueError("evaluation requires --eval-run")
    if args.eval_task not in {"test", "ruler", "locomo"}:
        raise ValueError("eval-task must be test, ruler or locomo")
    if args.resume and not args.eval_id:
        raise ValueError("evaluation --resume requires an explicit --eval-id")
    if not args.resume and args.eval_task != "test" and any(
        hasattr(args, name) for name in ("max_test_samples", "max_new_tokens")
    ):
        raise ValueError("--max-test-samples and --max-new-tokens require --eval-task test")
    if not args.resume and args.eval_task != "ruler" and hasattr(args, "lengths"):
        raise ValueError("--lengths requires --eval-task ruler")
    values["task"] = values.pop("eval_task")
    if "eval_data" in values:
        values["data"] = values.pop("eval_data")
    if "max_test_samples" in values:
        values["max_samples"] = values.pop("max_test_samples")
    return argparse.Namespace(**values)


def run_evaluations(args):
    args = _evaluation_arguments(args)
    source_run_directory = Path(args.eval_run).expanduser().resolve()
    training_directory = source_run_directory / "train"
    eval_id = (args.eval_id if args.eval_id is not None
               else datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S"))
    if not eval_id.strip() or Path(eval_id).name != eval_id or eval_id in {".", ".."} or "\\" in eval_id:
        raise ValueError("eval-id must be a nonempty directory name")
    run_directory = source_run_directory
    if args.output_root:
        run_directory = Path(args.output_root).expanduser().resolve() / run_directory.name
    task_directory = run_directory / "eval" / eval_id / args.task
    with ExitStack() as stack:
        # 源 checkpoint 与重定向后的结果目录均保持占用，训练覆盖不能与评估并行。
        for directory in sorted({source_run_directory, run_directory}):
            stack.enter_context(run_lock(directory))
        stack.enter_context(evaluation_lock(run_directory, eval_id, args.task))
        if args.resume:
            plan_path = run_directory / "plan" / f"eval-{eval_id}" / f"{args.task}.json"
            if not plan_path.is_file():
                raise ValueError(f"evaluation plan does not exist: {plan_path}")
            saved = read_evaluation_record(plan_path)
            paths = saved.get("checkpoints")
        else:
            paths = None
        if args.checkpoint is not None:
            if any(re.fullmatch(r"global_step_[0-9]+", name) is None for name in args.checkpoint):
                raise ValueError("--checkpoint must contain global_step_<step> names")
            paths = [training_directory / "checkpoints" / name / "state.pt"
                     for name in args.checkpoint]
        selected, config, _ = _checkpoints(training_directory, paths)
        settings = prepare_evaluation_plan(
            run_directory, config, eval_id, args.task,
            {"checkpoints": [str(path) for path, _ in selected],
             **{name: getattr(args, name) for name in ("data", "max_samples", "max_new_tokens", "lengths")
                if hasattr(args, name)}},
            overwrite=args.overwrite, resume=args.resume,
        )
        task_directory.mkdir(parents=True, exist_ok=True)
        status_path = task_directory / "status.json"
        history = (json.loads(status_path.read_text(encoding="utf-8"))["checkpoints"]
                   if status_path.exists() else {})
        results = []
        for checkpoint, step in selected:
            output = task_directory / f"global_step_{step}"
            cached = load_cached_evaluation(
                checkpoint, settings["data"], output, task=args.task,
                max_new_tokens=settings["max_new_tokens"], max_samples=settings["max_samples"],
                lengths=settings["lengths"], eval_id=eval_id,
            )
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-m", "latent_working_memory.v4_1.evaluation.evaluate",
                "--checkpoint", str(checkpoint), "--data", settings["data"],
                "--task", settings["task"], "--output-dir", str(output), "--eval-id", eval_id,
                "--eval-batch-size-per-gpu",
                str(getattr(args, "eval_batch_size_per_gpu", DEFAULT_EVAL_BATCH_SIZE)),
            ]
            for name in ("max_samples", "max_new_tokens"):
                if settings[name] is not None:
                    command.extend(("--" + name.replace("_", "-"), str(settings[name])))
            if settings["lengths"] is not None:
                command.extend(("--lengths", *map(str, settings["lengths"])))
            result = {
                "checkpoint": str(checkpoint), "step": step, "output_dir": str(output),
                "log_path": str(output / "eval.log"), "command": command,
                "status": "finished" if cached is not None else "pending",
                **({"compute_status": "cached", "return_code": 0} if cached is not None else {}),
            }
            results.append(result)
            history[output.name] = result
            if cached is not None:
                print(f"[evaluate step={step}] cached {args.task}", flush=True)
        pending = any(result["status"] == "pending" for result in results)
        summary = {
            "action": "evaluate", "eval_run": str(source_run_directory), "eval_id": eval_id,
            "task": args.task, "gpus": "", "started_at": _now(), "status": "running",
            "eval_batch_size_per_gpu": getattr(args, "eval_batch_size_per_gpu", DEFAULT_EVAL_BATCH_SIZE),
            "selected_checkpoints": [f"global_step_{step}" for _, step in selected],
            "checkpoints": history,
        }
        _write_status(status_path, summary, results)
        try:
            if pending:
                _compute(results, summary, status_path, settings)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", nargs="+", help="本次评估的 global_step_<step> 名称；省略时新建选全部，续跑沿用计划")
    parser.add_argument("--eval-run", required=True,
                        help="运行根目录，包含 train/、plan/ 与 eval/；用于选择 checkpoint 和定位计划")
    parser.add_argument("--eval-task", choices=("test", "ruler", "locomo"))
    parser.add_argument("--eval-data",
                        help="评估数据目录，新建或覆盖计划时必填")
    parser.add_argument("--eval-id", help="评估计划标识，默认 YYYYMMDD-HHMMSS")
    parser.add_argument("--eval-batch-size-per-gpu", type=positive_int,
                        help="每卡评估 batch，默认 8")
    parser.add_argument("--output-root", help="产物父目录；默认在原 run 下保存评估结果")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true", help="恢复指定 eval-id 下已有的任务计划")
    restart.add_argument("--overwrite", action="store_true",
                         help="清理当前 task 的评估结果和计划后重新评估")
    parser.add_argument("--max-test-samples", type=optional_positive_int)
    parser.add_argument("--max-new-tokens", type=positive_int,
                        help="仅用于 test；无既有计划时默认 128")
    parser.add_argument("--lengths", nargs="+", type=positive_int)
    args = parser.parse_args(argv)
    if getattr(args, "resume", False):
        if not hasattr(args, "eval_task"):
            parser.error("evaluation --resume requires an explicit --eval-task")
        ignore_resume_options(args, {
            "resume", "overwrite", "checkpoint", "eval_run", "eval_id", "eval_task", "output_root",
            "eval_batch_size_per_gpu",
        })
    for name, default in {
        "resume": False, "overwrite": False, "checkpoint": None, "eval_run": None,
        "eval_task": "test", "eval_id": None, "output_root": None,
        "eval_batch_size_per_gpu": DEFAULT_EVAL_BATCH_SIZE,
    }.items():
        vars(args).setdefault(name, default)
    run_evaluations(args)


if __name__ == "__main__":
    main()
