"""四种运行模式的同步编排；训练与评估模块负责启动并等待计算进程。"""

from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from .plans import create_run, directory_name, evaluation_plan, inspect_resume
from .state import checkpoint_file, read_checkpoint, scheduler_lock, write_json


def _now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def start_run(config, method_name, world_size, train, evaluate):
    config = create_run(config, method_name)
    return _train_and_test(config, world_size, None, train, evaluate)


def resume_run(checkpoint, world_size, train, evaluate):
    # 接口不接受新配置或 overrides；训练参数从原计划恢复。
    config, state = inspect_resume(checkpoint, world_size)
    complete = state["cursor"]["step"] == state["config"]["training"]["max_steps"]
    if not complete:
        del state
        return _train_and_test(config, world_size, checkpoint_file(checkpoint), train, evaluate)
    output = Path(config["training"]["output_dir"])
    result = {
        "step": state["cursor"]["step"],
        "checkpoint": str(checkpoint_file(checkpoint)),
        "output_dir": str(output),
    }
    del state
    write_json(output / "summary.json", result)
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    status.update(status="finished", return_code=0, finished_at=_now())
    status.pop("error", None)
    write_json(output / "status.json", status)
    _automatic_test(config, result, evaluate)
    return result


def _train_and_test(config, world_size, checkpoint, train, evaluate):
    output = Path(config["training"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    job = {
        "action": "resume" if checkpoint else "train",
        "world_size": world_size,
        "started_at": _now(),
        "status": "running",
    }
    write_json(output / "status.json", job)
    try:
        train(output.parent / "plan", checkpoint)
        result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        saved, state = inspect_resume(result["checkpoint"], world_size)
        if (
            saved != config
            or result["step"] != state["cursor"]["step"]
            or result["step"] != state["config"]["training"]["max_steps"]
            or result["output_dir"] != str(output)
        ):
            raise ValueError("training returned before its planned budget completed")
        del state
    except BaseException as error:
        stopped = isinstance(error, (KeyboardInterrupt, SystemExit))
        code = (
            error.returncode
            if isinstance(error, subprocess.CalledProcessError)
            else (error.code if isinstance(error, SystemExit) else 130 if stopped else 1)
        )
        write_json(
            output / "status.json",
            {
                **job,
                "status": "stopped" if stopped else "failed",
                "return_code": code,
                "error": f"{type(error).__name__}: {error}",
                "finished_at": _now(),
            },
        )
        raise
    write_json(
        output / "status.json",
        {**job, "status": "finished", "return_code": 0, "finished_at": _now()},
    )
    _automatic_test(config, result, evaluate)
    return result


def _automatic_test(config, result, evaluate):
    if not config["training"]["train_only"]:
        evaluate_checkpoints(
            [result["checkpoint"]], config["training"]["eval_id"], "test", {}, evaluate
        )


def evaluate_checkpoints(checkpoints, eval_id, task, options, evaluate):
    """独立评估与自动 test 复用本入口；evaluate 一次负责一个 checkpoint。"""
    selected, config = [], None
    for checkpoint in checkpoints:
        path = checkpoint_file(checkpoint)
        state = read_checkpoint(path)
        current = state["config"]
        step = state["cursor"]["step"]
        if (
            path
            != Path(current["training"]["output_dir"])
            / "checkpoints"
            / f"global_step_{step}"
            / "state.pt"
        ):
            raise ValueError("evaluation checkpoint path differs from its saved run and step")
        if config is not None and current != config:
            raise ValueError("evaluation checkpoints must belong to the same run and configuration")
        config = current
        selected.append((path, step))
        del state
    if not selected or len({step for _, step in selected}) != len(selected):
        raise ValueError("select distinct checkpoints from one run")
    selected.sort(key=lambda item: item[1])
    eval_id = eval_id or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    directory_name(eval_id)
    directory_name(task)
    run_dir = Path(config["training"]["output_dir"]).parent
    directory = run_dir / "eval" / eval_id / task
    with scheduler_lock(directory):
        settings = evaluation_plan(run_dir, config, eval_id, task, options)
        path = directory / "status.json"
        history = (
            json.loads(path.read_text(encoding="utf-8"))["checkpoints"] if path.exists() else {}
        )
        names = [f"global_step_{step}" for _, step in selected]
        status = {
            "action": "evaluate",
            "eval_id": eval_id,
            "task": task,
            "selected_checkpoints": names,
            "checkpoints": history,
            "status": "running",
            "started_at": _now(),
        }
        for name in names:
            history[name] = {"status": "pending"}
        write_json(path, status)
        for checkpoint, step in selected:
            row = {"checkpoint": str(checkpoint), "status": "running", "started_at": _now()}
            history[f"global_step_{step}"] = row
            write_json(path, status)
            try:
                output = directory / f"global_step_{step}"
                output.mkdir(parents=True, exist_ok=True)
                verified = evaluate(checkpoint, settings, output)
                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                if (
                    verified != summary
                    or summary["checkpoint"] != str(checkpoint)
                    or summary["task"] != task
                    or summary["eval_id"] != eval_id
                    or not (output / "details.jsonl").is_file()
                ):
                    raise ValueError("evaluation did not commit matching summary and details")
            except Exception as error:
                row.update(
                    status="failed",
                    return_code=(
                        error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
                    ),
                    error=f"{type(error).__name__}: {error}",
                    finished_at=_now(),
                )
            except BaseException as error:
                row.update(
                    status="stopped",
                    return_code=130,
                    error=f"{type(error).__name__}: {error}",
                    finished_at=_now(),
                )
                for name in names:
                    if history[name]["status"] == "pending":
                        history[name]["status"] = "stopped"
                status.update(status="stopped", finished_at=_now())
                write_json(path, status)
                raise
            else:
                row.update(status="finished", return_code=0, finished_at=_now())
            write_json(path, status)
        failures = sum(history[name]["status"] != "finished" for name in names)
        status.update(status="failed" if failures else "finished", finished_at=_now())
        write_json(path, status)
        if failures:
            raise RuntimeError(
                f"evaluation failed for {failures}/{len(names)} checkpoints; see {path}"
            )
    return directory
