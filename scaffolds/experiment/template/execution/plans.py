"""先保存请求计划，再由 worker 解析数据与实际预算；不依赖具体方法模块。"""

from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from ..config import normalize_saved_config
from .state import checkpoint_file, read_checkpoint, write_json


def directory_name(name):
    if not name.strip() or Path(name).name != name or name in {".", ".."} or "\\" in name:
        raise ValueError("identifiers must be single directory names")
    return name


def resolve_training(config, samples, world_size):
    result = deepcopy(config)
    training = result["training"]
    batch = world_size * training["micro_batch_size"] * training["gradient_accumulation_steps"]
    budgets = []
    if training["epochs"] is not None:
        budgets.append(training["epochs"] * math.ceil(samples / batch))
    if training["max_steps"] is not None:
        budgets.append(training["max_steps"])
    if samples < 1 or not budgets or min(budgets) < 1:
        raise ValueError("training requires selected samples and a positive step budget")
    training["max_steps"] = min(budgets)
    return result


def resolve_run(config, method_name):
    """在加锁前确定运行路径；尚不创建或删除产物。"""
    result = deepcopy(config)
    training = result["training"]
    run_id = training["run_id"] or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    for name in (method_name, run_id):
        directory_name(name)
    training["output_root"] = str(Path(training["output_root"]).expanduser().resolve())
    directory = Path(training["output_root"]) / f"{method_name}_{run_id}"
    training.update(run_id=run_id, output_dir=str(directory / "train"))
    training["dataset_dir"] = str(Path(training["dataset_dir"]).expanduser().resolve())
    return result


def create_run(config, overwrite=False):
    """调用方持有 run 排他锁；清理后先保存训练计划，再启动计算。"""
    directory = Path(config["training"]["output_dir"]).parent
    if overwrite and not Path(config["training"]["dataset_dir"]).is_dir():
        raise FileNotFoundError(config["training"]["dataset_dir"])
    if directory.exists():
        if overwrite:
            shutil.rmtree(directory)
        elif any(directory.iterdir()):
            raise ValueError("run directory is not empty; resume it or use --overwrite")
    plan = directory / "plan"
    plan.mkdir(parents=True, exist_ok=False)
    write_json(
        plan / "config.json", {key: value for key, value in config.items() if key != "training"}
    )
    write_json(plan / "train.json", config["training"])


def load_plan(run_dir):
    directory = Path(run_dir).expanduser().resolve()
    common = json.loads((directory / "plan/config.json").read_text(encoding="utf-8"))
    training = json.loads((directory / "plan/train.json").read_text(encoding="utf-8"))
    if Path(training["output_dir"]) != directory / "train":
        raise ValueError("plan output directory differs from its location")
    return normalize_saved_config({**common, "training": training})


def inspect_resume(checkpoint, world_size):
    path = checkpoint_file(checkpoint)
    state = read_checkpoint(path)
    output = Path(state["config"]["training"]["output_dir"])
    step = state["cursor"]["step"]
    if type(step) is not int or not 0 <= step <= state["config"]["training"]["max_steps"]:
        raise ValueError("checkpoint step is outside its training budget")
    if path != output / "checkpoints" / f"global_step_{step}" / "state.pt":
        raise ValueError("checkpoint path differs from its saved run and step")
    latest = max(
        int(p.parent.name.rsplit("_", 1)[1])
        for p in output.glob("checkpoints/global_step_*/state.pt")
    )
    if latest != step:
        raise ValueError("resume requires the latest complete checkpoint")
    if any(
        not isinstance(state[key], list) or len(state[key]) != world_size
        for key in ("rng", "method_rng")
    ):
        raise ValueError("resume requires the original process count")
    config = load_plan(output.parent)
    if (
        resolve_training(config, state["binding"]["train"]["episodes"], world_size)
        != state["config"]
    ):
        raise ValueError("checkpoint configuration differs from the permanent plan")
    if state["scheduler"]["last_epoch"] != step:
        raise ValueError("scheduler and checkpoint step differ")
    return config, state


def evaluation_plan(run_dir, config, eval_id, task, options, overwrite=False):
    if task != "test":
        raise ValueError("the template implements test; add a task adapter before extending it")
    for name in (eval_id, task):
        directory_name(name)
    path = Path(run_dir) / "plan" / f"eval-{eval_id}" / f"{task}.json"
    saved = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() and not overwrite else None
    )
    previous = saved or {}
    data = options.get("data", previous.get("data"))
    if data is None:
        raise ValueError("data is required when creating an evaluation plan")
    settings = {
        "training_run": config["training"]["output_dir"],
        "model": config["model"],
        "seed": config["seed"],
        "eval_id": eval_id,
        "task": task,
        "data": str(Path(data).expanduser().resolve()),
        "max_samples": options.get("max_samples", previous.get("max_samples")),
        "max_new_tokens": options.get("max_new_tokens", previous.get("max_new_tokens", 128)),
    }
    for key in ("max_samples", "max_new_tokens"):
        value = settings[key]
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be null or a positive integer")
    if task == "test" and settings["max_new_tokens"] is None:
        raise ValueError("test generation requires a positive max_new_tokens")
    if saved is not None:
        if settings != saved:
            raise ValueError("evaluation settings changed; use a new eval-id")
    if overwrite:
        if not Path(settings["data"]).is_dir():
            raise FileNotFoundError(settings["data"])
        output = Path(run_dir) / "eval" / eval_id / task
        if output.exists():
            shutil.rmtree(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, settings)
    return settings
