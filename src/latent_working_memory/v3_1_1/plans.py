"""训练与独立评估的永久计划：先保存设置，再启动计算。"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

from .config import config_from_dict, config_to_dict
from .evaluation.records import normalize_lengths, write_json
from .training.engine import training_steps


def save_training_plan(config):
    directory = Path(config.training.run.output_dir).parent / "plan"
    directory.mkdir(parents=True, exist_ok=False)
    payload = config_to_dict(config)
    write_json(directory / "config.json", {
        key: payload[key] for key in ("stage", "seed", "model_name_or_path", "model", "compute", "method")
    })
    write_json(directory / "train.json", payload["training"])
    return directory


def load_training_plan(plan_dir):
    directory = Path(plan_dir).expanduser().resolve()
    common = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    training = json.loads((directory / "train.json").read_text(encoding="utf-8"))
    if not isinstance(common, dict) or set(common) != {
        "stage", "seed", "model_name_or_path", "model", "compute", "method",
    }:
        raise ValueError(
            "common plan must contain exactly stage, seed, model_name_or_path, model, compute and method"
        )
    config = config_from_dict({**common, "training": training})
    if directory != Path(config.training.run.output_dir).expanduser().resolve().parent / "plan":
        raise ValueError("training plan directory differs from its saved run directory")
    return config


def validate_training_plan(config, checkpoint):
    """请求预算留在计划中，checkpoint 记录选样后推导的实际总步数。"""
    batch_size = len(checkpoint["rng"]) * config.training.loop.micro_batch_size * config.training.loop.gradient_accumulation_steps
    steps = training_steps(config.training.loop, checkpoint["binding"]["train"]["episodes"], batch_size)
    expected = replace(config, training=replace(
        config.training, loop=replace(config.training.loop, max_steps=steps),
    ))
    if config_to_dict(expected) != checkpoint["config"]:
        raise ValueError("checkpoint configuration differs from its saved training plan")


def prepare_evaluation_plan(run_dir, config, eval_id, task, options, overwrite=False, resume=False):
    """新建、恢复或重建任务计划；恢复固定协议，并可追加 checkpoint。"""
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if (not isinstance(eval_id, str) or not eval_id.strip()
        or Path(eval_id).name != eval_id or eval_id in {".", ".."} or "\\" in eval_id):
        raise ValueError("eval-id must be a nonempty directory name")
    if task not in {"test", "locomo", "ruler"}:
        raise ValueError("task must be test, locomo or ruler")
    path = Path(run_dir).expanduser().resolve() / "plan" / f"eval-{eval_id}" / f"{task}.json"
    output = Path(run_dir).expanduser().resolve() / "eval" / eval_id / task
    if resume:
        if not path.is_file():
            raise ValueError(f"evaluation plan does not exist: {path}")
        saved = json.loads(path.read_text(encoding="utf-8"))
        identity = {
            "training_run": str(Path(config.training.run.output_dir).expanduser().resolve()),
            "model_name_or_path": config.model_name_or_path,
            "compute": asdict(config.compute),
            "model": asdict(config.model), "seed": config.seed, "eval_id": eval_id, "task": task,
        }
        if any(saved[key] != value for key, value in identity.items()):
            raise ValueError("evaluation plan settings differ from the original training run")
        checkpoints = sorted(set(saved["checkpoints"]) | set(options["checkpoints"]),
                             key=lambda value: int(Path(value).parent.name.removeprefix("global_step_")))
        if checkpoints != saved["checkpoints"]:
            saved["checkpoints"] = checkpoints
            write_json(path, saved)
        return saved
    elif not overwrite:
        if path.exists():
            raise ValueError("evaluation plan already exists; use --resume or --overwrite")
        if output.exists() and any(output.iterdir()):
            raise ValueError("evaluation results already exist; use --overwrite to rebuild the task")
    data = options.get("data")
    if data is None:
        raise ValueError("specify --eval-data when creating an evaluation plan")
    settings = {
        "training_run": str(Path(config.training.run.output_dir).expanduser().resolve()),
        "model_name_or_path": config.model_name_or_path,
        "compute": asdict(config.compute),
        "model": asdict(config.model), "seed": config.seed, "eval_id": eval_id, "task": task,
        "checkpoints": list(options["checkpoints"]),
        "data": str(Path(data).expanduser().resolve()),
        "max_samples": options.get("max_samples") if task == "test" else None,
        "max_new_tokens": (options.get("max_new_tokens", 128)
                           if task == "test" else None),
        "lengths": (list(normalize_lengths(options.get("lengths")))
                    if task == "ruler" else None),
    }
    for name in ("max_samples", "max_new_tokens"):
        value = settings[name]
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be null or a positive integer")
    if settings["max_new_tokens"] is None and task == "test":
        raise ValueError("max_new_tokens must be a positive integer")
    if overwrite and not Path(settings["data"]).is_dir():
        raise FileNotFoundError(f"evaluation data directory does not exist: {settings['data']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if output.exists():
            shutil.rmtree(output)
    write_json(path, settings)
    return settings
