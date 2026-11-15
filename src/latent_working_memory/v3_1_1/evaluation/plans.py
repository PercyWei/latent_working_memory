"""独立评估计划的创建、恢复与重建。"""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

from ..io import write_json


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
        "max_samples": options["max_samples"],
        "max_new_tokens": options["max_new_tokens"],
        "lengths": options["lengths"],
    }
    for name in ("max_samples", "max_new_tokens"):
        value = settings[name]
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be null or a positive integer")
    if overwrite and not Path(settings["data"]).is_dir():
        raise FileNotFoundError(f"evaluation data directory does not exist: {settings['data']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if output.exists():
            shutil.rmtree(output)
    write_json(path, settings)
    return settings
