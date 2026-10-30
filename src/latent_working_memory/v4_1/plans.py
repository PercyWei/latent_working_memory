"""训练与评估的永久计划：先确定设置，再启动实际计算。"""

from dataclasses import asdict, replace
import json
from pathlib import Path

from .config import config_from_dict
from .evaluation.records import normalize_lengths, write_json
from .training.engine import training_steps


def save_training_plan(config):
    directory = Path(config.training.output_dir).parent / "plan"
    directory.mkdir(parents=True, exist_ok=False)
    payload = asdict(config)
    write_json(directory / "config.json", {key: payload[key] for key in ("seed", "model", "teacher")})
    write_json(directory / "train.json", payload["training"])
    return directory


def load_training_plan(plan_dir):
    directory = Path(plan_dir).expanduser().resolve()
    common = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    training = json.loads((directory / "train.json").read_text(encoding="utf-8"))
    if not isinstance(common, dict) or set(common) != {"seed", "model", "teacher"}:
        raise ValueError("common plan must contain exactly seed, model and teacher")
    config = config_from_dict({**common, "training": training})
    if directory != Path(config.training.output_dir).expanduser().resolve().parent / "plan":
        raise ValueError("training plan directory differs from its saved run directory")
    if not config.training.train_only:
        path = directory / f"eval-{config.training.eval_id}" / "test.json"
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "training_run": str(Path(config.training.output_dir).expanduser().resolve()),
            "model": asdict(config.model), "seed": config.seed,
            "eval_id": config.training.eval_id, "task": "test",
            "data": str(Path(config.training.test_data).expanduser().resolve()),
            "max_samples": config.training.max_test_samples,
            "max_new_tokens": config.training.max_new_tokens, "lengths": None,
        }
        if evaluation != expected:
            raise ValueError("automatic test plan differs from the saved training execution settings")
    return config


def validate_training_plan(config, checkpoint):
    """计划保留请求预算，checkpoint 保存选样后实际推导的总步数。"""
    batch_size = (len(checkpoint["rng"]) * config.training.micro_batch_size
                  * config.training.gradient_accumulation_steps)
    steps = training_steps(config.training, checkpoint["binding"]["train"]["episodes"], batch_size)
    expected = replace(config, training=replace(config.training, max_steps=steps))
    if asdict(expected) != checkpoint["config"]:
        raise ValueError("checkpoint configuration differs from the saved training plan")


def prepare_evaluation_plan(run_dir, config, eval_id, task, options):
    """省略的值沿用已有计划；首次创建必须明确数据，RULER 仅允许扩展。"""
    if (not isinstance(eval_id, str) or not eval_id.strip()
        or Path(eval_id).name != eval_id or eval_id in {".", ".."} or "\\" in eval_id):
        raise ValueError("eval-id must be a nonempty directory name")
    if task not in {"test", "locomo", "ruler"}:
        raise ValueError("task must be test, locomo or ruler")
    path = Path(run_dir).expanduser().resolve() / "plan" / f"eval-{eval_id}" / f"{task}.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    previous = saved or {}
    data = options.get("data", previous.get("data"))
    if data is None:
        option = "--test-data" if task == "test" else "--benchmark-data"
        raise ValueError(f"specify {option} when creating an evaluation plan")
    settings = {
        "training_run": str(Path(config.training.output_dir).expanduser().resolve()),
        "model": asdict(config.model), "seed": config.seed, "eval_id": eval_id, "task": task,
        "data": str(Path(data).expanduser().resolve()),
        "max_samples": options.get("max_samples", previous.get("max_samples")) if task == "test" else None,
        "max_new_tokens": (options.get("max_new_tokens", previous.get("max_new_tokens", 128))
                           if task == "test" else None),
        "lengths": (list(normalize_lengths(options.get("lengths", previous.get("lengths"))))
                    if task == "ruler" else None),
    }
    for name in ("max_samples", "max_new_tokens"):
        value = settings[name]
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be null or a positive integer")
    if settings["max_new_tokens"] is None and task == "test":
        raise ValueError("max_new_tokens must be a positive integer")
    if saved is not None:
        if any(value != saved[key] for key, value in settings.items() if key != "lengths"):
            raise ValueError("evaluation plan settings differ; use a new --eval-id or --output-root")
        if task == "ruler" and not set(saved["lengths"]).issubset(settings["lengths"]):
            raise ValueError("RULER lengths may only stay the same or expand within an evaluation plan")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, settings)
    return settings
