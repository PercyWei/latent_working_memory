"""训练计划保存、读取与 checkpoint 配置核对。"""

from dataclasses import replace
import json
from pathlib import Path

from ..config import config_from_dict, config_to_dict
from ..io import write_json
from .engine import training_steps


def save_training_plan(config):
    directory = Path(config.training.run.output_dir).parent / "plan"
    directory.mkdir(parents=True, exist_ok=False)
    payload = config_to_dict(config)
    write_json(directory / "config.json", {
        key: payload[key] for key in ("stage", "seed", "model_name_or_path", "model", "compute")
    })
    write_json(directory / "train.json", payload["training"])
    return directory


def load_training_plan(plan_dir):
    directory = Path(plan_dir).expanduser().resolve()
    common = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    training = json.loads((directory / "train.json").read_text(encoding="utf-8"))
    if not isinstance(common, dict) or set(common) != {
        "stage", "seed", "model_name_or_path", "model", "compute",
    }:
        raise ValueError(
            "common plan must contain exactly stage, seed, model_name_or_path, model and compute"
        )
    config = config_from_dict({**common, "training": training})
    if directory != Path(config.training.run.output_dir).expanduser().resolve().parent / "plan":
        raise ValueError("training plan directory differs from its saved run directory")
    return config


def validate_training_plan(config, checkpoint):
    """请求预算留在计划中，checkpoint 记录选样后推导的实际总步数。"""
    batch_size = len(checkpoint["rng"]) * config.training.loop.micro_batch_size * config.training.loop.gradient_accumulation_steps
    statistics = checkpoint["binding"]["train"]
    bin_sizes = ([bucket["samples"] for bucket in statistics["length_bins"]]
                 if config.training.data.length_curriculum else None)
    steps = training_steps(config.training.loop, statistics["episodes"], batch_size, bin_sizes)
    expected = replace(config, training=replace(
        config.training, loop=replace(config.training.loop, max_steps=steps),
    ))
    if config_to_dict(expected) != checkpoint["config"]:
        raise ValueError("checkpoint configuration differs from its saved training plan")
