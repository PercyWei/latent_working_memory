"""从任意已保存 checkpoint 读取新训练计划的权重来源及历史前缀。"""

import json
import math
from pathlib import Path
import re

from latent_working_memory.v4.checkpoint import read_checkpoint


def load_continuation(path):
    """不依赖来源计划是否完成，仅校验 checkpoint 与已记录的连续训练前缀。"""
    path = Path(path).expanduser().resolve()
    if path.parent.name != "checkpoints" or re.fullmatch(r"global_step_[0-9]+", path.name) is None:
        raise ValueError("continuation requires an output/checkpoints/global_step_<step> directory")
    step = int(path.name.removeprefix("global_step_"))
    checkpoint = read_checkpoint(path)
    if type(checkpoint["cursor"]["step"]) is not int or checkpoint["cursor"]["step"] != step or step < 1:
        raise ValueError("continuation checkpoint directory step differs from its saved cursor")
    output = path.parent.parent
    if Path(checkpoint["run"]["training"]["output_dir"]).expanduser().resolve() != output:
        raise ValueError("continuation checkpoint must belong to its saved output_dir/checkpoints")
    metrics_path = output / "metrics.jsonl"
    history = []
    with metrics_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            if not isinstance(record, dict) or type(record.get("step")) is not int:
                raise ValueError(f"{metrics_path}:{line_number}: expected an integer step")
            if record["step"] <= step:
                history.append(record)
            if record["step"] == step and "train/loss" in record:
                break
    training_steps = [record["step"] for record in history if "train/loss" in record]
    if training_steps != list(range(1, step + 1)):
        raise ValueError(f"{metrics_path} must contain every training step from 1 through {step}")
    if "continuation" not in checkpoint["run"]:
        # 已有原始训练的历史指标尚未包含阶段编号，其阶段固定为 0。
        for record in history:
            if "train/loss" in record:
                record.setdefault("train/stage", 0)
    last_training = next(record for record in reversed(history) if "train/loss" in record)
    if type(last_training["train/stage"]) is not int or last_training["train/stage"] < 0:
        raise ValueError(f"{metrics_path}: last train/stage must be a nonnegative integer")
    epoch = last_training["train/epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or not math.isfinite(epoch) or epoch < 0:
        raise ValueError(f"{metrics_path}: last train/epoch must be finite and nonnegative")
    return checkpoint, history
