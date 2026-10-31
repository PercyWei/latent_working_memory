"""方法接口示例：从一段数值序列预测标量，不承担运行调度。"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class Sample:
    id: str
    values: tuple[float, ...]
    target: float
    document_id: str


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_samples(config, split, data_dir=None, max_samples=None):
    """读取一种 JSONL 格式；长度过滤后按稳定排序取样，身份绑定实际内容。"""
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"unknown split: {split}")
    training = config["training"]
    directory = Path(data_dir or training["dataset_dir"]).expanduser().resolve()
    samples = []
    ids = set()
    for line_number, line in enumerate(
        (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        row = json.loads(line)
        location = f"{split}.jsonl:{line_number}"
        if not isinstance(row, dict) or set(row) != {"id", "values", "target"}:
            raise ValueError(f"{location}: expected id, values and target")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ValueError(f"{location}: sample id must be non-empty and unique")
        if not isinstance(row["values"], list) or not row["values"]:
            raise ValueError(f"{location}: values must be a non-empty numeric list")
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in [*row["values"], row["target"]]
        ):
            raise ValueError(f"{location}: inputs and target must be finite numbers")
        ids.add(row["id"])
        samples.append(
            Sample(
                row["id"],
                tuple(float(value) for value in row["values"]),
                float(row["target"]),
                row["id"],
            )
        )
    statistics = {"raw_document_ids": sorted(ids), "raw_samples": len(samples)}
    if split in {"train", "dev"}:
        max_length = training["max_train_data_tokens"]
        if max_length is not None:
            samples = [sample for sample in samples if len(sample.values) <= max_length]
        if max_samples is None:
            max_samples = training[f"max_{split}_samples"]
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be a positive integer or null")
    samples.sort(key=lambda sample: _digest([config["seed"], split, sample.id]))
    if max_samples is not None:
        samples = samples[:max_samples]
    content = [
        {"id": sample.id, "values": sample.values, "target": sample.target} for sample in samples
    ]
    identity = {"split": split, "episodes": len(samples), "content_sha256": _digest(content)}
    statistics.update(
        selected_samples=len(samples), selected_tokens=sum(len(sample.values) for sample in samples)
    )
    return tuple(samples), identity, statistics


class SequenceRegressor(nn.Module):
    def __init__(self, hidden_dim, noise_std):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.noise_std = noise_std

    def forward(self, means):
        return self.network(means).squeeze(-1)


def method_name(config):
    return f"mean-regression-h{config['model']['hidden_dim']}"


def build_model(config, device):
    return SequenceRegressor(**config["model"]).to(device)


def build_optimizer(model, config):
    training = config["training"]
    return torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )


def compute_losses(model, samples, generator=None):
    """返回逐样本损失；全局 batch 的归约由训练流程统一处理。"""
    device = next(model.parameters()).device
    means = torch.tensor(
        [[sum(sample.values) / len(sample.values)] for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor(
        [sample.target for sample in samples], dtype=torch.float32, device=device
    )
    if model.training and generator is not None:
        # 这个 RNG 独立于模型初始化和数据排列，必须随 checkpoint 一并恢复。
        noise = torch.randn(means.shape, generator=generator, device="cpu").to(device)
        means = means + noise * model.noise_std
    mse = (model(means) - targets).square()
    return {"loss": mse, "mse": mse}


@torch.no_grad()
def predict(model, sample, settings):
    device = next(model.parameters()).device
    means = torch.tensor(
        [[sum(sample.values) / len(sample.values)]], dtype=torch.float32, device=device
    )
    prediction = model(means).item()
    error = prediction - sample.target
    return {
        "id": sample.id,
        "document_id": sample.document_id,
        "prediction": prediction,
        "reference": sample.target,
        "squared_error": error**2,
        "absolute_error": abs(error),
    }


def summarize(records):
    count = len(records)
    return {
        "samples": count,
        "mse": sum(row["squared_error"] for row in records) / count,
        "mae": sum(row["absolute_error"] for row in records) / count,
    }


def evaluation_protocol(config, settings):
    """记录真正影响输入与评分的规则；本任务没有生成 token 的步骤。"""
    return {
        "task": "sequence-regression",
        "input": "arithmetic-mean-of-all-values",
        "prediction": "scalar",
        "metrics": ["mse", "mae"],
        "aggregation": "unweighted-sample-mean",
        "training_noise": False,
    }


def trainable_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_trainable_state_dict(model, state):
    model.load_state_dict(state, strict=True)
