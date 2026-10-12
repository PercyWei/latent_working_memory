"""加载本阶段共同使用的冻结 checkpoint 和开发集。"""

from dataclasses import replace
import os
from pathlib import Path

import torch
import torch.distributed as dist

from latent_working_memory.v3.config import (
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainingConfig,
)
from latent_working_memory.v3.data import load_factqa
from latent_working_memory.v3.engine import initialize_device
from latent_working_memory.v3.evaluate import _check_qa_sources
from latent_working_memory.v3.objective import TokenMemoryTask
from latent_working_memory.v3.runtime import (
    dataset_identity,
    load_codec,
    read_checkpoint,
    select_examples,
    write_json as save_json,
)

__all__ = ["load_dev_task", "save_json"]


def load_dev_task(checkpoint_path, dataset_dir, device, max_trajectories):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
        dist.is_initialized() and dist.get_world_size() != 1
    ):
        raise ValueError("探索评估使用单进程，请勿使用 torchrun")
    if max_trajectories is not None and max_trajectories < 1:
        raise ValueError("max_trajectories 必须为正整数")
    checkpoint = read_checkpoint(checkpoint_path)
    raw = checkpoint["run"]["config"]
    config = ExperimentConfig(
        ModelConfig(**raw["model"]),
        ObjectiveConfig(**raw["objective"]),
        TrainingConfig(**raw["training"]),
    )
    if config.objective.method != "memory_change" or config.objective.stage not in {
        "warmup",
        "policy",
    }:
        raise ValueError("需要 memory_change 的 warmup 或 policy checkpoint")
    device = initialize_device(device)
    revision = checkpoint["run"]["resolved_model_revision"]
    model_config = replace(config.model, revision=revision) if revision else config.model
    torch.manual_seed(config.training.seed)
    codec, tokenizer = load_codec(model_config, device, config.objective)
    codec.load_trainable_state_dict(checkpoint["trainable"])
    task = TokenMemoryTask(codec, tokenizer, config.objective)
    task.requires_grad_(False)
    task.eval()
    trajectories = load_factqa(dataset_dir, tokenizer)["dev"]
    _check_qa_sources(trajectories, checkpoint["run"]["pretraining_sources"])
    source_identity = dataset_identity(trajectories)
    if source_identity != checkpoint["run"]["source_data"]["dev"]:
        raise ValueError("开发集与 checkpoint 记录的数据不一致")
    trajectories = select_examples(
        trajectories, max_trajectories, config.training.seed, "dev", pretraining=False
    )
    if not trajectories:
        raise ValueError("开发集为空")
    metadata = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "split": "dev",
        "source_dataset_signature": source_identity["fingerprint"],
        "dataset_signature": dataset_identity(trajectories)["fingerprint"],
        "config": config.to_dict(),
        "resolved_model_revision": revision,
        "device": str(device),
        "autocast_dtype": "bfloat16" if device.type == "cuda" else None,
        "pretraining": checkpoint["run"]["pretraining"],
        "global_step": checkpoint["run"]["step_offset"] + checkpoint["cursor"]["step"],
        "selection": {
            "total": source_identity["examples"],
            "selected": len(trajectories),
            "limit": max_trajectories,
            "seed": config.training.seed,
        },
    }
    return task, trajectories, metadata
