"""跨阶段精度上下文、资源统计与可训练模型状态。"""

from __future__ import annotations
from contextlib import nullcontext
from typing import Any
import torch
import torch.distributed as dist
from latent_working_memory.v1.backbone import LatentMemoryBackbone
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter


def precision_context(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def training_resources(device: torch.device, seconds: float) -> dict[str, float | int]:
    """Report the slowest rank and largest allocated peak; callers own timing/reset boundaries."""
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    if dist.is_initialized() and dist.get_world_size() > 1:
        resources = torch.tensor([seconds, peak_memory], dtype=torch.float64, device=device)
        dist.all_reduce(resources, op=dist.ReduceOp.MAX)
        seconds, peak_memory = resources.tolist()
    return {"seconds": seconds, "peak_memory_bytes": int(peak_memory)}


def trainable_model_state(
    backbone: LatentMemoryBackbone, writer: JointMemoryWriter, value_network: GrowthValueNetwork
) -> dict[str, Any]:
    return {
        "backbone": backbone.trainable_state_dict(),
        "writer": writer.state_dict(),
        "value_network": value_network.state_dict(),
    }


def load_trainable_model_state(
    state: dict[str, Any],
    backbone: LatentMemoryBackbone,
    writer: JointMemoryWriter,
    value_network: GrowthValueNetwork,
) -> None:
    if set(state) != {"backbone", "writer", "value_network"}:
        raise ValueError("invalid trainable model_state fields")
    backbone.load_trainable_state_dict(state["backbone"])
    writer.load_state_dict(state["writer"])
    value_network.load_state_dict(state["value_network"])
