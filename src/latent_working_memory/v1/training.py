"""跨阶段精度上下文与可训练模型状态的保存、加载。"""

from __future__ import annotations
from contextlib import nullcontext
from typing import Any
import torch
from latent_working_memory.v1.backbone import LatentMemoryBackbone
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter


def precision_context(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


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
