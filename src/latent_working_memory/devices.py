from __future__ import annotations

import os

import torch


def validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select physical GPU 0 or 1")
    physical_devices = tuple(part.strip() for part in visible.split(",") if part.strip())
    if not physical_devices or any(value not in {"0", "1"} for value in physical_devices):
        raise RuntimeError("this project only permits physical GPU 0 and 1")
    logical_index = 0 if device.index is None else device.index
    if logical_index < 0 or logical_index >= len(physical_devices):
        raise RuntimeError("the requested logical CUDA device is not visible")
