from __future__ import annotations

import os

import torch


def validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select physical GPUs")
    allowed = os.environ.get("LWM_ALLOWED_PHYSICAL_GPUS", "0,1")
    allowed_devices = {part.strip() for part in allowed.split(",")}
    if not allowed_devices or any(not value.isdecimal() for value in allowed_devices):
        raise ValueError("LWM_ALLOWED_PHYSICAL_GPUS must contain physical GPU indices")
    physical_devices = tuple(part.strip() for part in visible.split(",") if part.strip())
    if not physical_devices or any(value not in allowed_devices for value in physical_devices):
        raise RuntimeError(f"this run only permits physical GPUs {allowed}")
    logical_index = 0 if device.index is None else device.index
    if logical_index < 0 or logical_index >= len(physical_devices):
        raise RuntimeError("the requested logical CUDA device is not visible")
