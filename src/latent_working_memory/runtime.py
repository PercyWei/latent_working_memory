"""v3／v4 共用的设备初始化和随机状态保存。"""

from datetime import timedelta
import os
import random
import re

import numpy as np
import torch
import torch.distributed as dist
from verl.utils.distributed import initialize_global_process_group


def initialize_device(device):
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("memory training supports CPU and CUDA")
    if device.type == "cuda":
        visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
        if (
            any(re.fullmatch(r"[0-9]+", value) is None for value in visible)
            or len({int(value) for value in visible}) != len(visible)
        ):
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must explicitly select distinct non-negative physical GPU indices"
            )
        index = int(os.environ.get("LOCAL_RANK", device.index or 0))
        if not 0 <= index < len(visible):
            raise ValueError("the requested logical CUDA device is not visible")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        if device.type == "cuda":
            initialize_global_process_group(timeout_second=7200)
        else:
            dist.init_process_group("gloo", timeout=timedelta(hours=2))
    return device


def capture_rng(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
