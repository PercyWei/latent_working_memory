"""设备、通用随机状态与计算身份；不依赖具体实验版本。"""

import hashlib
import os
from pathlib import Path
import random

import numpy as np
import torch


def initialize_device(requested):
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        if requested == "cuda"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    else:
        torch.set_num_threads(1)
    return device


def seed_all(seed, device=None):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed(seed)


def capture_rng(device=None):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device)
        if device is not None and device.type == "cuda"
        else None,
    }


def restore_rng(state, device=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker_count(device, cpu_processes):
    if device == "cpu":
        return cpu_processes
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ValueError("set CUDA_VISIBLE_DEVICES explicitly before using CUDA")
    count = torch.cuda.device_count()
    if count < 1:
        raise ValueError("CUDA_VISIBLE_DEVICES contains no available GPUs")
    return count
