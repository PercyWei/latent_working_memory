"""Accelerate runtime for replicated pretraining on CPU or CUDA."""

from datetime import timedelta

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, InitProcessGroupKwargs

from latent_working_memory.devices import validate_device


def pretraining_accelerator(device: torch.device) -> Accelerator:
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("pretraining supports CPU and CUDA devices")
    accelerator = Accelerator(
        cpu=device.type == "cpu",
        mixed_precision="bf16" if device.type == "cuda" else "no",
        # One optimizer update consumes a complete global sample batch. Its number
        # of microbatches may vary after expanding the legal memory capacities.
        gradient_accumulation_steps=1,
        device_placement=False,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                broadcast_buffers=False,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            ),
            InitProcessGroupKwargs(timeout=timedelta(hours=2)),
        ],
    )
    if accelerator.distributed_type not in {
        DistributedType.NO,
        DistributedType.MULTI_CPU,
        DistributedType.MULTI_GPU,
    }:
        raise ValueError(
            "pretraining currently supports Accelerate single-device and DDP execution"
        )
    validate_device(accelerator.device)
    return accelerator
