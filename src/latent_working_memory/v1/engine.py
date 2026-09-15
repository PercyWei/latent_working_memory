"""verl engine extension for continuous-memory training with replicated parameters."""

from contextlib import contextmanager, nullcontext
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from verl.utils.distributed import initialize_global_process_group
from verl.workers.config import FSDPOptimizerConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.engine import BaseEngine, EngineRegistry

from latent_working_memory.devices import validate_device
from latent_working_memory.v1.training import precision_context


def initialize_device(device):
    device = torch.device(device)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("v1 training supports CPU and CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        if device.type == "cuda":
            initialize_global_process_group(timeout_second=7200)
        else:
            dist.init_process_group("gloo", timeout=timedelta(hours=2))
    if device.type == "cuda":
        if dist.is_initialized():
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        elif device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    validate_device(device)
    return device


class MemoryEngine(BaseEngine):
    """Common verl train_batch lifecycle; stages implement forward_backward_batch.

    The replicated backend deliberately does not issue parameter collectives on
    every forward: episode/BPTT segment counts may differ between ranks.
    """

    def __init__(self, model, device, learning_rate, weight_decay, gradient_clip, optimizer_fused=False):
        self.model, self.device = model, torch.device(device)
        if optimizer_fused and self.device.type != "cuda":
            raise ValueError("optimizer_fused requires CUDA")
        self.optimizer_config = FSDPOptimizerConfig(
            lr=learning_rate, weight_decay=weight_decay, clip_grad=gradient_clip,
            override_optimizer_config={"fused": True} if optimizer_fused else None
        )
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.mode = None

    def initialize(self):
        self.parameters = list(self.model.trainable_parameters())
        self.module = self.model
        if self.world_size > 1:
            self.module = DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
        self.optimizer = build_optimizer(self.parameters, self.optimizer_config)

    @property
    def is_param_offload_enabled(self):
        return False

    @property
    def is_optimizer_offload_enabled(self):
        return False

    def get_data_parallel_rank(self):
        return self.rank

    def get_data_parallel_size(self):
        return self.world_size

    def get_data_parallel_group(self):
        return dist.group.WORLD if self.world_size > 1 else None

    def is_mp_src_rank_with_outputs(self):
        return True

    @contextmanager
    def train_mode(self, **kwargs):
        self.mode = "train"
        self.module.train()
        try:
            yield
        finally:
            self.mode = None

    @contextmanager
    def eval_mode(self, **kwargs):
        was_training = self.model.training
        self.mode = "eval"
        self.module.eval()
        try:
            yield
        finally:
            self.module.train(was_training)
            self.mode = None

    def gradient_context(self, synchronize):
        return nullcontext() if synchronize or self.world_size == 1 else self.module.no_sync()

    def autocast(self):
        return precision_context(self.device)

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self):
        # Models divide by the global sample count. Preserve their backward
        # scale, then undo DDP's average on FP32 parameter gradients.
        if self.world_size > 1:
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(self.world_size)
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.optimizer_config.clip_grad, error_if_nonfinite=True
        )
        self.optimizer.step()
        return float(norm)


__all__ = ["MemoryEngine", "EngineRegistry", "initialize_device"]
