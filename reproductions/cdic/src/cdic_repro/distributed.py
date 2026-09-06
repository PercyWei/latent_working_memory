from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: str

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            torch.distributed.barrier()

    def broadcast_parameters(self, parameters: tuple[object, ...]) -> None:
        if not self.enabled:
            return
        for parameter in parameters:
            torch.distributed.broadcast(parameter.data, src=0)

    def average_gradients(
        self,
        parameters: tuple[object, ...],
        *,
        active_workers: int,
    ) -> None:
        if active_workers < 1 or active_workers > self.world_size:
            raise ValueError("active_workers must be within the distributed world size")
        if not self.enabled:
            return
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            torch.distributed.all_reduce(
                parameter.grad,
                op=torch.distributed.ReduceOp.SUM,
            )
            parameter.grad.div_(active_workers)

    def gather_rng_states(self) -> list[dict[str, object]]:
        local_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(torch.device(self.device)),
        }
        if not self.enabled:
            return [local_state]
        gathered: list[dict[str, object] | None] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, local_state)
        if any(state is None for state in gathered):
            raise RuntimeError("failed to gather RNG state from every distributed worker")
        return [state for state in gathered if state is not None]

    def close(self) -> None:
        if self.enabled and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def initialize_distributed(
    *,
    primary_device: str,
    devices: tuple[str, ...],
) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    configured_devices = devices or (primary_device,)

    if world_size != len(configured_devices):
        if len(configured_devices) > 1 and world_size == 1:
            raise RuntimeError("multiple devices require torchrun with matching --nproc-per-node")
        raise RuntimeError(
            f"distributed world size {world_size} does not match configured devices "
            f"{configured_devices}"
        )
    if not 0 <= rank < world_size or not 0 <= local_rank < world_size:
        raise RuntimeError("invalid distributed rank assignment")
    device = configured_devices[local_rank]
    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise ValueError("distributed C-DIC training requires CUDA devices")
    torch.cuda.set_device(parsed)
    if world_size > 1:
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        torch.distributed.init_process_group(backend="nccl")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )
