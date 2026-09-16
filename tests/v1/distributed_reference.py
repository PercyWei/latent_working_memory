from __future__ import annotations

import torch
import torch.distributed as dist


def synchronize_gradients(parameters):
    """Sum globally normalized local gradients; preserve globally unused parameters."""
    used = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    active = [p for p, present in zip(parameters, used.tolist(), strict=True) if present]
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in active])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for p in active:
        p.grad = flat[offset:offset + p.numel()].view_as(p)
        offset += p.numel()

