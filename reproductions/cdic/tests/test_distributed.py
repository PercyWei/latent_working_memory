from __future__ import annotations

import pytest
import torch

from cdic_repro.experiments.distributed import DistributedContext, initialize_distributed


def test_distributed_gradient_average_materializes_missing_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_all_reduce(tensor: torch.Tensor, op: object) -> None:
        assert op is torch.distributed.ReduceOp.SUM
        tensor.mul_(2.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device="cuda:0",
    )
    first = torch.nn.Parameter(torch.tensor([1.0]))
    first.grad = torch.tensor([3.0])
    second = torch.nn.Parameter(torch.tensor([2.0]))

    context.average_gradients((first, second), active_workers=2)

    assert first.grad is not None and first.grad.item() == 3.0
    assert second.grad is not None and second.grad.item() == 0.0


def test_multiple_devices_require_matching_torchrun_world(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    with pytest.raises(RuntimeError, match="torchrun"):
        initialize_distributed(
            primary_device="cuda:0",
            devices=("cuda:0", "cuda:1"),
        )


def test_single_worker_gather_objects_returns_the_local_value() -> None:
    context = DistributedContext(rank=0, local_rank=0, world_size=1, device="cuda:0")
    value = {"loss": 2.0}

    assert context.gather_objects(value) == (value,)
