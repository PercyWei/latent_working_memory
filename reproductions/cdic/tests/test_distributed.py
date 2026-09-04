from __future__ import annotations

import pytest

from cdic_repro.distributed import DistributedContext, initialize_distributed


class FakeTensor:
    def __init__(self, value: float) -> None:
        self.value = value

    def div_(self, denominator: int) -> None:
        self.value /= denominator


class FakeParameter:
    def __init__(self, gradient: FakeTensor | None) -> None:
        self.grad = gradient


class FakeDistributed:
    class ReduceOp:
        SUM = "sum"

    def all_reduce(self, tensor: FakeTensor, *, op: str) -> None:
        assert op == self.ReduceOp.SUM
        tensor.value *= 2.0


class FakeCuda:
    def set_device(self, _device: object) -> None:
        raise AssertionError("set_device must not be reached for an invalid launch")


class FakeTorch:
    def __init__(self) -> None:
        self.distributed = FakeDistributed()
        self.cuda = FakeCuda()

    def zeros_like(self, _parameter: object) -> FakeTensor:
        return FakeTensor(0.0)

    def device(self, value: str) -> object:
        return type("Device", (), {"type": value.split(":", 1)[0]})()


def test_distributed_gradient_average_materializes_missing_gradients() -> None:
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device="cuda:0",
        torch=FakeTorch(),
    )
    first = FakeParameter(FakeTensor(3.0))
    second = FakeParameter(None)

    context.average_gradients((first, second), active_workers=2)

    assert first.grad is not None and first.grad.value == 3.0
    assert second.grad is not None and second.grad.value == 0.0


def test_multiple_devices_require_matching_torchrun_world(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    with pytest.raises(RuntimeError, match="torchrun"):
        initialize_distributed(
            FakeTorch(),
            primary_device="cuda:0",
            devices=("cuda:0", "cuda:1"),
        )
