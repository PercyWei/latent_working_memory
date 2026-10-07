from copy import deepcopy
from datetime import timedelta
import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from latent_working_memory.v3.engine import TokenMemoryEngine, initialize_device
from latent_working_memory.v3.config import TrainingConfig
from latent_working_memory.v4 import engine as engine_module


@pytest.mark.parametrize("rank", range(4))
def test_four_gpu_local_rank_maps_to_logical_cuda_device(monkeypatch, rank):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "4")
    devices, groups = [], []
    monkeypatch.setattr(torch.cuda, "set_device", devices.append)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        engine_module,
        "initialize_global_process_group",
        lambda **kwargs: groups.append(kwargs),
    )

    assert initialize_device("cuda") == torch.device("cuda", rank)
    assert devices == [torch.device("cuda", rank)]
    assert groups == [{"timeout_second": 7200}]


@pytest.mark.parametrize("visible,requested", [("2,4", "cuda:1"), ("7,4", "cuda:1"), ("6", "cuda")])
def test_gpu_subset_uses_logical_indices_without_torchrun(monkeypatch, visible, requested):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    devices = []
    monkeypatch.setattr(torch.cuda, "set_device", devices.append)

    result = initialize_device(requested)
    assert result == torch.device("cuda", torch.device(requested).index or 0)
    assert devices == [result]


@pytest.mark.parametrize("visible", ["", "-1", "4,4", "4,04", "4,other", "4,"])
def test_device_initialization_rejects_invalid_or_duplicate_gpus(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(RuntimeError, match="distinct non-negative physical GPU indices"):
        initialize_device("cuda")


def test_device_initialization_rejects_a_rank_outside_visible_subset(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    monkeypatch.setenv("LOCAL_RANK", "2")
    with pytest.raises(ValueError, match="logical CUDA device"):
        initialize_device("cuda")


def test_cpu_initialization_does_not_require_gpu_selection(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert initialize_device("cpu") == torch.device("cpu")


class BranchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.second = torch.nn.Parameter(torch.tensor(-0.2, dtype=torch.float64))
        self.frozen = torch.nn.Parameter(
            torch.tensor(0.1, dtype=torch.float64), requires_grad=False
        )
        self.calls = []

    def forward(self, example, epoch=0, differentiable=True, batched=False):
        self.calls.append((epoch, differentiable, torch.is_grad_enabled(), self.training))
        rows = example if batched else [example]
        parameters = torch.stack(
            [self.first if row["branch"] == 0 else self.second for row in rows]
        )
        x = self.first.new_tensor([row["x"] for row in rows])
        y = self.first.new_tensor([row["y"] for row in rows])
        loss = (parameters * x + self.frozen - y).square().mean()
        return {
            "loss": loss,
            "metrics": {
                "length": sum(row["length"] for row in rows) / len(rows),
                "epoch": float(epoch),
            },
        }


def config(micro_batch_size=1):
    return TrainingConfig(
        dataset_dir="unused",
        output_dir="unused",
        micro_batch_size_per_gpu=micro_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=0.05,
        weight_decay=0.0,
        gradient_clip=100.0,
    )


def examples():
    return [
        {"branch": 0, "x": 2.0, "y": 0.0, "length": 1},
        {"branch": 1, "x": 3.0, "y": 1.0, "length": 17},
        {"branch": 0, "x": -1.0, "y": 0.5, "length": 3},
    ]


def reference_step(model, optimizer, batch, epoch=0):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.stack([model(row, epoch=epoch)["loss"] for row in batch]).mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], 100.0
    )
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(norm)}


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
def test_single_device_matches_global_trajectory_mean_and_one_optimizer_step(micro_batch_size):
    model = BranchModel()
    reference = deepcopy(model)
    engine = TokenMemoryEngine(model, config(micro_batch_size), "cpu")
    engine.initialize()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in reference.parameters() if parameter.requires_grad],
        lr=0.05,
        weight_decay=0.0,
    )

    expected = reference_step(reference, optimizer, examples(), epoch=2)
    observed = engine.step(examples(), epoch=2)

    assert observed["loss"] == pytest.approx(expected["loss"])
    assert observed["grad_norm"] == pytest.approx(expected["grad_norm"])
    assert observed["samples"] == 3
    assert observed["length"] == 7.0
    assert observed["epoch"] == 2.0
    assert model.calls == [(2, True, True, True)] * math.ceil(3 / micro_batch_size)
    assert model.frozen.grad is None
    assert len(engine.parameters) == 2
    for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, wanted)
    for state in engine.optimizer.state.values():
        assert state["step"].item() == 1


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
def test_evaluation_preserves_parameters_optimizer_gradients_and_training_mode(micro_batch_size):
    model = BranchModel()
    engine = TokenMemoryEngine(model, config(micro_batch_size), "cpu")
    engine.initialize()
    engine.step(examples())
    before = [parameter.detach().clone() for parameter in model.parameters()]
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    expected = sum(float(model(row)["loss"].detach()) for row in examples()) / len(examples())
    optimizer_steps = [state["step"].item() for state in engine.optimizer.state.values()]
    model.calls.clear()

    observed = engine.eval_batch(examples(), epoch=4)

    assert observed["loss"] == pytest.approx(expected)
    assert observed["samples"] == 3
    assert observed["grad_norm"] is None
    assert model.calls == [(4, False, False, False)] * math.ceil(3 / micro_batch_size)
    assert model.training
    for p, value, grad in zip(model.parameters(), before, gradients, strict=True):
        torch.testing.assert_close(p, value)
        if grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, grad)
    assert [state["step"].item() for state in engine.optimizer.state.values()] == optimizer_steps


def test_empty_batches_are_rejected():
    engine = TokenMemoryEngine(BranchModel(), config(), "cpu")
    engine.initialize()
    for operation in (engine.step, engine.eval_batch):
        with pytest.raises(ValueError, match="at least one trajectory"):
            operation([])


def _distributed_worker(rank, rendezvous, world_size, micro_batch_size):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        model = BranchModel()
        reference = deepcopy(model)
        engine = TokenMemoryEngine(model, config(micro_batch_size), "cpu")
        engine.initialize()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in reference.parameters() if parameter.requires_grad],
            lr=0.05,
            weight_decay=0.0,
        )
        batches = [(examples() * 3)[:8], examples(), examples()[:1], examples()[1:]]
        for epoch, batch in enumerate(batches):
            expected = reference_step(reference, optimizer, batch, epoch=epoch)
            calls_before = len(model.calls)
            observed = engine.step(batch, epoch=epoch)
            assert observed["loss"] == pytest.approx(expected["loss"])
            assert observed["grad_norm"] == pytest.approx(expected["grad_norm"])
            assert observed["samples"] == len(batch)
            assert observed["length"] == pytest.approx(
                sum(row["length"] for row in batch) / len(batch)
            )
            expected_calls = max(1, math.ceil(len(batch[rank::world_size]) / micro_batch_size))
            assert len(model.calls) - calls_before == expected_calls
            for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual, wanted, rtol=1e-12, atol=1e-12)
                if wanted.grad is None:
                    assert actual.grad is None
                else:
                    torch.testing.assert_close(actual.grad, wanted.grad, rtol=1e-12, atol=1e-12)
            expected_eval = sum(float(reference(row)["loss"].detach()) for row in batch) / len(
                batch
            )
            evaluation = engine.eval_batch(batch, epoch=epoch)
            assert evaluation["loss"] == pytest.approx(expected_eval)
            assert evaluation["samples"] == len(batch)
            assert evaluation["grad_norm"] is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
def test_distributed_branch_updates_full_uneven_and_empty_tail_batches_match_serial(
    tmp_path, world_size, micro_batch_size
):
    mp.spawn(
        _distributed_worker,
        args=(str(tmp_path / "rendezvous"), world_size, micro_batch_size),
        nprocs=world_size,
        join=True,
    )
