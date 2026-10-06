from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from latent_working_memory.v3.engine import TokenMemoryEngine


class BranchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.second = torch.nn.Parameter(torch.tensor(-0.2, dtype=torch.float64))
        self.frozen = torch.nn.Parameter(
            torch.tensor(0.1, dtype=torch.float64), requires_grad=False
        )
        self.calls = []

    def forward(self, example, epoch=0, differentiable=True):
        self.calls.append((epoch, differentiable, torch.is_grad_enabled(), self.training))
        parameter = self.first if example["branch"] == 0 else self.second
        loss = (parameter * example["x"] + self.frozen - example["y"]).square()
        return {
            "loss": loss,
            "metrics": {"length": float(example["length"]), "epoch": float(epoch)},
        }


def config():
    return SimpleNamespace(
        global_batch_size=4, learning_rate=0.05, weight_decay=0.0, gradient_clip=100.0
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


def test_single_device_matches_global_trajectory_mean_and_one_optimizer_step():
    model = BranchModel()
    reference = deepcopy(model)
    engine = TokenMemoryEngine(model, config(), "cpu")
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
    assert model.calls == [(2, True, True, True)] * 3
    assert model.frozen.grad is None
    assert len(engine.parameters) == 2
    for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, wanted)
    for state in engine.optimizer.state.values():
        assert state["step"].item() == 1


def test_evaluation_preserves_parameters_optimizer_gradients_and_training_mode():
    model = BranchModel()
    engine = TokenMemoryEngine(model, config(), "cpu")
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
    assert model.calls == [(4, False, False, False)] * 3
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


def _distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        model = BranchModel()
        reference = deepcopy(model)
        engine = TokenMemoryEngine(model, config(), "cpu")
        engine.initialize()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in reference.parameters() if parameter.requires_grad],
            lr=0.05,
            weight_decay=0.0,
        )
        batches = [examples(), examples()[:1], examples()[1:]]
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
            expected_calls = max(1, len(batch[rank::2]))
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


def test_two_rank_branch_updates_uneven_batch_and_empty_tail_match_serial(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
