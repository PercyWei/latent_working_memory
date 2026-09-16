import pytest
import torch

from latent_working_memory.v2.working_memory import WorkingMemory, MemoryUpdater


def test_initialization_growth_limits_and_identity(model):
    system = WorkingMemory(model, MemoryUpdater(model.width, heads=2, layers=1, slot_limit=6))
    ids = torch.tensor([4, 5, 6, 7])
    first = system.write(None, ids, 2)
    expected = model.encode(ids[None], torch.ones_like(ids[None], dtype=torch.bool), 2)[0]
    torch.testing.assert_close(first.values, expected)
    second = system.write(first, ids.flip(0), 2, grow_by=2)
    torch.testing.assert_close(second.values[:2], first.values)
    assert second.values.shape == (4, model.width)
    assert (second.seen_tokens, second.updates) == (8, 1)
    with pytest.raises(ValueError, match="slot_limit"):
        system.write(second, ids, 2, grow_by=3)
    with pytest.raises(ValueError, match="first write"):
        system.write(None, ids, 2, grow_by=1)
    with pytest.raises(ValueError, match="current pooled"):
        system.write(first, ids, 2, grow_by=3)


def test_multiple_writes_backpropagate_and_detach_is_explicit(model):
    system = WorkingMemory(model, MemoryUpdater(model.width, heads=2, layers=1))
    first = system.write(None, torch.tensor([4, 5, 6, 7]), 2)
    second = system.write(first, torch.tensor([8, 9, 10]), 2, 1)
    second.values.retain_grad()
    third = system.write(second, torch.tensor([6, 5, 4]), 2)
    loss = model.read(
        [third.values], torch.tensor([[11]]), torch.tensor([[True]]), torch.tensor([[4, 2]])
    ).loss
    loss.backward()
    assert system.updater.delta.weight.grad.abs().sum() > 0
    assert second.values.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    detached = second.detached()
    assert not detached.values.requires_grad
    assert detached.seen_tokens == second.seen_tokens


def test_dynamic_forward_and_content_update(model):
    system = WorkingMemory(model, MemoryUpdater(model.width, heads=2, layers=1))
    optimizer = torch.optim.AdamW(system.updater.parameters(), lr=0.01)
    segments = [torch.tensor([4, 5, 6, 7]), torch.tensor([8, 9, 10])]
    output = system(
        segments,
        [2, 2],
        [0, 1],
        torch.tensor([[11]]),
        torch.tensor([[True]]),
        torch.tensor([[4, 2]]),
    )
    output.loss.backward()
    optimizer.step()
    first = system.write(None, segments[0], 2)
    second = system.write(first, segments[1], 2)
    assert not torch.equal(first.values, second.values)
