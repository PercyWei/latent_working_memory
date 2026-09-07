from __future__ import annotations

import pytest
import torch

from latent_working_memory.v1.state import MemoryState, choose_growth, legal_growth_actions


def test_memory_state_contract() -> None:
    state = MemoryState(torch.zeros(2, 4), seen_tokens=3)
    assert state.num_slots == 2
    assert state.width == 4
    assert state.detached().values.grad_fn is None

    with pytest.raises(ValueError, match="finite"):
        MemoryState(torch.tensor([[float("nan")]]), seen_tokens=0)
    with pytest.raises(ValueError, match="non-negative"):
        MemoryState(torch.zeros(1, 1), seen_tokens=-1)


def test_growth_legality_and_tie_breaking() -> None:
    actions = (0, 8, 16)
    assert legal_growth_actions(8, actions, 16) == (0, 8)
    assert choose_growth(torch.tensor([0.0, -1.0, -9.0]), actions, 8, 16) == 8
    assert choose_growth(torch.tensor([-1.0, -1.0, 5.0]), actions, 8, 32) == 0


def test_growth_rejects_nonfinite_legal_costs() -> None:
    with pytest.raises(ValueError, match="finite"):
        choose_growth(torch.tensor([0.0, float("nan"), 1.0]), (0, 8, 16), 8, 32)
