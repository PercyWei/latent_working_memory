from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MemoryState:
    values: Tensor
    seen_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.values, Tensor):
            raise TypeError("values must be a torch.Tensor")
        if self.values.ndim != 2:
            raise ValueError("values must have shape [K, d_mem]")
        if self.values.shape[0] <= 0 or self.values.shape[1] <= 0:
            raise ValueError("values must have non-zero slot and feature dimensions")
        if not self.values.is_floating_point():
            raise TypeError("values must have a floating-point dtype")
        if not bool(torch.isfinite(self.values.detach()).all()):
            raise ValueError("values must contain only finite values")
        if type(self.seen_tokens) is not int or self.seen_tokens < 0:
            raise ValueError("seen_tokens must be a non-negative integer")

    @property
    def num_slots(self) -> int:
        return self.values.shape[0]

    @property
    def width(self) -> int:
        return self.values.shape[1]

    def detached(self) -> MemoryState:
        return MemoryState(self.values.detach(), self.seen_tokens)


def legal_growth_actions(
    num_slots: int,
    growth_actions: Sequence[int],
    slot_limit: int,
) -> tuple[int, ...]:
    _validate_action_contract(num_slots, growth_actions, slot_limit)
    return tuple(action for action in growth_actions if num_slots + action <= slot_limit)


def choose_growth(
    predicted_cost_deltas: Tensor,
    growth_actions: Sequence[int],
    num_slots: int,
    slot_limit: int,
) -> int:
    if predicted_cost_deltas.ndim != 1:
        raise ValueError("predicted_cost_deltas must have shape [num_actions]")
    if predicted_cost_deltas.shape[0] != len(growth_actions):
        raise ValueError("predicted_cost_deltas must align with growth_actions")

    legal_actions = legal_growth_actions(num_slots, growth_actions, slot_limit)
    action_to_index = {action: index for index, action in enumerate(growth_actions)}
    detached_costs = predicted_cost_deltas.detach().to(dtype=torch.float32, device="cpu")
    legal_costs = {
        action: float(detached_costs[action_to_index[action]].item()) for action in legal_actions
    }
    if not all(math.isfinite(cost) for cost in legal_costs.values()):
        raise ValueError("all legal predicted costs must be finite")
    return min(legal_actions, key=lambda action: (legal_costs[action], action))


def _validate_action_contract(
    num_slots: int,
    growth_actions: Sequence[int],
    slot_limit: int,
) -> None:
    if type(num_slots) is not int or num_slots <= 0:
        raise ValueError("num_slots must be a positive integer")
    if type(slot_limit) is not int or slot_limit <= 0:
        raise ValueError("slot_limit must be a positive integer")
    if num_slots > slot_limit:
        raise ValueError("num_slots must not exceed slot_limit")
    if not growth_actions:
        raise ValueError("growth_actions must not be empty")
    if any(type(action) is not int or action < 0 for action in growth_actions):
        raise ValueError("growth_actions must contain non-negative integers")
    if len(set(growth_actions)) != len(growth_actions):
        raise ValueError("growth_actions must be unique")
    if 0 not in growth_actions:
        raise ValueError("growth_actions must include 0")
