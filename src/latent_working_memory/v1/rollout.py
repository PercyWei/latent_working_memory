from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.v1.state import (
    MemoryState,
    choose_growth,
    legal_growth_actions,
)


@dataclass(frozen=True, slots=True)
class GrowthTrace:
    step: int
    prefix_start: int
    prefix_end: int
    slots_before: int
    slots_after: int
    predicted_cost_deltas: tuple[float, ...]
    legal_actions: tuple[int, ...]
    selected_action: int
    reached_limit: bool


def policy_update(
    writer: JointMemoryWriter,
    value_network: GrowthValueNetwork,
    state: MemoryState,
    features: Tensor,
    step: int,
) -> tuple[MemoryState, GrowthTrace]:
    if value_network.growth_actions != writer.growth_actions:
        raise ValueError("writer and value network must use the same growth actions")
    predicted = value_network.predict(state, features)
    if predicted.ndim != 1:
        raise ValueError("a single-state policy prediction must have shape [num_actions]")
    action = choose_growth(
        predicted,
        writer.growth_actions,
        state.num_slots,
        writer.slot_limit,
    )
    legal = legal_growth_actions(state.num_slots, writer.growth_actions, writer.slot_limit)
    next_state = writer(state, features, action)
    return next_state, GrowthTrace(
        step=step,
        prefix_start=state.seen_tokens,
        prefix_end=next_state.seen_tokens,
        slots_before=state.num_slots,
        slots_after=next_state.num_slots,
        predicted_cost_deltas=tuple(
            float(value) for value in predicted.detach().to(dtype=torch.float32, device="cpu")
        ),
        legal_actions=legal,
        selected_action=action,
        reached_limit=next_state.num_slots == writer.slot_limit,
    )


def rollout_with_policy(
    writer: JointMemoryWriter,
    value_network: GrowthValueNetwork,
    state: MemoryState,
    feature_chunks: Iterable[Tensor],
) -> tuple[MemoryState, tuple[GrowthTrace, ...]]:
    traces: list[GrowthTrace] = []
    current = state
    for step, features in enumerate(feature_chunks):
        current, trace = policy_update(writer, value_network, current, features, step)
        traces.append(trace)
    return current, tuple(traces)
