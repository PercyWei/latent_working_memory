from __future__ import annotations

import torch

from latent_working_memory.v1.model import (
    GrowthValueNetwork,
    JointMemoryWriter,
    sinusoidal_positions,
)
from latent_working_memory.v1.rollout import policy_update, rollout_with_policy
from latent_working_memory.v1.state import MemoryState


def _writer() -> JointMemoryWriter:
    torch.manual_seed(11)
    return JointMemoryWriter(
        d_mem=8,
        num_layers=2,
        num_heads=2,
        ffn_dim=16,
        slot_limit=48,
    )


def test_sinusoidal_positions_are_deterministic_and_distinct() -> None:
    positions = sinusoidal_positions(3, 7, "cpu", torch.float32)
    assert positions.shape == (3, 7)
    assert torch.equal(positions, sinusoidal_positions(3, 7, "cpu", torch.float32))
    assert not torch.equal(positions[0], positions[1])


def test_writer_shapes_seen_tokens_and_non_mutation() -> None:
    writer = _writer()
    initial = writer.initialize_state(2)
    old_copy = initial.values.detach().clone()

    unchanged_size = writer(initial, torch.randn(3, 8), 0)
    grown = writer(initial, torch.randn(3, 8), 8)
    grown_more = writer(initial, torch.randn(3, 8), 16)

    assert unchanged_size.values.shape == (2, 8)
    assert grown.values.shape == (10, 8)
    assert grown_more.values.shape == (18, 8)
    assert grown.seen_tokens == 3
    assert torch.equal(initial.values.detach(), old_copy)


def test_writer_preserves_bfloat16_runtime_state() -> None:
    writer = _writer().to(dtype=torch.bfloat16)
    state = writer.initialize_state(2)
    output = writer(state, torch.randn(3, 8, dtype=torch.bfloat16), 8)
    assert state.values.dtype == torch.bfloat16
    assert output.values.dtype == torch.bfloat16


def test_new_and_old_outputs_depend_on_old_memory_and_current_features() -> None:
    writer = _writer()
    old_values = torch.randn(2, 8, requires_grad=True)
    features = torch.randn(3, 8, requires_grad=True)
    state = MemoryState(old_values, seen_tokens=0)
    output = writer(state, features, 8)

    new_from_old, new_from_features = torch.autograd.grad(
        output.values[2:].square().sum(),
        (old_values, features),
        retain_graph=True,
    )
    (old_from_features,) = torch.autograd.grad(
        output.values[:2].square().sum(),
        (features,),
        retain_graph=True,
    )
    assert torch.count_nonzero(new_from_old).item() > 0
    assert torch.count_nonzero(new_from_features).item() > 0
    assert torch.count_nonzero(old_from_features).item() > 0

    loss = output.values.square().sum()
    loss.backward()
    assert old_values.grad is not None and torch.count_nonzero(old_values.grad).item() > 0
    assert features.grad is not None and torch.count_nonzero(features.grad).item() > 0
    assert writer.birth_from_memory.weight.grad is not None


def test_growth_value_features_are_detached_and_policy_rollout_is_auditable() -> None:
    writer = _writer()
    value_network = GrowthValueNetwork(d_mem=8)
    with torch.no_grad():
        for parameter in value_network.parameters():
            parameter.zero_()
        value_network.network[-1].bias.copy_(torch.tensor([0.0, -1.0, 1.0]))

    state_values = torch.randn(2, 8, requires_grad=True)
    features = torch.randn(3, 8, requires_grad=True)
    state = MemoryState(state_values, seen_tokens=5)
    summarized = value_network.policy_features(state, features)
    predicted = value_network(summarized)
    assert summarized.shape == (35,)
    assert not summarized.requires_grad
    assert predicted.shape == (3,)
    assert predicted.requires_grad

    next_state, trace = policy_update(writer, value_network, state, features, step=4)
    assert trace.selected_action == 8
    assert trace.prefix_start == 5
    assert trace.prefix_end == 8
    assert trace.slots_after == 10
    assert next_state.num_slots == 10

    final_state, traces = rollout_with_policy(
        writer,
        value_network,
        state,
        (torch.randn(2, 8), torch.randn(1, 8)),
    )
    assert [item.selected_action for item in traces] == [8, 8]
    assert final_state.num_slots == 18
    assert final_state.seen_tokens == 8
