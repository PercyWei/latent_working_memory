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
    assert torch.equal(
        positions[1:],
        sinusoidal_positions(2, 7, "cpu", torch.float32, start=1),
    )


def test_writer_shapes_seen_tokens_and_non_mutation() -> None:
    writer = _writer()
    initial = MemoryState(torch.zeros(2, 8, dtype=writer.old_type.dtype), 0)
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
    state = MemoryState(torch.zeros(2, 8, dtype=writer.old_type.dtype), 0)
    output = writer(state, torch.randn(3, 8, dtype=torch.bfloat16), 8)
    assert state.values.dtype == torch.bfloat16
    assert output.values.dtype == torch.bfloat16


def test_writer_keeps_fp32_parameters_with_bfloat16_autocast_state() -> None:
    writer = _writer()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        state = MemoryState(torch.zeros(2, 8, dtype=torch.bfloat16), 0)
        output = writer(state, torch.randn(3, 8, dtype=torch.bfloat16), 0)
    output.values.float().sum().backward()

    assert writer.old_type.dtype == torch.float32
    assert state.values.dtype == torch.bfloat16
    assert output.values.dtype == torch.bfloat16
    assert writer.output_projection.weight.grad is not None
    assert writer.output_projection.weight.grad.dtype == torch.float32


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
    assert writer.blocks[0].cross_attention.in_proj_weight.grad is not None


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


def test_empty_first_allocation_and_batch_masks():
    writer = _writer()
    empty = writer.initialize_state()
    assert empty.num_slots == 0 and empty.seen_tokens == 0
    assert not any("seed" in name or "birth" in name for name, _ in writer.named_parameters())
    features = [torch.randn(3, 8, requires_grad=True), torch.randn(11, 8, requires_grad=True)]
    batched = writer.update_batch([empty, empty], features, [0, 0], [3, 7])
    individual = [writer(empty, f, first_slots=k) for f, k in zip(features, (3, 7))]
    for first, second in zip(batched, individual):
        torch.testing.assert_close(first.values, second.values, rtol=1e-5, atol=1e-7)
    grads = torch.autograd.grad(batched[0].values.square().sum(), features, allow_unused=True)
    assert grads[0].abs().sum() > 0
    assert grads[1].abs().sum() == 0
    with torch.no_grad():
        writer.output_projection.weight.zero_()
        writer.output_projection.bias.zero_()
    assert torch.count_nonzero(writer(empty, features[0], first_slots=5).values) == 0
