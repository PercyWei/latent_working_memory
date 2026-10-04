import math

import pytest
import torch

from latent_working_memory.v4.memory import LayerMemory
from latent_working_memory.v4.config import ModelConfig


def make_memory(query_mode="conditioned", inner_steps=2, **query_settings):
    torch.manual_seed(31)
    config = ModelConfig(
        "tiny",
        memory_dim=4,
        query_dim=3,
        value_dim=3,
        num_slots=3,
        num_probes=4,
        query_mode=query_mode,
        inner_steps=inner_steps,
        inner_lr=0.2,
        correction_scale=0.1,
        **query_settings,
    )
    return LayerMemory(5, 6, config).double()


def outer_loss(module, hidden, future_queries, target):
    source = module.source(hidden)
    state = module.initial_state(hidden.shape[0], differentiable=True)
    updated, _ = module.write(state, source, differentiable=True)
    return (module.read(future_queries, updated) - target).square().sum()


@pytest.mark.parametrize("query_mode", ["fixed", "conditioned"])
def test_fixed_norm_uses_separate_read_and_compression_scales(query_mode, monkeypatch):
    module = make_memory(
        query_mode,
        query_normalization="fixed_norm",
        read_query_norm=2.5,
        compression_query_norm=4.0,
    )
    hidden = torch.randn(2, 5, 5, dtype=torch.float64)
    source = module.source(hidden)
    query = module.queries(source)
    torch.testing.assert_close(query.norm(dim=-1), query.new_full((2, 4), 4.0))
    captured = []
    original_read = module.read

    def capture_read(query, bank):
        captured.append(query)
        return original_read(query, bank)

    monkeypatch.setattr(module, "read", capture_read)
    module.corrections(hidden, module.initial_state(2, differentiable=False))
    torch.testing.assert_close(captured[0].norm(dim=-1), query.new_full((2, 5), 2.5))


@pytest.mark.parametrize("query_normalization", ["none", "fixed_norm"])
def test_query_length_is_free_only_without_normalization(query_normalization, monkeypatch):
    module = make_memory("fixed", query_normalization=query_normalization)
    hidden = torch.randn(1, 5, 5, dtype=torch.float64)
    source = module.source(hidden)
    memory = module.initial_state(1, differentiable=False)
    captured = []
    original_read = module.read

    def capture_read(query, bank):
        captured.append(query)
        return original_read(query, bank)

    monkeypatch.setattr(module, "read", capture_read)
    compression_before = module.queries(source).clone()
    module.corrections(hidden, memory)
    with torch.no_grad():
        module.probe_seeds.mul_(2)
        module.runtime_query.weight.mul_(2)
    compression_after = module.queries(source)
    module.corrections(hidden, memory)
    factor = 2 if query_normalization == "none" else 1
    torch.testing.assert_close(compression_after, factor * compression_before)
    torch.testing.assert_close(captured[1], factor * captured[0])


@pytest.mark.parametrize("query_mode", ["fixed", "conditioned"])
@pytest.mark.parametrize("query_normalization", ["none", "fixed_norm"])
def test_meta_gradient_matches_finite_difference(query_mode, query_normalization):
    module = make_memory(query_mode, query_normalization=query_normalization)
    hidden = torch.randn(1, 5, 5, dtype=torch.float64)
    future_queries = torch.randn(1, 2, 3, dtype=torch.float64)
    target = torch.randn(1, 2, 3, dtype=torch.float64)
    loss = outer_loss(module, hidden, future_queries, target)
    gradient = torch.autograd.grad(loss, module.probe_seeds)[0]
    index = tuple(torch.unravel_index(gradient.abs().argmax(), gradient.shape))
    assert gradient[index].abs() > 1e-6
    original = module.probe_seeds[index].item()
    epsilon = 1e-5
    with torch.no_grad():
        module.probe_seeds[index] = original + epsilon
    positive = outer_loss(module, hidden, future_queries, target).item()
    with torch.no_grad():
        module.probe_seeds[index] = original - epsilon
    negative = outer_loss(module, hidden, future_queries, target).item()
    with torch.no_grad():
        module.probe_seeds[index] = original
    finite_difference = (positive - negative) / (2 * epsilon)
    torch.testing.assert_close(
        gradient[index], gradient.new_tensor(finite_difference), rtol=1e-5, atol=1e-9
    )


def test_teacher_query_path_is_retained():
    module = make_memory(inner_steps=1)
    hidden = torch.randn(1, 5, 5, dtype=torch.float64)
    source = module.source(hidden)
    memory = module.initial_state(1, differentiable=True)
    future_queries = torch.randn(1, 2, 3, dtype=torch.float64)
    future_target = torch.randn(1, 2, 3, dtype=torch.float64)

    updated, _ = module.write(memory, source, differentiable=True)
    actual_loss = (module.read(future_queries, updated) - future_target).square().sum()
    actual = torch.autograd.grad(actual_loss, module.probe_seeds, retain_graph=True)[0]

    query = module.queries(source)
    target = module.read(query, source)
    state = memory.clone()
    per_sample_loss = 0.5 * (module.read(query, state) - target).square().sum(-1).mean(-1)
    step = torch.autograd.grad(per_sample_loss.sum(), state, create_graph=True)[0]
    expected_state = state - module.inner_lr * step
    expected_loss = (module.read(future_queries, expected_state) - future_target).square().sum()
    expected = torch.autograd.grad(expected_loss, module.probe_seeds, retain_graph=True)[0]
    torch.testing.assert_close(actual, expected)

    detached_loss = 0.5 * (module.read(query, state) - target.detach()).square().sum(-1).mean(-1)
    detached_step = torch.autograd.grad(detached_loss.sum(), state, create_graph=True)[0]
    detached_state = state - module.inner_lr * detached_step
    detached_outer = (module.read(future_queries, detached_state) - future_target).square().sum()
    detached_gradient = torch.autograd.grad(detached_outer, module.probe_seeds)[0]
    assert not torch.allclose(actual, detached_gradient, rtol=1e-3, atol=1e-7)


def test_conditioned_generator_and_source_receive_outer_gradients():
    module = make_memory()
    hidden = torch.randn(1, 5, 5, dtype=torch.float64)
    future_queries = torch.randn(1, 2, 3, dtype=torch.float64)
    target = torch.randn(1, 2, 3, dtype=torch.float64)
    loss = outer_loss(module, hidden, future_queries, target)
    loss.backward()
    for parameter in (
        module.probe_seeds,
        module.probe_projection.weight,
        module.source_projection.weight,
        module.initial_slots,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0


def test_source_depending_on_old_memory_is_fixed_only_for_inner_partial():
    module = make_memory(inner_steps=1)
    old_memory = module.initial_state(1, differentiable=True)
    source = torch.cat((0.7 * old_memory + 0.4, old_memory.flip(1) - 0.3), dim=1)
    actual, _ = module.write(old_memory, source, differentiable=True)

    # 独立解析梯度：只对 bank 求导，Q/Y 不参与内循环偏导；所有张量仍保留外层图。
    query = module.queries(source)
    target = module.read(query, source)
    keys = module.key_projection(old_memory)
    values = module.value_projection(old_memory)
    attention = (query @ keys.transpose(-1, -2) / math.sqrt(module.query_dim)).softmax(-1)
    response = attention @ values
    d_response = (response - target) / query.shape[1]
    d_values = attention.transpose(-1, -2) @ d_response
    d_attention = d_response @ values.transpose(-1, -2)
    d_scores = attention * (d_attention - (attention * d_attention).sum(-1, keepdim=True))
    d_keys = d_scores.transpose(-1, -2) @ query / math.sqrt(module.query_dim)
    inner_gradient = (
        d_keys @ module.key_projection.weight + d_values @ module.value_projection.weight
    )
    expected = old_memory - module.inner_lr * inner_gradient
    torch.testing.assert_close(actual, expected)

    outer_target = torch.randn_like(old_memory)
    actual_outer = (actual - outer_target).square().sum()
    expected_outer = (expected - outer_target).square().sum()
    actual_gradient = torch.autograd.grad(actual_outer, old_memory, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected_outer, old_memory)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_eval_write_inside_no_grad_preserves_parameter_grads():
    module = make_memory().float().eval()
    hidden = torch.randn(2, 6, 5)
    saved_grads = {}
    for index, parameter in enumerate(module.parameters()):
        if index % 2:
            parameter.grad = torch.randn_like(parameter)
        saved_grads[parameter] = None if parameter.grad is None else parameter.grad.clone()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        source = module.source(hidden)
        memory = module.initial_state(2, differentiable=False)
        initial_copy = memory.clone()
        updated, metrics = module.write(memory, source, differentiable=False)
        delta_q, delta_o = module.corrections(hidden, updated)
    assert source.dtype == updated.dtype == torch.float32
    assert not updated.requires_grad and updated.grad_fn is None
    assert not torch.equal(updated, memory)
    torch.testing.assert_close(memory, initial_copy)
    assert delta_q.shape == (2, 6, 6)
    assert delta_o.shape == (2, 6, 5)
    assert metrics["write_loss_after"] < metrics["write_loss_before"]
    assert all(value.ndim == 0 and not value.requires_grad for value in metrics.values())
    for parameter, before in saved_grads.items():
        if before is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, before)


def test_slots_are_distinct_and_writes_are_batch_independent():
    module = make_memory()
    initial = module.initial_state(2, differentiable=False)
    assert not torch.equal(initial[0, 0], initial[0, 1])
    torch.testing.assert_close(initial[0], initial[1])
    hidden = torch.randn(2, 5, 5, dtype=torch.float64)
    source = module.source(hidden)
    batched, _ = module.write(initial, source, differentiable=False)
    individual = [
        module.write(initial[i : i + 1], source[i : i + 1], differentiable=False)[0]
        for i in range(2)
    ]
    torch.testing.assert_close(batched, torch.cat(individual))
