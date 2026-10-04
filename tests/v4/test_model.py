import pytest
import torch
from torch.nn import functional as F


MODEL_TYPES = ("llama", "qwen2", "qwen3")


def tokens(length):
    return (torch.arange(length, dtype=torch.long) * 7 + 3) % 41


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_before_first_write_matches_frozen_backbone(model_factory, model_type):
    model = model_factory(model_type)
    model.train()
    assert not model.backbone.training
    assert model.memory_layers.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    ids = tokens(model.config.pending_size + model.config.recent_size - 1)
    with torch.no_grad():
        expected = model.backbone(ids[None], use_cache=True).logits[0]
        actual, state = model.consume(ids)
    torch.testing.assert_close(actual, expected)
    assert state.write_events == 0
    for layer, memory in zip(model.memory_layers, state.memories, strict=True):
        torch.testing.assert_close(memory, layer.initial_state(1, differentiable=False))


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_chunked_and_token_streams_match_across_writes(model_factory, model_type):
    model = model_factory(model_type)
    ids = tokens(19)
    whole_logits, whole_state = model.consume(ids)
    state = None
    pieces = []
    for token in ids:
        logits, state = model.consume(token.reshape(1), state)
        pieces.append(logits)
    torch.testing.assert_close(torch.cat(pieces), whole_logits, rtol=1e-5, atol=1e-6)
    assert state.position == whole_state.position == len(ids)
    assert state.write_events == whole_state.write_events == 5
    assert state.peak_live_tokens == whole_state.peak_live_tokens == 5
    for index in range(len(model.memory_layers)):
        torch.testing.assert_close(
            state.memories[index], whole_state.memories[index], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            state.sources[index], whole_state.sources[index], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            state.cache.layers[index].keys,
            whole_state.cache.layers[index].keys,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            state.cache.layers[index].values,
            whole_state.cache.layers[index].values,
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_positions_and_retained_cache_at_write_boundaries(model_factory, monkeypatch, model_type):
    model = model_factory(model_type)
    absolute_positions = []
    physical_positions = []
    write_positions = []
    original_write = model._write
    pending, recent = model.config.pending_size, model.config.recent_size

    def capture_positions(module, args, kwargs):
        absolute_positions.append(kwargs["position_ids"][0].clone())
        physical_positions.append(kwargs["cache_position"].clone())

    def capture_write(state, differentiable):
        assert state.cache.get_seq_length() == pending + recent
        retained = [
            (
                layer.keys[..., pending:, :].detach().clone(),
                layer.values[..., pending:, :].detach().clone(),
            )
            for layer in state.cache.layers
        ]
        write_positions.append(state.position)
        original_write(state, differentiable)
        assert state.cache.get_seq_length() == recent
        for layer, (expected_keys, expected_values) in zip(
            state.cache.layers, retained, strict=True
        ):
            # 写入只能删除前缀，retained KV 的数值不能重算或改变。
            assert torch.equal(layer.keys, expected_keys)
            assert torch.equal(layer.values, expected_values)

    monkeypatch.setattr(model, "_write", capture_write)
    hook = model.backbone.register_forward_pre_hook(capture_positions, with_kwargs=True)
    _, state = model.consume(tokens(17))
    hook.remove()
    torch.testing.assert_close(torch.cat(absolute_positions), torch.arange(17))
    torch.testing.assert_close(physical_positions[0], torch.arange(pending + recent))
    for positions in physical_positions[1:]:
        torch.testing.assert_close(positions, torch.arange(recent, recent + len(positions)))
    assert write_positions == [5, 8, 11, 14, 17]
    assert state.peak_live_tokens <= pending + recent


@pytest.mark.parametrize("model_type", MODEL_TYPES)
@pytest.mark.parametrize("query_normalization", ["none", "fixed_norm"])
def test_full_lm_outer_gradient_and_inference_numerics(
    model_factory, model_type, query_normalization
):
    model = model_factory(model_type, query_normalization=query_normalization)
    ids = tokens(17)
    with torch.no_grad():
        inference = model(ids, differentiable=False)
    training = model(ids, differentiable=True)
    torch.testing.assert_close(training["loss"], inference["loss"])
    assert (
        training["target_tokens"]
        == len(ids) - model.config.pending_size - model.config.recent_size - 1
    )
    assert training["write_events"] == inference["write_events"] == 4
    assert training["peak_live_tokens"] == 5
    assert not inference["loss"].requires_grad
    training["loss"].backward()
    for layer in model.memory_layers:
        for parameter in (
            layer.probe_seeds,
            layer.probe_projection.weight,
            layer.source_projection.weight,
            layer.initial_slots,
            layer.query_correction.weight,
            layer.output_correction.weight,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


@pytest.mark.parametrize("query_normalization", ["none", "fixed_norm"])
def test_full_lm_probe_gradient_matches_finite_difference(model_factory, query_normalization):
    model = model_factory(query_normalization=query_normalization).double()
    ids = tokens(14)
    parameter = model.memory_layers[0].probe_seeds
    loss = model(ids, differentiable=True)["loss"]
    assert loss.dtype == torch.float64
    gradient = torch.autograd.grad(loss, parameter)[0]
    index = tuple(torch.unravel_index(gradient.abs().argmax(), gradient.shape))
    assert gradient[index].abs() > 1e-8
    original = parameter[index].item()
    # HF 的 eager attention/RMSNorm 内部仍转 fp32；过小差分步长会放大量化误差。
    # 两种查询模式的 eps 扫描选取 2e-2；纯局部 reader 使用更严格的 double 检查。
    epsilon = 2e-2
    with torch.no_grad():
        parameter[index] = original + epsilon
        positive = model(ids, differentiable=False)["loss"].item()
        parameter[index] = original - epsilon
        negative = model(ids, differentiable=False)["loss"].item()
        parameter[index] = original
    numerical = (positive - negative) / (2 * epsilon)
    torch.testing.assert_close(
        gradient[index], gradient.new_tensor(numerical), rtol=1e-3, atol=1e-9
    )


def test_outer_objective_is_only_post_write_next_token_cross_entropy(model_factory):
    model = model_factory()
    ids = tokens(17)
    output = model(ids)
    logits, _ = model.consume(ids[:-1], differentiable=True)
    boundary = model.config.pending_size + model.config.recent_size
    expected = F.cross_entropy(logits[boundary:], ids[boundary + 1 :])
    assert output["write_loss"] > 1e-4
    torch.testing.assert_close(output["loss"], expected)


def test_bfloat16_backbone_uses_fp32_memory_for_training_and_inference(model_factory):
    model = model_factory(backbone_dtype="bfloat16")
    model.backbone.bfloat16()
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.backbone.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.memory_layers.parameters())
    ids = tokens(17)
    output = model(ids, differentiable=True)
    assert output["loss"].dtype == torch.float32
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    for layer in model.memory_layers:
        for parameter in (layer.probe_seeds, layer.source_projection.weight, layer.initial_slots):
            assert parameter.grad is not None
            assert parameter.grad.dtype == torch.float32
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    with torch.no_grad():
        inference = model(ids, differentiable=False)
        logits, state = model.consume(ids)
    torch.testing.assert_close(inference["loss"], output["loss"])
    assert not inference["loss"].requires_grad
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()
    assert all(
        memory.dtype == torch.float32 and not memory.requires_grad for memory in state.memories
    )


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_future_tokens_do_not_change_earlier_logits(model_factory, model_type):
    model = model_factory(model_type)
    original = tokens(17)
    changed = original.clone()
    # 切点在一次 backbone forward 的中间，而不是压缩边界。
    changed[9:] = (changed[9:] + 11) % 41
    original_logits, _ = model.consume(original)
    changed_logits, _ = model.consume(changed)
    torch.testing.assert_close(original_logits[:9], changed_logits[:9], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(original_logits[9:], changed_logits[9:])


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_greedy_generation_matches_continued_consumption(model_factory, model_type):
    model = model_factory(model_type)
    prompt = tokens(6)
    generated = model.generate(prompt, max_new_tokens=7)
    logits, state = model.consume(prompt)
    expected = [prompt]
    for step in range(7):
        next_token = logits[-1].argmax().reshape(1)
        expected.append(next_token)
        if step < 6:
            logits, state = model.consume(next_token, state)
    assert torch.equal(generated, torch.cat(expected))
    assert state.write_events >= 3


def test_trainable_checkpoint_excludes_backbone_and_roundtrips(model_factory, tmp_path):
    model = model_factory()
    with torch.no_grad():
        model.memory_layers[0].probe_seeds.add_(0.17)
    saved = model.trainable_state_dict()
    assert set(saved) == set(model.memory_layers.state_dict())
    assert all(not value.requires_grad and value.device.type == "cpu" for value in saved.values())
    assert all("backbone" not in name and "embed_tokens" not in name for name in saved)
    assert sum(value.numel() for value in saved.values()) < sum(
        value.numel() for value in model.backbone.state_dict().values()
    )
    path = tmp_path / "memory.pt"
    torch.save(saved, path)
    restored = model_factory()
    restored.load_trainable_state_dict(torch.load(path, weights_only=True))
    expected_logits, expected_state = model.consume(tokens(14))
    actual_logits, actual_state = restored.consume(tokens(14))
    torch.testing.assert_close(actual_logits, expected_logits)
    for actual, expected in zip(actual_state.memories, expected_state.memories, strict=True):
        torch.testing.assert_close(actual, expected)
