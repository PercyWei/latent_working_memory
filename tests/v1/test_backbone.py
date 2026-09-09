from __future__ import annotations

import torch
import pytest

from latent_working_memory.v1.backbone import ReadTokens


def test_whole_unit_features_preserve_context_and_ignore_reader_lora(components):
    backbone, _ = components
    backbone.eval()
    units = [(4, 5, 6, 7), (7, 6)]
    batch = backbone.text_features(units, [0, 9])
    for unit, start, row in zip(units, [0, 9], batch):
        torch.testing.assert_close(row, backbone.text_features([unit], [start])[0])
    changed_prefix = backbone.text_features([(8, 5, 6, 7)], [0])[0]
    assert not torch.allclose(batch[0][-1], changed_prefix[-1])
    with torch.no_grad():
        for name, parameter in backbone.language_model.named_parameters():
            if "lora_" in name:
                parameter.fill_(0.7)
    after = backbone.text_features(units, [0, 9])
    for first, second in zip(batch, after):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_reader_batch_mask_target_alignment_and_gradient_path(components):
    backbone, writer = components
    backbone.eval()
    units = [(4, 5, 6), (7, 6, 5, 4, 8)]
    features = backbone.text_features(units, [0, 0])
    states = writer.update_batch(
        [writer.initialize_state(), writer.initialize_state()], features, [0, 0], [2, 5]
    )
    tokens = [ReadTokens((11, 12), (4, 5, 2)), ReadTokens((12,), (7, 8, 9, 10, 2))]
    outputs = backbone.read_batch([s.values for s in states], tokens)
    for state, task, output in zip(states, tokens, outputs):
        single = backbone.read_batch([state.values], [task])[0]
        torch.testing.assert_close(output.target_logits, single.target_logits, atol=1e-6, rtol=1e-5)
    loss = sum(o.mean_nll for o in outputs)
    loss.backward()
    assert backbone.input_projection.weight.grad.abs().sum() > 0
    assert backbone.memory_projection.weight.grad.abs().sum() > 0
    assert writer.output_projection.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in backbone.language_model.named_parameters()
        if "lora_" in n
    )
    assert all(
        p.grad is None and not p.requires_grad
        for n, p in backbone.language_model.named_parameters()
        if "lora_" not in n
    )
    empty_output = backbone.read_batch([writer.initialize_state().values], [tokens[0]])[0]
    assert torch.isfinite(empty_output.mean_nll)


def test_target_token_is_predicted_before_it_is_seen(components):
    backbone, writer = components
    backbone.eval()
    memory = writer.initialize_state().values
    original = backbone.read_batch([memory], [ReadTokens((11,), (4, 5, 2))])[0]
    changed = backbone.read_batch([memory], [ReadTokens((11,), (8, 5, 2))])[0]
    torch.testing.assert_close(original.target_logits[0], changed.target_logits[0])
    assert not torch.allclose(original.target_logits[1], changed.target_logits[1])


def test_raw_context_controls_match_causal_model_and_restore_reader_lora(components):
    backbone, writer = components
    backbone.eval()
    empty = writer.initialize_state().values
    tasks = [ReadTokens((11, 12), (4, 5, 2)), ReadTokens((12,), (7, 8, 9, 2))]
    contexts = [(6, 7, 8, 9), (8, 9)]
    with torch.no_grad():
        for name, parameter in backbone.language_model.named_parameters():
            if "lora_" in name:
                parameter.normal_(std=0.3)
        for enabled in (True, False):
            outputs = backbone.read_batch([empty, empty], tasks, contexts, enabled)
            for task, context, output in zip(tasks, contexts, outputs):
                ids = torch.tensor([(1, *context, *task.prompt_ids, *task.target_ids)])
                if enabled:
                    direct = backbone.language_model(input_ids=ids).logits[0]
                else:
                    with backbone.language_model.disable_adapter():
                        direct = backbone.language_model(input_ids=ids).logits[0]
                start = len(context) + len(task.prompt_ids)
                torch.testing.assert_close(
                    output.target_logits,
                    direct[start : start + len(task.target_ids)],
                    rtol=1e-5,
                    atol=1e-6,
                )
        enabled = backbone.read_batch([empty], tasks[:1], contexts[:1])[0]
        disabled = backbone.read_batch([empty], tasks[:1], contexts[:1], False)[0]
        restored = backbone.read_batch([empty], tasks[:1], contexts[:1])[0]
        torch.testing.assert_close(enabled.target_logits, restored.target_logits, rtol=0, atol=0)
        assert not torch.allclose(enabled.target_logits, disabled.target_logits)
        changed = backbone.read_batch(
            [empty], [ReadTokens(tasks[0].prompt_ids, (8, 5, 2))], contexts[:1]
        )[0]
        torch.testing.assert_close(enabled.target_logits[0], changed.target_logits[0])
        assert not torch.allclose(enabled.target_logits[1], changed.target_logits[1])
    backbone.train()
    backbone.read_batch([empty], tasks[:1], contexts[:1], False)
    assert backbone.language_model.training
    with pytest.raises(ValueError, match="exceeds model limit"):
        backbone.read_batch([empty], tasks[:1], [(4,) * 256])


def test_reader_padding_preserves_gradients_and_bf16_execution(components):
    backbone, writer = components
    backbone.eval()
    memories = [torch.randn(2, 8, requires_grad=True), torch.randn(7, 8, requires_grad=True)]
    tokens = [ReadTokens((11,), (4, 5, 2)), ReadTokens((11, 12, 13), (7, 8, 9, 10, 2))]
    batched = backbone.read_batch(memories, tokens)
    batch_grads = torch.autograd.grad(sum(o.mean_nll for o in batched), memories, retain_graph=True)
    single_grads = [
        torch.autograd.grad(backbone.read_batch([m], [t])[0].mean_nll, m)[0]
        for m, t in zip(memories, tokens)
    ]
    for batch, single in zip(batch_grads, single_grads):
        torch.testing.assert_close(batch, single, rtol=2e-5, atol=1e-7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        features = backbone.text_features([(4, 5, 6), (7, 8, 9, 10)], [0, 0])
        states = writer.update_batch(
            [writer.initialize_state(dtype=f.dtype) for f in features], features, [0, 0], [2, 5]
        )
        output = backbone.read_batch([s.values for s in states], tokens)
        loss = sum(o.mean_nll for o in output)
    loss.backward()
    assert torch.isfinite(loss)
    assert backbone.input_projection.weight.grad.abs().sum() > 0
    assert all(s.values.dtype == torch.bfloat16 for s in states)


def test_cached_generation_logits_match_full_prefix_read(components):
    backbone, _ = components
    backbone.eval()
    memory = torch.randn(4, backbone.d_mem)
    prompt = (11, 12)
    model = backbone.language_model.get_base_model()
    model.generation_config.min_new_tokens = 4
    cached_logits = []

    def capture_logits(module, args, kwargs, output):
        cached_logits.append(output.logits[0, -1].detach().clone())

    handle = model.register_forward_hook(capture_logits, with_kwargs=True)
    try:
        generated = backbone.greedy_students([memory], [prompt], [4])[0]
    finally:
        handle.remove()
    assert len(generated) == 4
    with torch.no_grad():
        full = backbone.read_batch(
            [memory], [ReadTokens(prompt, generated + (backbone.eos_token_id,))]
        )[0]
    torch.testing.assert_close(
        torch.stack(cached_logits), full.target_logits[:4], atol=1e-6, rtol=1e-5
    )


def test_batched_generation_matches_individual_with_different_lengths(components):
    backbone, _ = components
    backbone.eval()
    memories = [torch.randn(2, backbone.d_mem), torch.randn(7, backbone.d_mem)]
    prompts = [(11,), (11, 12, 13)]
    limits = [4, 6]
    batch = backbone.greedy_students(memories, prompts, limits)
    single = [
        backbone.greedy_students([m], [p], [limit])[0]
        for m, p, limit in zip(memories, prompts, limits)
    ]
    assert batch == single
