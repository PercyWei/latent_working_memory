"""输出、梯度与保存激活量共同验证 v3 读写显存优化。"""

from contextlib import contextmanager

import pytest
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .test_model import build_model, ids, randomize_adapter


def masked_compress(model, texts, histories):
    embed = model.language_model.get_input_embeddings()
    rows = [
        torch.cat(
            [
                *(block.to(embed.weight.dtype) for block in blocks),
                embed(text),
                model.memory_embeddings.to(embed.weight.dtype),
            ]
        )
        for text, blocks in zip(texts, histories, strict=True)
    ]
    inputs = pad_sequence(rows, batch_first=True)
    positions = torch.arange(inputs.shape[1])[None].expand(len(rows), -1)
    mask = positions < torch.tensor([len(row) for row in rows])[:, None]
    hidden = (
        model.language_model.get_base_model()
        .model(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions.masked_fill(~mask, 0),
            use_cache=False,
        )
        .last_hidden_state
    )
    return [
        hidden[index, len(row) - model.memory_slots : len(row)] for index, row in enumerate(rows)
    ]


def masked_answers(model, memories, prompts, answers):
    embed = model.decoder.get_input_embeddings()
    rows = [
        torch.cat((memory.to(embed.weight.dtype), embed(torch.cat((prompt, answer[:-1])))))
        for memory, prompt, answer in zip(memories, prompts, answers, strict=True)
    ]
    inputs = pad_sequence(rows, batch_first=True)
    positions = torch.arange(inputs.shape[1])[None].expand(len(rows), -1)
    mask = positions < torch.tensor([len(row) for row in rows])[:, None]
    hidden = model.decoder.model(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions.masked_fill(~mask, 0),
        use_cache=False,
    ).last_hidden_state
    losses = []
    for row, memory, prompt, answer in zip(hidden, memories, prompts, answers, strict=True):
        start = len(memory) + len(prompt) - 1
        logits = model.decoder.get_output_embeddings()(row[start : start + len(answer)])
        losses.append(F.cross_entropy(logits.float(), answer))
    return torch.stack(losses)


def recurrent_loss(model, masked=False):
    writer = (
        masked_compress
        if masked
        else lambda codec, text, history: codec.compress_batch(text, history)
    )
    reader = (
        masked_answers
        if masked
        else lambda codec, memory, prompt, answer: codec.answer_nll(memory, prompt, answer)
    )
    first = writer(model, [ids(3, 4, 5, 6, 7), ids(8, 9)], [[], []])
    for memory in first:
        memory.retain_grad()
    second = writer(model, [ids(10, 11), ids(12, 13, 14, 15)], [[first[0]], [first[1]]])
    losses = reader(
        model,
        [torch.cat((first[0], second[0])), second[1]],
        [ids(16, 17), ids(18)],
        [ids(19, 20, 21, 22), ids(23, 24)],
    )
    return first, second, losses


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
@pytest.mark.parametrize("attention_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpointed_causal_only_matches_masked_reference_across_recurrent_writes(
    model_type, attention_implementation, dtype
):
    reference = build_model(
        model_type,
        dtype=dtype,
        attention_implementation=attention_implementation,
        gradient_checkpointing=False,
    )
    randomize_adapter(reference)
    actual = build_model(
        model_type,
        dtype=dtype,
        attention_implementation=attention_implementation,
        gradient_checkpointing=True,
    )
    actual.load_state_dict(reference.state_dict())
    old_first, old_second, expected = recurrent_loss(reference, masked=True)
    first, second, losses = recurrent_loss(actual)
    losses.mean().backward()
    expected.mean().backward()
    tolerance = (
        {"rtol": 0.03, "atol": 1e-3}
        if dtype == torch.bfloat16
        else {
            "rtol": 2e-4,
            "atol": 1e-6,
        }
    )
    torch.testing.assert_close(losses, expected, **tolerance)
    torch.testing.assert_close(first + second, old_first + old_second, **tolerance)
    for memory, old in zip(first, old_first, strict=True):
        assert memory.grad is not None and memory.grad.abs().sum() > 0
        torch.testing.assert_close(memory.grad, old.grad, **tolerance)
    old_parameters = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        old = old_parameters[name]
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, old.grad, **tolerance)
        else:
            assert parameter.grad is None and old.grad is None


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_decoder_weights_are_independent_frozen_and_have_no_adapter(model_type):
    model = build_model(model_type)
    writer = model.language_model.get_base_model()
    reader = model.decoder
    assert writer is not reader
    assert not any("lora_" in name for name, _ in reader.named_parameters())
    assert all(not parameter.requires_grad for parameter in reader.parameters())
    original = reader.get_input_embeddings().weight.detach().clone()
    assert (
        writer.get_input_embeddings().weight.data_ptr()
        != reader.get_input_embeddings().weight.data_ptr()
    )
    with torch.no_grad():
        writer.get_input_embeddings().weight.add_(1)
    torch.testing.assert_close(reader.get_input_embeddings().weight, original, rtol=0, atol=0)


@pytest.mark.parametrize("training", [False, True])
def test_generation_uses_decoder_cache_and_restores_training_mode(training):
    model = build_model("qwen3", attention_implementation="sdpa")
    model.train(training)
    forwards = []

    def record(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        forwards.append(
            (module.training, kwargs["use_cache"], 0 if cache is None else cache.get_seq_length())
        )

    hook = model.decoder.model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        tokens = model.generate(torch.randn(3, model.width), ids(3, 4), 4, None, 0)
    finally:
        hook.remove()
    assert len(tokens) == 4 and len(forwards) == 4
    assert all(not mode and cached for mode, cached, _ in forwards)
    assert [length for _, _, length in forwards] == [0, 5, 6, 7]
    assert model.training is training
    assert model.language_model.training is training
    assert model.decoder.training is training


def test_generation_restores_decoder_mode_after_exception(monkeypatch):
    model = build_model()

    def fail(**kwargs):
        assert not model.decoder.training
        raise RuntimeError("generation failed")

    monkeypatch.setattr(model.decoder, "generate", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        model.generate(torch.randn(3, model.width), ids(3), 4, None, 0)
    assert model.decoder.training


@contextmanager
def saved_activation_bytes(model):
    parameters = {parameter.untyped_storage().data_ptr() for parameter in model.parameters()}
    storages = {}

    def save(tensor):
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in parameters and storage.nbytes():
            storages[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        yield storages


def test_layer_checkpointing_reduces_saved_activations_without_changing_gradient():
    plain = build_model("qwen3", max_positions=128, gradient_checkpointing=False)
    randomize_adapter(plain)
    checked = build_model("qwen3", max_positions=128, gradient_checkpointing=True)
    checked.load_state_dict(plain.state_dict())
    measured = []
    for model in (plain, checked):
        with saved_activation_bytes(model) as storages:
            first, _, losses = recurrent_loss(model)
        measured.append(sum(storages.values()))
        losses.mean().backward()
        assert all(memory.grad is not None for memory in first)
    assert measured[1] < measured[0] * 0.6
    for (name, parameter), (other_name, other) in zip(
        plain.named_parameters(), checked.named_parameters(), strict=True
    ):
        assert name == other_name
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-4, atol=1e-6)
