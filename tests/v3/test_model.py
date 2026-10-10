import copy

import pytest
import torch
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from latent_working_memory.v3.model import GistMemoryModel
from latent_working_memory.v3 import model as model_module


def build_model(
    model_type="llama",
    memory_slots=3,
    max_positions=64,
    dropout=0.0,
    dtype=torch.float32,
    attention_implementation="eager",
    gradient_checkpointing=True,
    write_slots=None,
    writer_mode="local",
    tag_tokens=3,
):
    torch.manual_seed(93)
    config_type, model_class = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    }[model_type]
    config = config_type(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=max_positions,
        attention_dropout=dropout,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = attention_implementation
    base = model_class(config).to(dtype=dtype)
    return GistMemoryModel(
        base,
        memory_slots,
        2,
        4,
        ("q_proj", "v_proj"),
        gradient_checkpointing=gradient_checkpointing,
        write_slots=write_slots,
        writer_mode=writer_mode,
        tag_tokens=tag_tokens,
    )


def ids(*values):
    return torch.tensor(values, dtype=torch.long)


def randomize_adapter(model):
    with torch.no_grad():
        for name, parameter in model.language_model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.2)


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_compression_reads_final_gist_states_and_supports_history_ranges(model_type):
    model = build_model(model_type)
    model.train()
    assert model.language_model.training and model.decoder.training
    assert model.training
    inputs, hidden = [], []
    base = model.language_model.get_base_model().model
    before = base.register_forward_pre_hook(
        lambda module, args, kwargs: inputs.append(kwargs["inputs_embeds"]), with_kwargs=True
    )
    after = base.register_forward_hook(
        lambda module, args, output: hidden.append(output.last_hidden_state)
    )
    try:
        first = model.compress(ids(3, 4, 5))
        second = model.compress(ids(6, 7), [first])
        third = model.compress(ids(8), [first, second])
    finally:
        before.remove()
        after.remove()
    for memory, states in zip([first, second, third], hidden, strict=True):
        torch.testing.assert_close(memory, states[0, -3:])
        assert memory.shape == (3, 16)
    assert [row.shape[1] for row in inputs] == [6, 8, 10]
    torch.testing.assert_close(inputs[1][0, :3], first)
    torch.testing.assert_close(inputs[2][0, :6], torch.cat((first, second)))
    torch.testing.assert_close(inputs[0][0, -3:], model.memory_embeddings)
    torch.testing.assert_close(inputs[1][0, -3:], model.memory_embeddings)
    assert all(
        not p.requires_grad for n, p in model.language_model.named_parameters() if "lora_" not in n
    )
    assert all(p.requires_grad for n, p in model.language_model.named_parameters() if "lora_" in n)


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_independent_frozen_reader_ignores_adapter_without_detaching_memory(model_type):
    model = build_model(model_type)
    memory = torch.randn(4, model.width, requires_grad=True)
    prompts, answers = [ids(3, 4)], [ids(8, 9, 10)]
    before = model.answer_nll([memory], prompts, answers)
    initial_write = model.compress(ids(3, 4, 5)).detach()
    randomize_adapter(model)
    after = model.answer_nll([memory], prompts, answers)
    torch.testing.assert_close(after, before)
    assert not torch.allclose(initial_write, model.compress(ids(3, 4, 5)))
    after.sum().backward()
    assert memory.grad is not None and memory.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.language_model.parameters())
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.decoder.parameters()
    )
    assert all(
        not module.disable_adapters
        for module in model.language_model.modules()
        if hasattr(module, "lora_A")
    )


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_reader_loss_reaches_gist_embeddings_and_writer_lora(model_type):
    model = build_model(model_type)
    randomize_adapter(model)
    first = model.compress(ids(3, 4, 5))
    first.retain_grad()
    latest = model.compress(ids(6, 7), [first])
    loss = model.answer_nll([latest], [ids(9, 10)], [ids(11, 12, 2)]).mean()
    loss.backward()
    assert first.grad is not None and first.grad.abs().sum() > 0
    assert model.memory_embeddings.grad is not None and model.memory_embeddings.grad.abs().sum() > 0
    adapter_grads = [p.grad for n, p in model.language_model.named_parameters() if "lora_" in n]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in adapter_grads)
    assert sum(grad.abs().sum() for grad in adapter_grads) > 0
    assert all(
        p.grad is None for n, p in model.language_model.named_parameters() if "lora_" not in n
    )


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_bfloat16_sdpa_backbone_keeps_float32_trainables_and_finite_gradient(model_type):
    model = build_model(model_type, dtype=torch.bfloat16, attention_implementation="sdpa")
    randomize_adapter(model)
    memory = model.compress(ids(3, 4, 5))
    assert memory.dtype == torch.bfloat16
    assert model.memory_embeddings.dtype == torch.float32
    loss = model.answer_nll(
        [memory, torch.cat((memory, memory))], [ids(6), ids(6, 7)], [ids(8, 9), ids(10)]
    ).mean()
    assert loss.dtype == torch.float32
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert all(parameter.dtype == torch.float32 for parameter in trainable)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainable
    )
    assert model.memory_embeddings.grad.abs().sum() > 0


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_ragged_question_batch_matches_individual_and_dense_answer_only_nll(model_type):
    model = build_model(model_type)
    randomize_adapter(model)
    memories = [torch.randn(length, model.width, requires_grad=True) for length in (0, 3, 7)]
    prompts = [ids(4, 5, 6), ids(7), ids(8, 9)]
    answers = [ids(10), ids(11, 12, 13, 14), ids(15, 16)]
    projection_rows = []
    head = model.decoder.get_output_embeddings()
    hook = head.register_forward_pre_hook(
        lambda module, args: projection_rows.append(args[0].shape)
    )
    batched = model.answer_nll(memories, prompts, answers)
    hook.remove()
    assert projection_rows == [torch.Size([sum(len(answer) for answer in answers), model.width])]
    separate = torch.stack(
        [
            model.answer_nll([memory], [prompt], [answer])[0]
            for memory, prompt, answer in zip(memories, prompts, answers, strict=True)
        ]
    )
    torch.testing.assert_close(batched, separate, rtol=1e-5, atol=1e-6)
    batched_grad = torch.autograd.grad(batched.mean(), memories, retain_graph=True)
    separate_grad = torch.autograd.grad(separate.mean(), memories)
    torch.testing.assert_close(batched_grad, separate_grad, rtol=1e-4, atol=1e-7)
    dense = []
    base = model.decoder
    for memory, prompt, answer in zip(memories, prompts, answers, strict=True):
        embeddings = torch.cat((memory, base.get_input_embeddings()(torch.cat((prompt, answer)))))
        logits = base(inputs_embeds=embeddings[None], use_cache=False).logits[0]
        offset = len(memory) + len(prompt) - 1
        dense.append(F.cross_entropy(logits[offset : offset + len(answer)], answer))
    torch.testing.assert_close(batched, torch.stack(dense), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_long_targets_chunk_checkpoint_head_matches_dense_loss_and_gradient(monkeypatch, dtype):
    model = build_model("qwen3", max_positions=384, dtype=dtype, attention_implementation="sdpa")
    randomize_adapter(model)
    memories = [torch.randn(length, model.width, requires_grad=True) for length in (3, 6)]
    prompts = [ids(4, 5), ids(6)]
    answers = [torch.arange(length).remainder(37) + 3 for length in (301, 41)]
    calls = []
    original = model_module.checkpoint

    def observed_checkpoint(function, hidden, targets, use_reentrant):
        assert use_reentrant is False
        calls.append(len(hidden))
        return original(function, hidden, targets, use_reentrant=False)

    monkeypatch.setattr(model_module, "checkpoint", observed_checkpoint)
    projected, adapters_enabled = [], []

    def observe_head(module, args):
        projected.append(len(args[0]))
        adapters_enabled.append(
            all(
                not module.disable_adapters
                for module in model.language_model.modules()
                if hasattr(module, "lora_A")
            )
        )

    hook = model.decoder.get_output_embeddings().register_forward_pre_hook(observe_head)
    actual = model.answer_nll(memories, prompts, answers)
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    assert calls == [256, 86]
    assert projected == [256, 86]
    actual.mean().backward()
    hook.remove()
    assert sorted(projected) == [86, 86, 256, 256]
    assert all(adapters_enabled)
    chunked_gradients = [memory.grad.clone() for memory in memories]
    for memory in memories:
        memory.grad = None
    expected = []
    base = model.decoder
    for memory, prompt, answer in zip(memories, prompts, answers, strict=True):
        text = base.get_input_embeddings()(torch.cat((prompt, answer)))
        inputs = torch.cat((memory.to(text.dtype), text))
        logits = base(inputs_embeds=inputs[None], use_cache=False).logits[0]
        offset = len(memory) + len(prompt) - 1
        expected.append(F.cross_entropy(logits[offset : offset + len(answer)].float(), answer))
    expected = torch.stack(expected)
    expected.mean().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        chunked_gradients,
        [memory.grad for memory in memories],
        rtol=0.03 if dtype == torch.bfloat16 else 1e-4,
        atol=1e-5 if dtype == torch.bfloat16 else 1e-7,
    )
    assert all(parameter.grad is None for parameter in model.language_model.parameters())
    with torch.no_grad():
        without_grad = model.answer_nll(memories, prompts, answers)
    torch.testing.assert_close(without_grad, actual)
    assert calls == [256, 86]
    # 长目标的 checkpoint 读取同样可以继续反传到实际 gist 写入器。
    memory = model.compress(ids(3, 4, 5))
    loss = model.answer_nll([memory], prompts[:1], answers[:1]).mean()
    loss.backward()
    assert torch.isfinite(model.memory_embeddings.grad).all()
    assert model.memory_embeddings.grad.abs().sum() > 0
    assert all(
        parameter.grad is None
        for name, parameter in model.language_model.named_parameters()
        if "lora_" not in name
    )


def test_backbone_dropout_is_disabled_while_training():
    model = build_model(dropout=0.7)
    model.train()
    first = model.compress(ids(3, 4, 5))
    second = model.compress(ids(3, 4, 5))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    before = model.answer_nll([first], [ids(6)], [ids(7, 8)])
    repeated = model.answer_nll([first], [ids(6)], [ids(7, 8)])
    model.eval()
    evaluated = model.answer_nll([first], [ids(6)], [ids(7, 8)])
    torch.testing.assert_close(before, repeated, rtol=0, atol=0)
    torch.testing.assert_close(before, evaluated, rtol=0, atol=0)


def test_memory_initialization_uses_zero_mean_gaussian():
    model = build_model(memory_slots=4096)
    assert abs(model.memory_embeddings.mean().item()) < 0.0003
    assert abs(model.memory_embeddings.std().item() - 0.02) < 0.0003


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_generation_returns_only_new_tokens_and_ignores_adapter(model_type):
    model = build_model(model_type)
    memory, prompt = torch.randn(5, model.width), ids(3, 4, 5)
    first = model.generate(memory, prompt, 4, eos_token_id=None, pad_token_id=0)
    randomize_adapter(model)
    second = model.generate(memory, prompt, 4, eos_token_id=None, pad_token_id=0)
    assert first.shape == (4,)
    assert first.dtype == torch.long
    assert not first.requires_grad
    torch.testing.assert_close(first, second)
    assert all(
        not module.disable_adapters
        for module in model.language_model.modules()
        if hasattr(module, "lora_A")
    )


def test_trainable_checkpoint_restores_outputs_and_rejects_invalid_state():
    model = build_model()
    randomize_adapter(model)
    with torch.no_grad():
        model.memory_embeddings.add_(0.3)
    expected = model.compress(ids(3, 4, 5)).detach()
    state = model.trainable_state_dict()
    assert set(state) == {"memory_embeddings", "adapter"}
    assert all("lora_" in name for name in state["adapter"])
    restored = build_model()
    restored.load_trainable_state_dict(state)
    torch.testing.assert_close(restored.compress(ids(3, 4, 5)), expected)
    with torch.no_grad():
        model.memory_embeddings.zero_()
    assert not torch.equal(state["memory_embeddings"], model.memory_embeddings)
    invalid = copy.deepcopy(state)
    invalid["adapter"].pop(next(iter(invalid["adapter"])))
    with pytest.raises(ValueError, match="adapter keys"):
        restored.load_trainable_state_dict(invalid)
    with pytest.raises(ValueError, match="exactly"):
        restored.load_trainable_state_dict({**state, "backbone": {}})
    invalid = copy.deepcopy(state)
    invalid["memory_embeddings"] = torch.zeros(1, model.width)
    with pytest.raises(ValueError, match="embedding shape"):
        restored.load_trainable_state_dict(invalid)


def test_context_limits_fail_without_truncation():
    model = build_model(max_positions=12)
    with pytest.raises(ValueError, match="exceeds model window"):
        model.compress(torch.arange(10))
    with pytest.raises(ValueError, match="exceeds model window"):
        model.compress(ids(3, 4, 5, 6), [torch.zeros(3, 16), torch.zeros(3, 16)])
    with pytest.raises(ValueError, match="exceeds model window"):
        model.answer_nll([torch.zeros(9, 16)], [ids(3, 4)], [ids(5, 6, 7, 8)])
    with pytest.raises(ValueError, match="exceeds model window"):
        model.generate(torch.zeros(8, 16), ids(3, 4), 3, 2, 0)
    assert model.compress(torch.arange(9)).shape == (3, 16)
    assert model.compress(torch.arange(11), output_slots=1).shape == (1, 16)
    assert model.compress(torch.arange(9), [torch.zeros(1, 16)], output_slots=2).shape == (2, 16)
    with pytest.raises(ValueError, match="exceeds model window"):
        model.compress(torch.arange(10), [torch.zeros(1, 16)], output_slots=2)
    assert model.answer_nll([torch.zeros(9, 16)], [ids(3, 4)], [ids(5, 6)]).shape == (1,)


def test_total_memory_budget_and_writer_embedding_count_are_distinct():
    model = build_model(memory_slots=8, write_slots=3)
    assert model.memory_slots == 8
    assert model.write_slots == 3
    assert model.memory_embeddings.shape == (3, model.width)
    first, second = model.compress_batch([ids(3, 4), ids(5, 6)], output_slots=[3, 2])
    assert [len(first), len(second)] == [3, 2]
    assert len(model.compress(ids(7, 8))) == 3
    with pytest.raises(ValueError, match="write_slots"):
        model.compress(ids(3, 4), output_slots=4)
    loss = model.answer_nll([torch.cat((first, second))], [ids(9)], [ids(10, 11)]).mean()
    loss.backward()
    assert torch.all(model.memory_embeddings.grad.abs().sum(dim=1) > 0)


@pytest.mark.parametrize("write_slots", [0, -1, 4, 1.0, True])
def test_invalid_writer_embedding_count_is_rejected(write_slots):
    with pytest.raises(ValueError, match="write_slots"):
        build_model(memory_slots=3, write_slots=write_slots)


def test_invalid_writer_blocks_and_empty_answers_are_rejected():
    model = build_model()
    for count in (0, 4):
        with pytest.raises(ValueError, match="memory_slots"):
            model.compress(ids(3, 4), [torch.zeros(count, 16)])
    with pytest.raises(ValueError, match="nonempty"):
        model.answer_nll([torch.zeros(3, 16)], [ids(3, 4)], [ids()])
    with pytest.raises(ValueError, match="align"):
        model.answer_nll([torch.zeros(3, 16)], [], [])


def test_unsupported_lora_dropout_is_explicit():
    base = build_model().decoder
    with pytest.raises(ValueError, match="lora_dropout=0"):
        GistMemoryModel(base, 3, 2, 4, ("q_proj",), lora_dropout=0.1)


@pytest.mark.parametrize("enabled", [False, True])
def test_native_gradient_checkpointing_configuration(enabled):
    model = build_model(gradient_checkpointing=enabled)
    assert model.language_model.is_gradient_checkpointing is enabled
    assert model.decoder.is_gradient_checkpointing is enabled
    assert model.language_model.training and model.decoder.training
    model.eval()
    assert not model.language_model.training and not model.decoder.training
    model.train()
    assert model.language_model.training and model.decoder.training


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
@pytest.mark.parametrize("attention_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("output_slots", [None, [3, 1, 2]])
def test_ragged_writer_batch_matches_single_writes_outputs_and_all_gradients(
    model_type, attention_implementation, output_slots
):
    batched = build_model(model_type, attention_implementation=attention_implementation)
    randomize_adapter(batched)
    separate = copy.deepcopy(batched)
    texts = [ids(3, 4, 5, 6), ids(7), ids(8, 9)]
    histories = [
        [],
        [torch.randn(1, batched.width, requires_grad=True)],
        [torch.randn(length, batched.width, requires_grad=True) for length in (3, 2)],
    ]
    counts = [batched.memory_slots] * len(texts) if output_slots is None else output_slots
    separate_histories = [
        [memory.detach().clone().requires_grad_() for memory in blocks] for blocks in histories
    ]
    calls = []
    states = []
    base = batched.language_model.get_base_model().model
    before = base.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs), with_kwargs=True
    )
    after = base.register_forward_hook(
        lambda module, args, output: states.append(output.last_hidden_state)
    )
    try:
        actual = batched.compress_batch(texts, histories, output_slots)
    finally:
        before.remove()
        after.remove()
    expected = [
        separate.compress(text, blocks, count)
        for text, blocks, count in zip(texts, separate_histories, counts, strict=True)
    ]
    assert len(calls) == 1
    lengths = [
        len(text) + sum(len(memory) for memory in blocks) + count
        for text, blocks, count in zip(texts, histories, counts, strict=True)
    ]
    assert calls[0]["inputs_embeds"].shape == (3, max(lengths), batched.width)
    assert calls[0]["attention_mask"] is None
    for index, (length, count) in enumerate(zip(lengths, counts, strict=True)):
        assert actual[index].shape == (count, batched.width)
        assert calls[0]["position_ids"][index].tolist() == list(range(max(lengths)))
        torch.testing.assert_close(actual[index], states[0][index, length - count : length])
        torch.testing.assert_close(
            calls[0]["inputs_embeds"][index, length - count : length],
            batched.memory_embeddings[:count],
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    weights = [torch.randn_like(memory) for memory in actual]
    sum((memory * weight).sum() for memory, weight in zip(actual, weights, strict=True)).backward()
    sum(
        (memory * weight).sum() for memory, weight in zip(expected, weights, strict=True)
    ).backward()
    for (name, parameter), (other_name, other) in zip(
        batched.named_parameters(), separate.named_parameters(), strict=True
    ):
        assert name == other_name
        if parameter.requires_grad:
            assert parameter.grad is not None and other.grad is not None
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-4, atol=5e-5)
        else:
            assert parameter.grad is None and other.grad is None
    for blocks, expected_blocks in zip(histories, separate_histories, strict=True):
        for memory, other in zip(blocks, expected_blocks, strict=True):
            assert memory.grad.abs().sum() > 0
            torch.testing.assert_close(memory.grad, other.grad, rtol=1e-4, atol=1e-6)
    assert batched.memory_embeddings.grad.abs().sum() > 0


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_short_writes_only_use_requested_embedding_prefix_and_retain_checkpoint_shape(model_type):
    model = build_model(model_type)
    randomize_adapter(model)
    written = model.compress_batch([ids(3, 4), ids(5, 6, 7)], output_slots=[1, 2])
    model.answer_nll(written, [ids(8), ids(9, 10)], [ids(11, 12), ids(13)]).mean().backward()
    assert model.memory_embeddings.grad[:2].abs().sum() > 0
    assert torch.count_nonzero(model.memory_embeddings.grad[2:]) == 0
    state = model.trainable_state_dict()
    assert state["memory_embeddings"].shape == (3, model.width)
    restored = build_model(model_type)
    restored.load_trainable_state_dict(state)
    torch.testing.assert_close(
        restored.compress_batch([ids(3, 4), ids(5, 6, 7)], output_slots=[1, 2]), written
    )


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_writer_batch_has_no_cross_sample_attention_or_history_gradient(model_type):
    model = build_model(model_type, attention_implementation="sdpa")
    randomize_adapter(model)
    histories = [
        [torch.randn(3, model.width, requires_grad=True)],
        [torch.randn(3, model.width, requires_grad=True)],
    ]
    first = model.compress_batch([ids(3, 4), ids(5, 6, 7, 8)], histories)
    weights = torch.randn_like(first[0])
    gradients = torch.autograd.grad((first[0] * weights).sum(), [blocks[0] for blocks in histories])
    assert gradients[0].abs().sum() > 0
    assert torch.count_nonzero(gradients[1]) == 0
    second = model.compress_batch(
        [ids(3, 4), ids(9, 10, 11, 12, 13)],
        [histories[0], [torch.randn(3, model.width), torch.randn(3, model.width)]],
    )
    torch.testing.assert_close(first[0], second[0], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(first[1], second[1])


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_batched_recurrent_writes_and_reader_retain_gradients_in_bfloat16(model_type):
    model = build_model(model_type, dtype=torch.bfloat16, attention_implementation="sdpa")
    randomize_adapter(model)
    first = model.compress_batch([ids(3, 4), ids(5, 6, 7)])
    for memory in first:
        memory.retain_grad()
    updated = model.compress_batch([ids(8, 9, 10), ids(11)], [[first[0]], [first[1]]])
    loss = model.answer_nll(updated, [ids(12), ids(13, 14)], [ids(15, 16), ids(17)]).mean()
    loss.backward()
    assert all(memory.dtype == torch.bfloat16 for memory in first + updated)
    assert all(memory.grad is not None and memory.grad.abs().sum() > 0 for memory in first)
    trainables = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert all(parameter.dtype == torch.float32 for parameter in trainables)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainables
    )
    assert model.memory_embeddings.grad.abs().sum() > 0


def test_writer_batch_rejects_invalid_contracts_before_transformer_forward():
    model = build_model(max_positions=12)
    calls = []
    hook = model.language_model.get_base_model().model.register_forward_pre_hook(
        lambda module, args: calls.append(args)
    )
    try:
        with pytest.raises(ValueError, match="nonempty"):
            model.compress_batch([])
        with pytest.raises(ValueError, match="align"):
            model.compress_batch([ids(3), ids(4)], [[]])
        for count in (0, model.memory_slots + 1):
            with pytest.raises(ValueError, match="memory_slots"):
                model.compress_batch([ids(3), ids(4)], [[], [torch.zeros(count, model.width)]])
        with pytest.raises(ValueError, match="align"):
            model.compress_batch([ids(3), ids(4)], output_slots=[1])
        for invalid_count in (0, -1, 4, 1.0, True, None):
            with pytest.raises(ValueError, match="output_slots"):
                model.compress_batch([ids(3), ids(4)], output_slots=[1, invalid_count])
        with pytest.raises(ValueError, match="output_slots"):
            model.compress(ids(3), output_slots=False)
        with pytest.raises(ValueError, match="nonempty"):
            model.compress_batch([ids(3), ids()])
        with pytest.raises(ValueError, match="exceeds model window"):
            model.compress_batch([ids(3), torch.arange(10)])
    finally:
        hook.remove()
    assert not calls


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
@pytest.mark.parametrize("tag_tokens", [1, 3, 5])
def test_tag_embeddings_follow_full_history_and_are_absent_from_initial_write(
    model_type, tag_tokens
):
    model = build_model(model_type, writer_mode="tag", tag_tokens=tag_tokens)
    history = [torch.randn(2, model.width), torch.randn(3, model.width)]
    calls, states = [], []
    base = model.language_model.get_base_model().model
    hook = base.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs), with_kwargs=True
    )
    after = base.register_forward_hook(
        lambda module, args, output: states.append(output.last_hidden_state)
    )
    try:
        written = model.compress_batch(
            [ids(3, 4), ids(5), ids(6, 7)],
            [history, history, []],
            [3, 2, 3],
            ["append", "overwrite", "initial"],
        )
    finally:
        hook.remove()
        after.remove()
    assert len(calls) == 1 and calls[0]["attention_mask"] is None
    inputs = calls[0]["inputs_embeds"]
    embed = model.language_model.get_input_embeddings()
    for i in range(2):
        torch.testing.assert_close(inputs[i, :5], torch.cat(history))
        torch.testing.assert_close(inputs[i, 5 : 5 + tag_tokens], model.tag_embeddings[i])
    text_start = 5 + tag_tokens
    torch.testing.assert_close(inputs[0, text_start : text_start + 2], embed(ids(3, 4)))
    torch.testing.assert_close(inputs[1, text_start : text_start + 1], embed(ids(5)))
    torch.testing.assert_close(inputs[2, :2], embed(ids(6, 7)))
    torch.testing.assert_close(inputs[2, 2:5], model.memory_embeddings)
    for index, (length, count) in enumerate(((10 + tag_tokens, 3), (8 + tag_tokens, 2), (5, 3))):
        torch.testing.assert_close(written[index], states[0][index, length - count : length])


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tag_tokens", [1, 3, 5])
def test_tag_rows_are_trained_by_qa_without_changing_frozen_word_embeddings(
    model_type, dtype, tag_tokens
):
    model = build_model(
        model_type,
        writer_mode="tag",
        dtype=dtype,
        attention_implementation="sdpa",
        tag_tokens=tag_tokens,
    )
    randomize_adapter(model)
    assert model.tag_embeddings.shape == (2, tag_tokens, model.width)
    assert model.tag_embeddings.dtype == torch.float32
    encoder_words = model.language_model.get_input_embeddings().weight
    decoder_words = model.decoder.get_input_embeddings().weight
    original_words = encoder_words.detach().clone()
    original_decoder_words = decoder_words.detach().clone()
    original_tags = model.tag_embeddings.detach().clone()
    history = [torch.randn(3, model.width), torch.randn(2, model.width)]
    calls = []
    hook = model.language_model.get_base_model().model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs), with_kwargs=True
    )
    try:
        memories = model.compress_batch(
            [ids(3, 4, 5), ids(3, 4, 5)],
            [history, history],
            [2, 2],
            ["append", "overwrite"],
        )
    finally:
        hook.remove()
    assert not torch.allclose(memories[0], memories[1])
    for row in calls[0]["inputs_embeds"]:
        torch.testing.assert_close(row[-2:], model.memory_embeddings[:2].to(dtype))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01
    )
    loss = model.answer_nll(memories, [ids(6), ids(6)], [ids(7, 8), ids(7, 8)]).mean()
    loss.backward()
    assert model.tag_embeddings.grad is not None
    assert torch.isfinite(model.tag_embeddings.grad).all()
    assert torch.all(model.tag_embeddings.grad.abs().sum(dim=-1) > 0)
    assert torch.all(model.memory_embeddings.grad[:2].abs().sum(dim=1) > 0)
    assert not encoder_words.requires_grad and encoder_words.grad is None
    assert not decoder_words.requires_grad and decoder_words.grad is None
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.decoder.parameters()
    )
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for name, parameter in model.language_model.named_parameters()
        if "lora_" not in name
    )
    optimizer.step()
    assert torch.all((model.tag_embeddings - original_tags).abs().sum(dim=-1) > 0)
    torch.testing.assert_close(encoder_words, original_words)
    torch.testing.assert_close(decoder_words, original_decoder_words)


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
@pytest.mark.parametrize("tag_tokens", [1, 3, 5])
def test_tag_gradient_checkpointing_matches_full_qa_backward(model_type, tag_tokens):
    checked = build_model(
        model_type, writer_mode="tag", attention_implementation="sdpa", tag_tokens=tag_tokens
    )
    randomize_adapter(checked)
    plain = build_model(
        model_type,
        writer_mode="tag",
        attention_implementation="sdpa",
        gradient_checkpointing=False,
        tag_tokens=tag_tokens,
    )
    plain.load_trainable_state_dict(checked.trainable_state_dict())

    def loss(model):
        first = model.compress(ids(3, 4), action="initial")
        appended = model.compress(ids(5), [first], 1, "append")
        overwritten = model.compress(ids(6, 7), [first, appended], 1, "overwrite")
        return model.answer_nll(
            [torch.cat((first, appended, overwritten))], [ids(8)], [ids(9, 10)]
        ).mean()

    checked_loss, plain_loss = loss(checked), loss(plain)
    torch.testing.assert_close(checked_loss, plain_loss, rtol=1e-6, atol=1e-7)
    checked_loss.backward()
    plain_loss.backward()
    for (name, actual), (expected_name, expected) in zip(
        checked.named_parameters(), plain.named_parameters(), strict=True
    ):
        assert name == expected_name
        if actual.requires_grad:
            assert actual.grad is not None and expected.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-6)
        else:
            assert actual.grad is None and expected.grad is None


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
@pytest.mark.parametrize("tag_tokens", [1, 3, 5])
def test_tag_checkpoint_initialization_and_strict_roundtrip(model_type, tag_tokens):
    source = build_model(model_type)
    randomize_adapter(source)
    source_state = source.trainable_state_dict()
    target = build_model(model_type, writer_mode="tag", tag_tokens=tag_tokens)
    initial_tags = target.tag_embeddings.detach().clone()
    # 标签参数在 gist 初始化后才创建，旧模式的随机初始化结果不变。
    torch.testing.assert_close(source.memory_embeddings, target.memory_embeddings)
    assert not hasattr(source, "tag_embeddings")
    with pytest.raises(ValueError, match="adapter keys"):
        target.load_trainable_state_dict(source_state)
    target.load_trainable_state_dict(source_state, initialize=True)
    torch.testing.assert_close(target.tag_embeddings, initial_tags)
    torch.testing.assert_close(
        target.compress(ids(3, 4), action="initial"), source.compress(ids(3, 4))
    )
    with torch.no_grad():
        target.tag_embeddings.add_(torch.tensor([[[0.2]], [[-0.3]]]))
    saved = target.trainable_state_dict()
    assert set(saved) == {"memory_embeddings", "adapter"}
    assert set(saved["adapter"]) == set(source_state["adapter"]) | {"tag_embeddings"}
    torch.testing.assert_close(saved["adapter"]["tag_embeddings"], target.tag_embeddings)
    restored = build_model(model_type, writer_mode="tag", tag_tokens=tag_tokens)
    restored.load_trainable_state_dict(saved)
    for action in ("initial", "append", "overwrite"):
        torch.testing.assert_close(
            restored.compress(ids(3, 4), action=action), target.compress(ids(3, 4), action=action)
        )
    with torch.no_grad():
        target.tag_embeddings.zero_()
    assert not torch.equal(saved["adapter"]["tag_embeddings"], target.tag_embeddings)
    with pytest.raises(ValueError, match="adapter keys"):
        source.load_trainable_state_dict(saved, initialize=True)
    missing = copy.deepcopy(saved)
    missing["adapter"].pop("tag_embeddings")
    with pytest.raises(ValueError, match="adapter keys"):
        restored.load_trainable_state_dict(missing)
    invalid = copy.deepcopy(saved)
    invalid["adapter"]["tag_embeddings"] = torch.zeros(1, target.width)
    with pytest.raises(ValueError, match="adapter shape differs for tag_embeddings"):
        restored.load_trainable_state_dict(invalid)
    mismatched = build_model(model_type, writer_mode="tag", tag_tokens=tag_tokens + 1)
    with pytest.raises(ValueError, match="adapter shape differs for tag_embeddings"):
        mismatched.load_trainable_state_dict(saved)


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
@pytest.mark.parametrize("attention_implementation", ["eager", "sdpa"])
def test_mask_writer_limits_only_gist_queries_at_every_layer(model_type, attention_implementation):
    model = build_model(
        model_type, writer_mode="mask", attention_implementation=attention_implementation
    )
    history = [torch.randn(2, model.width), torch.randn(3, model.width)]
    observed = []
    hooks = [
        layer.register_forward_pre_hook(
            lambda module, args, kwargs: observed.append(kwargs["attention_mask"]),
            with_kwargs=True,
        )
        for layer in model.language_model.get_base_model().model.layers
    ]
    try:
        outputs = model.compress_batch(
            [ids(3, 4), ids(5, 6, 7), ids(8)],
            [history, history, []],
            [2, 1, 3],
            ["append", "overwrite", "initial"],
        )
    finally:
        for hook in hooks:
            hook.remove()
    assert [len(memory) for memory in outputs] == [2, 1, 3]
    minimum = torch.finfo(torch.float32).min
    causal = torch.full((9, 9), minimum).triu(1)
    expected = causal[None, None].repeat(3, 1, 1, 1)
    expected[0, :, 7:9, :5] = minimum
    expected[1, :, 8:9, :2] = minimum
    assert len(observed) == 2
    for attention_mask in observed:
        torch.testing.assert_close(attention_mask, expected)


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_masked_history_has_an_indirect_gradient_through_new_text(model_type):
    model = build_model(model_type, writer_mode="mask", attention_implementation="sdpa")
    randomize_adapter(model)
    history = [torch.randn(3, model.width, requires_grad=True)]
    memory = model.compress(ids(3, 4, 5), history, action="append")
    weights = torch.randn_like(memory)
    gradient = torch.autograd.grad((memory * weights).sum(), history)[0]
    # S 每层都屏蔽历史，但新文本第一层读历史，第二层 S 再读取新文本。
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
@pytest.mark.parametrize(
    "writer_mode,tag_tokens", [("tag", 1), ("tag", 3), ("tag", 5), ("mask", 3), ("dual_lora", 3)]
)
def test_new_writer_ragged_mixed_batch_matches_separate_writes(model_type, writer_mode, tag_tokens):
    model = build_model(
        model_type, writer_mode=writer_mode, attention_implementation="sdpa", tag_tokens=tag_tokens
    )
    randomize_adapter(model)
    texts = [ids(3, 4, 5), ids(6), ids(7, 8)]
    histories = [
        [torch.randn(3, model.width), torch.randn(2, model.width)],
        [torch.randn(3, model.width)],
        [],
    ]
    actions, slots = ["overwrite", "append", "initial"], [2, 1, 3]
    batch = model.compress_batch(texts, histories, slots, actions)
    separate = [
        model.compress(text, history, count, action)
        for text, history, count, action in zip(texts, histories, slots, actions, strict=True)
    ]
    torch.testing.assert_close(batch, separate, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_dual_lora_recomputes_each_original_action_and_keeps_all_parameters_trainable(model_type):
    checked = build_model(model_type, writer_mode="dual_lora", attention_implementation="sdpa")
    randomize_adapter(checked)
    plain = build_model(
        model_type,
        writer_mode="dual_lora",
        attention_implementation="sdpa",
        gradient_checkpointing=False,
    )
    plain.load_trainable_state_dict(checked.trainable_state_dict())
    history = [torch.randn(3, checked.width), torch.randn(2, checked.width)]

    def loss(model):
        initial = model.compress(ids(3, 4), action="initial")
        appended = model.compress(ids(5, 6), history + [initial], 1, "append")
        overwritten = model.compress(ids(7, 8, 9), history + [initial], 3, "overwrite")
        # 两次 forward 后 active 恢复 default，反向仍需以 append 重算前两次。
        assert model.language_model.active_adapter == "default"
        assert all(
            parameter.requires_grad
            for name, parameter in model.language_model.named_parameters()
            if "lora_" in name
        )
        return model.answer_nll(
            [torch.cat((initial, appended)), overwritten],
            [ids(10, 11), ids(12)],
            [ids(13, 14), ids(15, 16)],
        ).mean()

    checked_loss, plain_loss = loss(checked), loss(plain)
    torch.testing.assert_close(checked_loss, plain_loss, rtol=1e-6, atol=1e-7)
    checked_loss.backward()
    plain_loss.backward()
    for (name, actual), (expected_name, expected) in zip(
        checked.named_parameters(), plain.named_parameters(), strict=True
    ):
        assert name == expected_name
        if actual.requires_grad:
            assert actual.grad is not None and expected.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-6)
        else:
            assert actual.grad is None and expected.grad is None
    assert checked.language_model.active_adapter == "default"


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_dual_checkpoint_roundtrip_and_explicit_single_adapter_initialization(model_type):
    source = build_model(model_type)
    randomize_adapter(source)
    state = source.trainable_state_dict()
    dual = build_model(model_type, writer_mode="dual_lora")
    with pytest.raises(ValueError, match="adapter keys"):
        dual.load_trainable_state_dict(state)
    dual.load_trainable_state_dict(state, initialize=True)
    torch.testing.assert_close(dual.memory_embeddings, source.memory_embeddings)
    expected = source.compress(ids(3, 4, 5))
    for action in ("initial", "append", "overwrite"):
        torch.testing.assert_close(dual.compress(ids(3, 4, 5), action=action), expected)
    dual_state = dual.trainable_state_dict()
    assert set(dual_state) == {"memory_embeddings", "adapter"}
    assert set(state["adapter"]) < set(dual_state["adapter"])
    assert len(dual_state["adapter"]) == 2 * len(state["adapter"])
    for name, tensor in state["adapter"].items():
        prefix, suffix = name.rsplit(".", 1)
        torch.testing.assert_close(dual_state["adapter"][name], tensor)
        torch.testing.assert_close(dual_state["adapter"][f"{prefix}.append.{suffix}"], tensor)
    randomize_adapter(dual)
    dual_state = dual.trainable_state_dict()
    restored = build_model(model_type, writer_mode="dual_lora")
    restored.load_trainable_state_dict(dual_state)
    for action in ("initial", "append", "overwrite"):
        torch.testing.assert_close(
            restored.compress(ids(3, 4), action=action), dual.compress(ids(3, 4), action=action)
        )
    with pytest.raises(ValueError, match="adapter keys"):
        source.load_trainable_state_dict(dual_state, initialize=True)


def test_new_writer_mode_and_action_contracts_are_validated():
    with pytest.raises(ValueError, match="writer_mode"):
        build_model(writer_mode="other")
    with pytest.raises(ValueError, match="writer_mode"):
        build_model(writer_mode="action")
    model = build_model(writer_mode="tag")
    with pytest.raises(ValueError, match="actions must align"):
        model.compress_batch([ids(3), ids(4)], actions=["append"])
    with pytest.raises(ValueError, match="actions must align"):
        model.compress(ids(3), action="other")
    limited = build_model(max_positions=5, writer_mode="tag")
    assert limited.compress(ids(3, 4), action="initial").shape == (3, limited.width)
    with pytest.raises(ValueError, match="exceeds model window"):
        limited.compress(ids(3, 4), action="append")


@pytest.mark.parametrize("tag_tokens", [0, -1, 1.5, True])
def test_invalid_tag_token_count_is_rejected(tag_tokens):
    with pytest.raises(ValueError, match="tag_tokens must be a positive integer"):
        build_model(writer_mode="tag", tag_tokens=tag_tokens)
