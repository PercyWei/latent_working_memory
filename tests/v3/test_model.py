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
    return GistMemoryModel(base, memory_slots, 2, 4, ("q_proj", "v_proj"))


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
    assert not model.language_model.training
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
def test_reader_disables_adapter_without_detaching_memory(model_type):
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
    head = model.language_model.get_output_embeddings()
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
    with model.language_model.disable_adapter():
        base = model.language_model.get_base_model()
        for memory, prompt, answer in zip(memories, prompts, answers, strict=True):
            embeddings = torch.cat(
                (memory, base.get_input_embeddings()(torch.cat((prompt, answer))))
            )
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
    projected, adapters_disabled = [], []

    def observe_head(module, args):
        projected.append(len(args[0]))
        adapters_disabled.append(
            all(
                module.disable_adapters
                for module in model.language_model.modules()
                if hasattr(module, "lora_A")
            )
        )

    hook = model.language_model.get_output_embeddings().register_forward_pre_hook(observe_head)
    actual = model.answer_nll(memories, prompts, answers)
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    assert calls == [256, 86]
    assert projected == [256, 86]
    actual.mean().backward()
    hook.remove()
    assert sorted(projected) == [86, 86, 256, 256]
    assert all(adapters_disabled)
    chunked_gradients = [memory.grad.clone() for memory in memories]
    for memory in memories:
        memory.grad = None
    expected = []
    with model.language_model.disable_adapter():
        base = model.language_model.get_base_model()
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
    assert model.answer_nll([torch.zeros(9, 16)], [ids(3, 4)], [ids(5, 6)]).shape == (1,)


def test_invalid_writer_blocks_and_empty_answers_are_rejected():
    model = build_model()
    with pytest.raises(ValueError, match="memory_slots"):
        model.compress(ids(3, 4), [torch.zeros(2, 16)])
    with pytest.raises(ValueError, match="nonempty"):
        model.answer_nll([torch.zeros(3, 16)], [ids(3, 4)], [ids()])
    with pytest.raises(ValueError, match="align"):
        model.answer_nll([torch.zeros(3, 16)], [], [])


def test_unsupported_dropout_and_native_gradient_checkpointing_are_explicit():
    base = build_model().language_model.get_base_model()
    with pytest.raises(ValueError, match="lora_dropout=0"):
        GistMemoryModel(base, 3, 2, 4, ("q_proj",), lora_dropout=0.1)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with pytest.raises(ValueError, match="native gradient checkpointing"):
        GistMemoryModel(base, 3, 2, 4, ("q_proj",))


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
@pytest.mark.parametrize("attention_implementation", ["eager", "sdpa"])
def test_ragged_writer_batch_matches_single_writes_outputs_and_all_gradients(
    model_type, attention_implementation
):
    batched = build_model(model_type, attention_implementation=attention_implementation)
    randomize_adapter(batched)
    separate = copy.deepcopy(batched)
    texts = [ids(3, 4, 5, 6), ids(7), ids(8, 9)]
    histories = [
        [],
        [torch.randn(3, batched.width, requires_grad=True)],
        [torch.randn(3, batched.width, requires_grad=True) for _ in range(2)],
    ]
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
        actual = batched.compress_batch(texts, histories)
    finally:
        before.remove()
        after.remove()
    expected = [
        separate.compress(text, blocks)
        for text, blocks in zip(texts, separate_histories, strict=True)
    ]
    assert len(calls) == 1
    assert calls[0]["inputs_embeds"].shape == (3, 11, batched.width)
    lengths = [7, 7, 11]
    for index, length in enumerate(lengths):
        assert calls[0]["attention_mask"][index].tolist() == [True] * length + [False] * (
            11 - length
        )
        assert calls[0]["position_ids"][index].tolist() == list(range(length)) + [0] * (11 - length)
        torch.testing.assert_close(actual[index], states[0][index, length - 3 : length])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    weights = torch.randn(len(actual), batched.memory_slots, batched.width)
    (torch.stack(actual) * weights).sum().backward()
    (torch.stack(expected) * weights).sum().backward()
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
        with pytest.raises(ValueError, match="memory_slots"):
            model.compress_batch([ids(3), ids(4)], [[], [torch.zeros(2, model.width)]])
        with pytest.raises(ValueError, match="nonempty"):
            model.compress_batch([ids(3), ids()])
        with pytest.raises(ValueError, match="exceeds model window"):
            model.compress_batch([ids(3), torch.arange(10)])
    finally:
        hook.remove()
    assert not calls
