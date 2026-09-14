from contextlib import nullcontext
from dataclasses import fields
import weakref

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from latent_working_memory.v1.backbone import LatentMemoryBackbone, ReadTokens
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.objectives import ReaderOutput
from reader_reference import dense_read_batch


def make_backbone(architecture):
    config_type, model_type = (
        (LlamaConfig, LlamaForCausalLM)
        if architecture == "llama"
        else (Qwen2Config, Qwen2ForCausalLM)
    )
    model = model_type(
        config_type(
            vocab_size=48,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    )
    backbone = LatentMemoryBackbone(model, 1, 2, 8, 2, 4, ("q_proj", "v_proj"), 0.0)
    with torch.no_grad():
        for name, parameter in backbone.language_model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.05)
    return backbone


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_packed_reader_matches_dense_loss_gradients_and_projection_rows(
    architecture, bf16, checkpointing
):
    torch.manual_seed(82)
    backbone = make_backbone(architecture)
    backbone.train()
    if checkpointing:
        backbone.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    writer = JointMemoryWriter(8, 1, 2, 16, 32)
    tasks = [
        ReadTokens((11, 12), (4, 5, 6, 2)),
        ReadTokens((11,), (7, 8, 2)),
        ReadTokens((12, 13, 14), (9, 10, 11, 12, 13, 2)),
    ]
    units = [(4, 5, 6, 7), (8, 9), (10, 11, 12)]
    weights = [0.25, 0.75, 1.0]
    lengths = []
    handle = backbone.language_model.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: lengths.append(tuple(args[0].shape))
    )
    try:
        observed = []
        for read in (dense_read_batch, lambda b, *a: b.read_batch(*a)):
            backbone.zero_grad(set_to_none=True)
            writer.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext():
                features = backbone.text_features(units, [0] * 3)
                states = writer.update_batch(
                    [writer.initialize_state(dtype=f.dtype) for f in features],
                    features,
                    [0] * 3,
                    [2, 5, 3],
                )
                outputs = read(backbone, [s.values for s in states], tasks)
                loss = sum(w * o.mean_nll for w, o in zip(weights, outputs)) / sum(weights)
            loss.backward()
            grads = {
                f"{label}.{n}": p.grad.detach().clone()
                for label, module in [("backbone", backbone), ("writer", writer)]
                for n, p in module.named_parameters()
                if p.grad is not None
            }
            observed.append(([o.token_nll.detach().clone() for o in outputs], grads))
        torch.testing.assert_close(observed[0][0], observed[1][0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            observed[0][1], observed[1][1], rtol=0.03 if bf16 else 5e-5, atol=4e-5 if bf16 else 2e-7
        )
        assert any("input_projection" in n for n in observed[1][1])
        assert any("memory_projection" in n for n in observed[1][1])
        assert any("lora_B" in n for n in observed[1][1])
        assert any(n.startswith("writer.") for n in observed[1][1])
        assert lengths[0][:2] == (3, 13)
        assert lengths[1] == (sum(len(t.target_ids) for t in tasks), 16)
        assert all(
            p.grad is None
            for n, p in backbone.language_model.named_parameters()
            if "lora_" not in n
        )
    finally:
        handle.remove()


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
@pytest.mark.parametrize("enabled", [True, False])
def test_raw_context_empty_memory_and_lora_restoration_match_dense(architecture, enabled):
    torch.manual_seed(8)
    backbone = make_backbone(architecture)
    backbone.train()
    memories = [torch.empty(0, 8), torch.empty(0, 8)]
    tasks = [ReadTokens((11, 12), (4, 5, 2)), ReadTokens((11,), (6, 7, 8, 9, 2))]
    contexts = [(4, 5, 6, 7), (8, 9)]
    expected = dense_read_batch(backbone, memories, tasks, contexts, enabled)
    actual = backbone.read_batch(memories, tasks, contexts, enabled)
    torch.testing.assert_close(
        [o.token_nll for o in expected], [o.token_nll for o in actual], rtol=1e-5, atol=1e-6
    )
    assert backbone.language_model.training
    assert not next(
        m for m in backbone.language_model.modules() if hasattr(m, "lora_A")
    ).disable_adapters
    if not enabled:
        assert all(not o.token_nll.requires_grad for o in actual)


def test_reader_output_does_not_keep_logits_alive(components):
    backbone, _ = components
    logits = []
    handle = backbone.language_model.get_output_embeddings().register_forward_hook(
        lambda module, args, output: logits.append(weakref.ref(output))
    )
    try:
        outputs = backbone.read_batch(
            [torch.randn(2, 8, requires_grad=True)], [ReadTokens((11,), (4, 5, 2))]
        )
        outputs[0].mean_nll.backward()
        assert [f.name for f in fields(ReaderOutput)] == ["token_nll"]
        assert logits[0]() is None
        assert outputs[0].target_length == 3
    finally:
        handle.remove()
