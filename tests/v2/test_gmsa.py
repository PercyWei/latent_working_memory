import torch
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from latent_working_memory.v2.gmsa_checkpoint import load_weights, save_model
from latent_working_memory.v2.gmsa_config import GMSAConfig
from latent_working_memory.v2.gmsa import GMSA, group_mean


def batch():
    return dict(
        context_ids=torch.tensor([[4, 5, 6, 7, 8], [8, 7, 6, 0, 0]]),
        context_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool),
        prompt_ids=torch.tensor([[10, 11], [12, 0]]),
        prompt_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        labels=torch.tensor([[4, 5, 2], [8, 2, -100]]),
    )


def test_group_pooling_remainder_padding_and_gradient():
    hidden = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4).requires_grad_()
    mask = batch()["context_mask"]
    result = group_mean(hidden, mask, 2)
    torch.testing.assert_close(
        result[0], torch.stack((hidden[0, :2].mean(0), hidden[0, 2:4].mean(0), hidden[0, 4]))
    )
    torch.testing.assert_close(result[1], torch.stack((hidden[1, :2].mean(0), hidden[1, 2])))
    sum(row.sum() for row in result).backward()
    assert hidden.grad[1, 3:].count_nonzero() == 0
    assert hidden.grad[0, 4].eq(1).all()
    with pytest.raises(ValueError, match="empty"):
        group_mean(hidden, torch.zeros_like(mask), 2)


def test_stage_freezing_and_ae_gradient_through_reader(model):
    model.train()
    model(**batch(), ratio=2).loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.encoder.named_parameters()
        if "lora_" in n
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.alignment.layers.parameters()
    )
    assert all(p.grad is None for p in model.decoder.parameters())
    assert all(p.grad is None for p in model.alignment.norm.parameters())
    model.zero_grad(set_to_none=True)
    model.set_stage("finetune")
    model.train()
    model(**batch(), ratio=4).loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.alignment.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())


def test_batch_positions_and_causal_targets(model):
    model.eval()
    inputs = batch()
    batched = model(**inputs, ratio=2)
    memories = model.encode(inputs["context_ids"], inputs["context_mask"], 2)
    nll, count = 0, 0
    for i in range(2):
        single = {key: value[i : i + 1] for key, value in inputs.items()}
        result = model(**single, ratio=2)
        valid_targets = (single["labels"] != -100).sum().item()
        nll += result.loss * valid_targets
        count += valid_targets
        prefix_len = len(memories[i]) + inputs["prompt_mask"][i].sum().item()
        torch.testing.assert_close(
            batched.logits[i, prefix_len - 1 : prefix_len + valid_targets - 1],
            result.logits[0, prefix_len - 1 : prefix_len + valid_targets - 1],
        )
    torch.testing.assert_close(batched.loss, nll / count)
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["labels"][0, 0] = 9
    other = model(**changed, ratio=2)
    torch.testing.assert_close(batched.logits[0, 4], other.logits[0, 4])
    assert not torch.allclose(batched.logits[0, 5], other.logits[0, 5])


def test_generation_batch_matches_single(model):
    model.eval()
    inputs = batch()
    memories = model.encode(inputs["context_ids"], inputs["context_mask"], 2)
    generated = model.generate(memories, inputs["prompt_ids"], inputs["prompt_mask"], 4, 2, 0)
    for i in range(2):
        single = model.generate(
            memories[i : i + 1],
            inputs["prompt_ids"][i : i + 1],
            inputs["prompt_mask"][i : i + 1],
            4,
            2,
            0,
        )
        valid = single[0].tolist()
        assert generated[i, : len(valid)].tolist() == valid


def test_strict_checkpoint_and_config(model, tmp_path):
    model.eval()
    expected = model(**batch(), ratio=2).logits.detach()
    save_model(model, tmp_path)
    with torch.no_grad():
        next(model.decoder.parameters()).add_(1)
    load_weights(model, tmp_path)
    torch.testing.assert_close(expected, model(**batch(), ratio=2).logits, rtol=0, atol=0)
    (tmp_path / "model.json").write_text(
        (tmp_path / "model.json").read_text().replace('"encoder_layers": 2', '"encoder_layers": 1')
    )
    with pytest.raises(ValueError, match="configuration"):
        load_weights(model, tmp_path)


def test_non_reentrant_checkpointing(model):
    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.train()
    assert model.decoder.training and model.decoder.is_gradient_checkpointing
    model(**batch(), ratio=2).loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.encoder.named_parameters()
        if "lora_" in n
    )


def test_qwen3_native_path(tmp_path):
    base = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    base.save_pretrained(tmp_path)
    model = GMSA(
        GMSAConfig(
            str(tmp_path),
            encoder_layers=2,
            alignment_layers=1,
            compression_ratios=(2,),
            lora_rank=2,
        )
    )
    result = model(**batch(), ratio=2)
    assert torch.isfinite(result.loss)
    result.loss.backward()
