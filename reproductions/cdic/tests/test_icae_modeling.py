from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from cdic_repro.icae import (
    IcaeConfig,
    LlamaICAE,
    load_icae_checkpoint,
    prepare_icae_token_layout,
    save_icae_checkpoint,
)


class TinyTokenizer:
    def __init__(self, vocabulary_size: int = 32) -> None:
        self.vocabulary_size = vocabulary_size
        self.pad_token_id: int | None = None
        self.eos_token_id: int | None = 2

    def __len__(self) -> int:
        return self.vocabulary_size

    def add_special_tokens(self, special_tokens: dict[str, str]) -> int:
        assert special_tokens == {"pad_token": "<|icae_pad|>"}
        self.pad_token_id = self.vocabulary_size
        self.vocabulary_size += 1
        return 1


def build_model() -> LlamaICAE:
    torch.manual_seed(7)
    base_model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    return LlamaICAE(
        base_model=base_model,
        tokenizer=TinyTokenizer(),  # type: ignore[arg-type]
        config=IcaeConfig(
            memory_size=4,
            lora_rank=2,
            lora_alpha=4,
            lora_dropout=0.0,
        ),
    )


def encoder_tokens(model: LlamaICAE) -> torch.Tensor:
    return torch.tensor(
        [[3, 4, *model.token_layout.memory_token_ids]],
        dtype=torch.long,
    )


def test_token_layout_uses_tokenizer_ids_and_preserves_published_offsets() -> None:
    tokenizer = TinyTokenizer()

    layout = prepare_icae_token_layout(tokenizer, memory_size=4)  # type: ignore[arg-type]

    assert tokenizer.pad_token_id == 32
    assert tokenizer.eos_token_id == 2
    assert layout.tokenizer_vocabulary_size == 33
    assert layout.memory_token_ids == [33, 34, 35, 36]
    assert layout.ae_token_id == 37
    assert layout.lm_token_id == 38
    assert layout.ft_token_id == 39
    assert layout.token_id_upper_bound == 40


def test_token_layout_requires_tokenizer_eos_token() -> None:
    tokenizer = TinyTokenizer()
    tokenizer.eos_token_id = None

    with pytest.raises(ValueError, match="eos_token_id"):
        prepare_icae_token_layout(tokenizer, memory_size=4)  # type: ignore[arg-type]


def test_training_forward_uses_modern_llama_and_peft() -> None:
    model = build_model()
    decoder_tokens = torch.tensor([[model.token_layout.ae_token_id, 5, 6]])
    labels = torch.tensor([[5, 6, model.tokenizer.eos_token_id]])

    outputs = model(
        encoder_tokens=encoder_tokens(model),
        decoder_tokens=decoder_tokens,
        labels=labels,
    )

    assert outputs.loss is not None
    assert outputs.loss.ndim == 0
    assert model.icae.get_input_embeddings().num_embeddings == len(model.tokenizer)
    assert model.token_layout.memory_token_start == len(model.tokenizer)
    assert model.token_layout.ft_token_id >= len(model.tokenizer)
    assert outputs.logits.shape == (1, 3, len(model.tokenizer))


@pytest.mark.parametrize(
    "encoder_tokens",
    [
        torch.tensor([[3, 4, 36, 35, 34, 33]]),
        torch.tensor([[33, 34, 35, 36, 3, 4]]),
    ],
)
def test_compress_requires_ordered_memory_token_suffix(encoder_tokens: torch.Tensor) -> None:
    model = build_model()

    with pytest.raises(ValueError, match="ordered ICAE memory token sequence"):
        model.compress(encoder_tokens)


def test_gradient_checkpointing_preserves_compressor_and_decoder_lora_routes() -> None:
    model = build_model()
    model.gradient_checkpointing_enable()
    model.train()

    latent = model.compress(encoder_tokens(model))
    latent.square().mean().backward()
    compressor_gradients = [
        parameter.grad for name, parameter in model.named_parameters() if ".lora_" in name
    ]
    assert any(
        gradient is not None and torch.count_nonzero(gradient) for gradient in compressor_gradients
    )

    model.zero_grad(set_to_none=True)
    decoder_embeddings = model.embed_tokens(torch.tensor([[3, 4]])).detach().requires_grad_()
    model.decode(decoder_embeddings).logits.square().mean().backward()
    decoder_gradients = [
        parameter.grad for name, parameter in model.named_parameters() if ".lora_" in name
    ]
    assert decoder_embeddings.grad is not None
    assert all(
        gradient is None or not torch.count_nonzero(gradient) for gradient in decoder_gradients
    )


def test_icae_checkpoint_round_trip_uses_only_trainable_parameters(tmp_path: Path) -> None:
    model = build_model()
    checkpoint_path = tmp_path / "icae.pt"
    expected_state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    save_icae_checkpoint(model, checkpoint_path)
    checkpoint_state = torch.load(checkpoint_path, weights_only=True)

    assert set(checkpoint_state) == set(expected_state)
    assert all(isinstance(value, torch.Tensor) for value in checkpoint_state.values())

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in expected_state:
                parameter.zero_()
    load_icae_checkpoint(model, checkpoint_path)

    restored_state = model.state_dict()
    assert all(torch.equal(restored_state[name], value) for name, value in expected_state.items())
