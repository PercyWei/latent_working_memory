from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from cdic_repro.credit import CreditPlan
from cdic_repro.icae import IcaeConfig, LlamaICAE
from cdic_repro.icae_adapter import (
    IcaeV1AdapterConfig,
    IcaeV1TrainingAdapter,
    _is_trainable_icae_parameter,
)
from cdic_repro.memory_state import ThreadState


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

    def __call__(
        self,
        text: str,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
        padding: bool,
        return_attention_mask: bool,
    ) -> dict[str, list[int]]:
        del truncation, padding, return_attention_mask
        token_ids = [3 + index % 20 for index, _ in enumerate(text.split())] or [3]
        if add_special_tokens:
            token_ids.insert(0, 1)
        return {"input_ids": token_ids[:max_length]}


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


def make_config(**overrides: object) -> IcaeV1AdapterConfig:
    values: dict[str, object] = {
        "model_path": Path("model"),
        "checkpoint_path": Path("checkpoint.pt"),
    }
    values.update(overrides)
    return IcaeV1AdapterConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"memory_size": 0},
        {"max_turn_tokens": 0},
        {"max_new_tokens": 0},
        {"dtype": "float16"},
        {"lora_rank": 0},
        {"gradient_window_size": 0},
        {"turn_template": "{query}"},
    ],
)
def test_invalid_adapter_config_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_config(**overrides)


def test_training_parameter_filter_keeps_only_lora_and_compression_tokens() -> None:
    assert _is_trainable_icae_parameter("icae.base_model.model.q_proj.lora_A.default.weight")
    assert _is_trainable_icae_parameter("icae.base_model.model.q_proj.lora_B.default.weight")
    assert _is_trainable_icae_parameter("memory_token_embed.weight")
    assert not _is_trainable_icae_parameter("icae.base_model.model.q_proj.base_layer.weight")


def test_training_adapter_uses_modern_icae_encoder_and_decoder() -> None:
    config = make_config(
        device="cpu",
        memory_size=4,
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
    )
    adapter = IcaeV1TrainingAdapter(config, build_model())
    compressed = adapter.compress_gold(
        (),
        query="where is the key",
        response="in the drawer",
        credit=CreditPlan(connected_state_id=None, detached_state_ids=()),
    )
    state = ThreadState(
        state_id="state-1",
        thread_id="thread-1",
        revision=0,
        latent=compressed.latent,
        retrieval_key=compressed.retrieval_key,
        created_turn=1,
        written_turn=1,
        last_retrieved_turn=1,
    )

    loss = adapter.response_loss(
        (state,),
        query="where is the key",
        response="in the drawer",
        credit=CreditPlan(connected_state_id=state.state_id, detached_state_ids=()),
        collect_token_nll=True,
    )
    loss.value.backward()

    assert compressed.latent.shape == (4, 16)
    assert loss.value.ndim == 0
    assert loss.token_nll is not None
    assert len(loss.token_nll) == loss.token_count
    assert any(
        parameter.grad is not None
        for name, parameter in adapter.model.named_parameters()
        if ".lora_" in name
    )
