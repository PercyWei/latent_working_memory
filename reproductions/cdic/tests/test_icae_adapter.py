from __future__ import annotations

from pathlib import Path

import pytest

from cdic_repro.icae_adapter import (
    IcaeV1AdapterConfig,
    _is_trainable_icae_parameter,
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
