from __future__ import annotations

from pathlib import Path

import pytest

from icae_repro.inference import InferenceConfig, resolve_stop_token_id


def make_config(**overrides: object) -> InferenceConfig:
    values: dict[str, object] = {
        "model_path": Path("model"),
        "checkpoint_path": Path("checkpoint.pt"),
        "context": "context",
        "prompt": "prompt",
    }
    values.update(overrides)
    return InferenceConfig(**values)  # type: ignore[arg-type]


def test_inference_defaults_match_icae_v1() -> None:
    config = make_config()

    assert config.memory_size == 128
    assert config.model_max_length == 512
    assert config.repeat == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"memory_size": 0},
        {"model_max_length": 0},
        {"max_new_tokens": 0},
        {"repeat": 0},
    ],
)
def test_invalid_inference_config_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_config(**overrides)


def test_icae_stop_token_takes_precedence_over_tokenizer_eos() -> None:
    tokenizer = type("Tokenizer", (), {"eos_token_id": 2})()
    model = type("Model", (), {"eos_id": 1, "tokenizer": tokenizer})()

    assert resolve_stop_token_id(model) == 1
