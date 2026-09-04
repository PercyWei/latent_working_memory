from __future__ import annotations

from pathlib import Path

import pytest

from cdic_repro.icae_adapter import IcaeV1AdapterConfig, format_turn, resolve_stop_token_id


def make_config(**overrides: object) -> IcaeV1AdapterConfig:
    values: dict[str, object] = {
        "model_path": Path("model"),
        "checkpoint_path": Path("checkpoint.pt"),
    }
    values.update(overrides)
    return IcaeV1AdapterConfig(**values)  # type: ignore[arg-type]


def test_turn_template_is_explicit_and_reproducible() -> None:
    config = make_config()

    assert format_turn(config.turn_template, query="hello", response="hi") == (
        "<s>[INST] hello [/INST] hi </s>"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"memory_size": 0},
        {"max_turn_tokens": 0},
        {"max_new_tokens": 0},
        {"lora_rank": 0},
        {"turn_template": "{query}"},
    ],
)
def test_invalid_adapter_config_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_config(**overrides)


def test_icae_stop_token_takes_precedence_over_tokenizer_eos() -> None:
    tokenizer = type("Tokenizer", (), {"eos_token_id": 2})()
    model = type("Model", (), {"eos_id": 1, "tokenizer": tokenizer})()

    assert resolve_stop_token_id(model) == 1
