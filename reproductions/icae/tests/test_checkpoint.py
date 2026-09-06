from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from icae_repro.checkpoint import (
    ICAE_V1_LORA_RANK,
    load_checkpoint_state_dict,
    restore_zero_placeholder_checkpoint,
)


def test_checkpoint_loader_accepts_canonical_lora_rank(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    state = {
        "q_proj.lora_A.default.weight": torch.zeros(ICAE_V1_LORA_RANK, 4),
        "q_proj.lora_B.default.weight": torch.zeros(4, ICAE_V1_LORA_RANK),
        "v_proj.lora_A.default.weight": torch.zeros(ICAE_V1_LORA_RANK, 4),
    }
    torch.save(state, path)

    assert load_checkpoint_state_dict(path).keys() == state.keys()


def test_restore_zero_placeholder_checkpoint_uses_base_parameters() -> None:
    trained = torch.zeros(ICAE_V1_LORA_RANK, 4)
    base = torch.zeros(4, 4)
    checkpoint = OrderedDict(
        {
            "base.weight": 0.0,
            "q_proj.lora_A.default.weight": trained,
        }
    )
    base_state = {
        "base.weight": base,
        "q_proj.lora_A.default.weight": torch.zeros(ICAE_V1_LORA_RANK, 4),
    }

    restored, count = restore_zero_placeholder_checkpoint(checkpoint, base_state)

    assert count == 1
    assert restored["base.weight"] is base
    assert restored["q_proj.lora_A.default.weight"] is trained


def test_restore_rejects_key_mismatch() -> None:
    with pytest.raises(ValueError, match="key mismatch"):
        restore_zero_placeholder_checkpoint(
            {"checkpoint-only": 0.0},
            {"base-only": torch.zeros(1)},
        )


def test_checkpoint_loader_rejects_wrapped_state_dict(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"weight": torch.zeros(1)}}, path)

    with pytest.raises(TypeError, match="unsupported ICAE checkpoint value"):
        load_checkpoint_state_dict(path)
