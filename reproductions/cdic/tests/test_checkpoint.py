from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cdic_repro.checkpoint import ICAE_V1_LORA_RANK, load_icae_checkpoint_state_dict
from cdic_repro.inspect_checkpoint import describe_state_dict


def test_checkpoint_loader_requires_direct_state_dict(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"layer.weight": torch.zeros(2, 2)}}, path)

    with pytest.raises(TypeError, match="unsupported ICAE checkpoint value"):
        load_icae_checkpoint_state_dict(path)


def test_checkpoint_schema_uses_canonical_icae_v1_rank() -> None:
    state_dict = {
        "icae.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": (
            torch.zeros(ICAE_V1_LORA_RANK, 4)
        ),
        "icae.base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": (
            torch.zeros(4, ICAE_V1_LORA_RANK)
        ),
        "icae.base_model.model.model.layers.0.self_attn.k_proj.weight": 0.0,
    }

    schema = describe_state_dict(Path("checkpoint.pt"), state_dict)
    assert schema.checkpoint_entries == 3
    assert schema.tensor_count == 2
    assert schema.zero_placeholders == 1
    assert schema.total_numel == 8 * ICAE_V1_LORA_RANK
    assert schema.lora_rank == ICAE_V1_LORA_RANK


def test_noncanonical_lora_rank_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"q_proj.lora_A.default.weight": torch.zeros(64, 4)}, path)

    with pytest.raises(ValueError, match="invalid ICAE v1 LoRA A weight shape"):
        load_icae_checkpoint_state_dict(path)
