from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest

from cdic_repro.checkpoint import describe_state_dict, infer_lora_rank, unwrap_state_dict


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str = "float32") -> None:
        self.shape = shape
        self.dtype = dtype

    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


def test_unwrap_state_dict_accepts_plain_and_wrapped_mappings() -> None:
    plain = OrderedDict({"layer.weight": FakeTensor((2, 2))})

    assert unwrap_state_dict(plain) is plain
    assert unwrap_state_dict({"state_dict": plain}) is plain


def test_lora_rank_is_inferred_from_a_projection() -> None:
    state_dict = {
        "icae.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": (
            FakeTensor((64, 4096))
        ),
        "icae.base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": (
            FakeTensor((4096, 64))
        ),
        "icae.base_model.model.model.layers.0.self_attn.k_proj.weight": 0.0,
    }

    assert infer_lora_rank(state_dict) == 64
    schema = describe_state_dict(Path("checkpoint.pt"), state_dict)
    assert schema.checkpoint_entries == 3
    assert schema.tensor_count == 2
    assert schema.zero_placeholders == 1
    assert schema.total_numel == 2 * 64 * 4096
    assert schema.lora_rank == 64


def test_inconsistent_lora_ranks_are_rejected() -> None:
    state_dict = {
        "a.lora_A.default.weight": FakeTensor((64, 4096)),
        "b.lora_A.default.weight": FakeTensor((128, 4096)),
    }

    with pytest.raises(ValueError, match="inconsistent"):
        infer_lora_rank(state_dict)
