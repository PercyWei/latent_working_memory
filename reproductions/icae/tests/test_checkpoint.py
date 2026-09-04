from __future__ import annotations

from collections import OrderedDict

import pytest

from icae_repro.checkpoint import infer_lora_rank, restore_zero_weight_state_dict


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def is_fake_tensor(value: object) -> bool:
    return isinstance(value, FakeTensor)


def test_infer_lora_rank_from_checkpoint() -> None:
    state = {
        "q_proj.lora_A.default.weight": FakeTensor((128, 4096)),
        "q_proj.lora_B.default.weight": FakeTensor((4096, 128)),
        "v_proj.lora_A.default.weight": FakeTensor((128, 4096)),
    }

    assert infer_lora_rank(state) == 128


def test_restore_zero_weight_checkpoint_uses_base_parameters() -> None:
    trained = FakeTensor((128, 4096))
    base = FakeTensor((4096, 4096))
    checkpoint = OrderedDict(
        {
            "base.weight": 0.0,
            "q_proj.lora_A.default.weight": trained,
        }
    )
    base_state = {
        "base.weight": base,
        "q_proj.lora_A.default.weight": FakeTensor((128, 4096)),
    }

    restored, count = restore_zero_weight_state_dict(
        checkpoint,
        base_state,
        is_tensor=is_fake_tensor,
    )

    assert count == 1
    assert restored["base.weight"] is base
    assert restored["q_proj.lora_A.default.weight"] is trained


def test_restore_rejects_key_mismatch() -> None:
    with pytest.raises(ValueError, match="key mismatch"):
        restore_zero_weight_state_dict(
            {"checkpoint-only": 0.0},
            {"base-only": FakeTensor((1,))},
            is_tensor=is_fake_tensor,
        )
