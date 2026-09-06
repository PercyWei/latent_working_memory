from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


LORA_A_WEIGHT_SUFFIX = ".lora_A.default.weight"
ICAE_V1_LORA_RANK = 128


@dataclass(frozen=True, slots=True)
class CheckpointLoadReport:
    checkpoint_entries: int
    tensor_entries: int
    zero_placeholders_restored: int
    lora_rank: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def restore_zero_placeholder_checkpoint(
    checkpoint_state: Mapping[str, object],
    base_state: Mapping[str, object],
) -> tuple[OrderedDict[str, object], int]:
    """
    恢复 checkpoint 中以 0.0 表示的占位参数.

    注意:
    - 要求 checkpoint_state 与 base_state 的参数键完全一致.
    - checkpoint_state 中的 Tensor 参数直接保留.
    - checkpoint_state 中值为 0.0 的参数, 使用 base_state 中对应的原始参数进行替换.
    """
    missing = sorted(set(base_state) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(base_state))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint key mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    restored: OrderedDict[str, object] = OrderedDict()
    placeholder_count = 0
    for key, value in checkpoint_state.items():
        if isinstance(value, Tensor):
            restored[key] = value
        elif isinstance(value, float) and value == 0.0:
            restored[key] = base_state[key]
            placeholder_count += 1
        else:
            raise TypeError(f"unsupported checkpoint value for {key}: {type(value).__name__}")
    return restored, placeholder_count


def load_zero_placeholder_checkpoint(
    model: object,
    checkpoint_state: Mapping[str, object],
) -> CheckpointLoadReport:
    restored_state, placeholder_count = restore_zero_placeholder_checkpoint(
        checkpoint_state,
        model.state_dict(),
    )
    load_result = model.load_state_dict(restored_state, strict=True)
    return CheckpointLoadReport(
        checkpoint_entries=len(checkpoint_state),
        tensor_entries=sum(torch.is_tensor(value) for value in checkpoint_state.values()),
        zero_placeholders_restored=placeholder_count,
        lora_rank=ICAE_V1_LORA_RANK,
        missing_keys=tuple(load_result.missing_keys),
        unexpected_keys=tuple(load_result.unexpected_keys),
    )


def load_checkpoint_state_dict(checkpoint_path: Path) -> Mapping[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("ICAE checkpoint must be a direct state-dict mapping")
    lora_a_weights = 0
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            raise TypeError("ICAE checkpoint keys must be strings")
        if isinstance(value, torch.Tensor):
            if key.endswith(LORA_A_WEIGHT_SUFFIX):
                if value.ndim != 2 or value.shape[0] != ICAE_V1_LORA_RANK:
                    raise ValueError(
                        f"invalid ICAE v1 LoRA A weight shape for {key}: {tuple(value.shape)}"
                    )
                lora_a_weights += 1
            continue
        if isinstance(value, float) and value == 0.0:
            continue
        raise TypeError(f"unsupported ICAE checkpoint value for {key}")
    if lora_a_weights == 0:
        raise ValueError("ICAE v1 checkpoint contains no LoRA A weights")
    return checkpoint
