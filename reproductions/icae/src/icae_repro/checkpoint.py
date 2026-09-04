from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass


IsTensor = Callable[[object], bool]


@dataclass(frozen=True, slots=True)
class CheckpointLoadReport:
    checkpoint_entries: int
    tensor_entries: int
    zero_placeholders_restored: int
    lora_rank: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def infer_lora_rank(state_dict: Mapping[str, object]) -> int:
    ranks: set[int] = set()
    for key, value in state_dict.items():
        if ".lora_A." not in key or not key.endswith(".weight"):
            continue
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError(f"unexpected LoRA A value for {key}")
        ranks.add(int(shape[0]))
    if not ranks:
        raise ValueError("checkpoint contains no LoRA A weights")
    if len(ranks) != 1:
        raise ValueError(f"checkpoint contains inconsistent LoRA ranks: {sorted(ranks)}")
    return ranks.pop()


def restore_zero_weight_state_dict(
    checkpoint_state: Mapping[str, object],
    base_state: Mapping[str, object],
    *,
    is_tensor: IsTensor,
) -> tuple[OrderedDict[str, object], int]:
    missing = sorted(set(base_state) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(base_state))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint key mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    restored: OrderedDict[str, object] = OrderedDict()
    placeholder_count = 0
    for key, value in checkpoint_state.items():
        if is_tensor(value):
            restored[key] = value
        elif isinstance(value, float) and value == 0.0:
            restored[key] = base_state[key]
            placeholder_count += 1
        else:
            raise TypeError(f"unsupported checkpoint value for {key}: {type(value).__name__}")
    return restored, placeholder_count


def apply_zero_weight_checkpoint(
    model: object,
    checkpoint_state: object,
    torch_module: object,
) -> CheckpointLoadReport:
    if not isinstance(checkpoint_state, Mapping):
        raise TypeError("ICAE checkpoint must be a state-dict mapping")
    lora_rank = infer_lora_rank(checkpoint_state)
    restored_state, placeholder_count = restore_zero_weight_state_dict(
        checkpoint_state,
        model.state_dict(),
        is_tensor=torch_module.is_tensor,
    )
    load_result = model.load_state_dict(restored_state, strict=True)
    return CheckpointLoadReport(
        checkpoint_entries=len(checkpoint_state),
        tensor_entries=sum(torch_module.is_tensor(value) for value in checkpoint_state.values()),
        zero_placeholders_restored=placeholder_count,
        lora_rank=lora_rank,
        missing_keys=tuple(load_result.missing_keys),
        unexpected_keys=tuple(load_result.unexpected_keys),
    )


def load_zero_weight_checkpoint(
    model: object,
    checkpoint_path: str,
    torch_module: object,
) -> CheckpointLoadReport:
    checkpoint_state = torch_module.load(checkpoint_path, map_location="cpu")
    return apply_zero_weight_checkpoint(model, checkpoint_state, torch_module)
