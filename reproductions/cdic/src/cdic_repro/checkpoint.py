from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TensorRecord:
    key: str
    shape: tuple[int, ...]
    dtype: str
    numel: int


@dataclass(frozen=True, slots=True)
class CheckpointSchema:
    path: str
    checkpoint_entries: int
    tensor_count: int
    zero_placeholders: int
    total_numel: int
    lora_rank: int
    tensors: tuple[TensorRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "checkpoint_entries": self.checkpoint_entries,
            "tensor_count": self.tensor_count,
            "zero_placeholders": self.zero_placeholders,
            "total_numel": self.total_numel,
            "lora_rank": self.lora_rank,
            "tensors": [
                {
                    "key": tensor.key,
                    "shape": list(tensor.shape),
                    "dtype": tensor.dtype,
                    "numel": tensor.numel,
                }
                for tensor in self.tensors
            ],
        }


def unwrap_state_dict(checkpoint: object) -> Mapping[str, object]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping or contain a state_dict mapping")
    for wrapper_key in ("state_dict", "model_state_dict", "model"):
        wrapped = checkpoint.get(wrapper_key)
        if isinstance(wrapped, Mapping) and all(isinstance(key, str) for key in wrapped):
            return wrapped
    if not all(isinstance(key, str) for key in checkpoint):
        raise TypeError("state-dict keys must be strings")
    return checkpoint


def infer_lora_rank(state_dict: Mapping[str, object]) -> int:
    ranks: set[int] = set()
    for key, tensor in state_dict.items():
        if ".lora_A." not in key or not key.endswith(".weight"):
            continue
        shape = _shape_of(tensor)
        if len(shape) != 2:
            raise ValueError(f"unexpected LoRA A shape for {key}: {shape}")
        ranks.add(shape[0])
    if not ranks:
        raise ValueError("checkpoint contains no LoRA A weights")
    if len(ranks) != 1:
        raise ValueError(f"checkpoint contains inconsistent LoRA ranks: {sorted(ranks)}")
    return ranks.pop()


def describe_state_dict(path: Path, state_dict: Mapping[str, object]) -> CheckpointSchema:
    tensors: list[TensorRecord] = []
    zero_placeholders = 0
    for key, tensor in state_dict.items():
        if isinstance(tensor, float) and tensor == 0.0:
            zero_placeholders += 1
            continue
        shape = _shape_of(tensor)
        numel = _numel_of(tensor, shape)
        tensors.append(
            TensorRecord(
                key=key,
                shape=shape,
                dtype=str(getattr(tensor, "dtype", "unknown")),
                numel=numel,
            )
        )
    tensors.sort(key=lambda tensor: tensor.key)
    return CheckpointSchema(
        path=str(path),
        checkpoint_entries=len(state_dict),
        tensor_count=len(tensors),
        zero_placeholders=zero_placeholders,
        total_numel=sum(tensor.numel for tensor in tensors),
        lora_rank=infer_lora_rank(state_dict),
        tensors=tuple(tensors),
    )


def load_checkpoint_state_dict(path: Path) -> Mapping[str, object]:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    return unwrap_state_dict(checkpoint)


def inspect_checkpoint(path: Path) -> tuple[Mapping[str, object], CheckpointSchema]:
    state_dict = load_checkpoint_state_dict(path)
    return state_dict, describe_state_dict(path, state_dict)


def _shape_of(tensor: object) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise TypeError("state dict contains a non-tensor value without shape")
    return tuple(int(dimension) for dimension in shape)


def _numel_of(tensor: object, shape: tuple[int, ...]) -> int:
    numel = getattr(tensor, "numel", None)
    if callable(numel):
        return int(numel())
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an ICAE checkpoint before model loading")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    _, schema = inspect_checkpoint(arguments.checkpoint)
    serialized = json.dumps(schema.to_dict(), indent=2, sort_keys=True)
    if arguments.output is None:
        print(serialized)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
