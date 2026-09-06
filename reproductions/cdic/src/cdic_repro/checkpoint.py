from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from icae_repro.checkpoint import ICAE_V1_LORA_RANK, load_checkpoint_state_dict


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


def describe_state_dict(path: Path, state_dict: Mapping[str, object]) -> CheckpointSchema:
    tensors: list[TensorRecord] = []
    zero_placeholders = 0
    for key, tensor in state_dict.items():
        if isinstance(tensor, float) and tensor == 0.0:
            zero_placeholders += 1
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"unsupported ICAE checkpoint value for {key}")
        tensors.append(
            TensorRecord(
                key=key,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                numel=tensor.numel(),
            )
        )
    tensors.sort(key=lambda tensor: tensor.key)
    return CheckpointSchema(
        path=str(path),
        checkpoint_entries=len(state_dict),
        tensor_count=len(tensors),
        zero_placeholders=zero_placeholders,
        total_numel=sum(tensor.numel for tensor in tensors),
        lora_rank=ICAE_V1_LORA_RANK,
        tensors=tuple(tensors),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an ICAE checkpoint before model loading")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    state_dict = load_checkpoint_state_dict(arguments.checkpoint)
    schema = describe_state_dict(arguments.checkpoint, state_dict)
    serialized = json.dumps(asdict(schema), indent=2, sort_keys=True)
    if arguments.output is None:
        print(serialized)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
