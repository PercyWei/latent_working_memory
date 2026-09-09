from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.state import MemoryState


MODEL_CHECKPOINT_FIELDS = frozenset(
    {
        "phase",
        "config",
        "model_state",
        "optimizer_state",
        "progress",
        "rng_state",
    }
)
RUNTIME_MEMORY_FIELDS = frozenset({"model_checkpoint", "values", "seen_tokens"})


@dataclass(frozen=True, slots=True)
class LoadedModelCheckpoint:
    phase: str
    config: ExperimentConfig
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    progress: dict[str, Any]
    rng_state: dict[str, Any]


def capture_rng_state() -> dict[str, Any]:
    cuda_state = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": cuda_state,
    }


def restore_rng_state(rng_state: Mapping[str, Any]) -> None:
    _require_exact_fields(rng_state, {"python", "torch", "cuda"}, "rng_state")
    random.setstate(rng_state["python"])
    torch.set_rng_state(rng_state["torch"])
    cuda_state = tuple(rng_state["cuda"])
    if cuda_state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)


def save_model_checkpoint(
    path: str | Path,
    phase: str,
    config: ExperimentConfig,
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    progress: Mapping[str, Any],
    rng_state: Mapping[str, Any],
) -> None:
    if not phase:
        raise ValueError("phase must not be empty")
    _require_exact_fields(rng_state, {"python", "torch", "cuda"}, "rng_state")
    payload = {
        "phase": phase,
        "config": config.to_dict(),
        "model_state": dict(model_state),
        "optimizer_state": dict(optimizer_state),
        "progress": dict(progress),
        "rng_state": dict(rng_state),
    }
    _atomic_torch_save(payload, Path(path))


def load_model_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> LoadedModelCheckpoint:
    payload = _load_payload(Path(path), map_location)
    _require_exact_fields(payload, MODEL_CHECKPOINT_FIELDS, "model checkpoint")
    if not isinstance(payload["phase"], str) or not payload["phase"]:
        raise ValueError("checkpoint phase must be a non-empty string")
    config_raw = payload["config"]
    if not isinstance(config_raw, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    config = ExperimentConfig.from_mapping(config_raw)
    for field in ("model_state", "optimizer_state", "progress", "rng_state"):
        if not isinstance(payload[field], Mapping):
            raise TypeError(f"checkpoint {field} must be a mapping")
    _require_exact_fields(payload["rng_state"], {"python", "torch", "cuda"}, "rng_state")
    return LoadedModelCheckpoint(
        phase=payload["phase"],
        config=config,
        model_state=dict(payload["model_state"]),
        optimizer_state=dict(payload["optimizer_state"]),
        progress=dict(payload["progress"]),
        rng_state=dict(payload["rng_state"]),
    )


def save_runtime_memory(
    path: str | Path,
    model_checkpoint: str,
    state: MemoryState,
) -> None:
    if not model_checkpoint:
        raise ValueError("model_checkpoint must not be empty")
    if state.values.dtype != torch.bfloat16:
        raise TypeError("v1 runtime memory must use torch.bfloat16")
    payload = {
        "model_checkpoint": model_checkpoint,
        "values": state.values.detach().to(device="cpu"),
        "seen_tokens": state.seen_tokens,
    }
    _atomic_torch_save(payload, Path(path))


def load_runtime_memory(
    path: str | Path,
    expected_model_checkpoint: str,
    expected_width: int,
    map_location: str | torch.device = "cpu",
) -> MemoryState:
    if not expected_model_checkpoint:
        raise ValueError("expected_model_checkpoint must not be empty")
    if type(expected_width) is not int or expected_width <= 0:
        raise ValueError("expected_width must be a positive integer")
    payload = _load_payload(Path(path), map_location)
    _require_exact_fields(payload, RUNTIME_MEMORY_FIELDS, "runtime memory")
    if payload["model_checkpoint"] != expected_model_checkpoint:
        raise ValueError("runtime memory belongs to a different model checkpoint")
    values = payload["values"]
    if not isinstance(values, torch.Tensor):
        raise TypeError("runtime memory values must be a tensor")
    if values.dtype != torch.bfloat16:
        raise TypeError("v1 runtime memory must use torch.bfloat16")
    state = MemoryState(values=values, seen_tokens=payload["seen_tokens"])
    if state.width != expected_width:
        raise ValueError(f"runtime memory width must be {expected_width}")
    return state


def _load_payload(path: Path, map_location: str | torch.device) -> Mapping[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    return payload


def _require_exact_fields(
    mapping: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    label: str,
) -> None:
    actual = set(mapping)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unknown = sorted(actual - set(expected))
        raise ValueError(f"invalid {label} fields; missing={missing}, unknown={unknown}")


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
