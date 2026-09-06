from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from cdic_repro.model_protocol import CdicTrainingAdapter, TrainableStateAdapter


CHECKPOINT_VERSION = 2


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    epoch: int = 0
    next_episode_position: int = 0
    global_step: int = 0


def save_training_checkpoint(
    path: Path,
    model: CdicTrainingAdapter,
    optimizer: Any,
    progress: TrainingProgress,
    config_fingerprint: str,
    rng_states: list[dict[str, object]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "version": CHECKPOINT_VERSION,
        "config_fingerprint": config_fingerprint,
        "progress": asdict(progress),
        "model_state": dict(model.trainable_state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "rng_states": rng_states or [_capture_rng_state()],
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    latest = path.parent / "latest.json"
    latest.write_text(
        json.dumps({"checkpoint": str(path), "progress": asdict(progress)}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_training_checkpoint(
    path: Path,
    model: CdicTrainingAdapter,
    optimizer: Any,
    expected_config_fingerprint: str,
    rank: int = 0,
) -> TrainingProgress:
    if not path.is_file():
        raise FileNotFoundError(f"training checkpoint does not exist: {path}")
    payload = _load_payload(path)
    if not isinstance(payload, dict):
        raise TypeError("training checkpoint must contain a mapping")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported training checkpoint version: {payload.get('version')}")
    if payload.get("config_fingerprint") != expected_config_fingerprint:
        raise ValueError("training checkpoint config fingerprint does not match")
    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    progress = payload.get("progress")
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        raise TypeError("training checkpoint is missing model or optimizer state")
    if not isinstance(progress, dict):
        raise TypeError("training checkpoint is missing progress")
    model.load_trainable_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    rng_states = payload.get("rng_states")
    if not isinstance(rng_states, list) or rank >= len(rng_states):
        raise ValueError("training checkpoint has no RNG state for this rank")
    _restore_rng_state(rng_states[rank])
    return TrainingProgress(
        epoch=int(progress["epoch"]),
        next_episode_position=int(progress["next_episode_position"]),
        global_step=int(progress["global_step"]),
    )


def load_model_from_training_checkpoint(
    path: Path,
    model: TrainableStateAdapter,
) -> TrainingProgress:
    """Restore only trainable model tensors for evaluation or inference."""
    if not path.is_file():
        raise FileNotFoundError(f"training checkpoint does not exist: {path}")
    payload = _load_payload(path)
    if not isinstance(payload, dict):
        raise TypeError("training checkpoint must contain a mapping")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported training checkpoint version: {payload.get('version')}")
    model_state = payload.get("model_state")
    progress = payload.get("progress")
    if not isinstance(model_state, dict):
        raise TypeError("training checkpoint is missing model state")
    if not isinstance(progress, dict):
        raise TypeError("training checkpoint is missing progress")
    model.load_trainable_state_dict(model_state, strict=True)
    return TrainingProgress(
        epoch=int(progress["epoch"]),
        next_episode_position=int(progress["next_episode_position"]),
        global_step=int(progress["global_step"]),
    )


def _load_payload(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _capture_rng_state() -> dict[str, object]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: object) -> None:
    if not isinstance(state, dict) or "torch" not in state:
        raise TypeError("invalid RNG state in training checkpoint")
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_state)
