from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cdic_repro.model_protocol import CdicTrainingAdapter


CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    epoch: int = 0
    next_episode_position: int = 0
    global_step: int = 0


def save_training_checkpoint(
    path: Path,
    *,
    model: CdicTrainingAdapter,
    optimizer: Any,
    progress: TrainingProgress,
    config_fingerprint: str,
    torch_module: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "version": CHECKPOINT_VERSION,
        "config_fingerprint": config_fingerprint,
        "progress": asdict(progress),
        "model_state": dict(model.trainable_state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch_module.get_rng_state(),
        "cuda_rng_state_all": (
            torch_module.cuda.get_rng_state_all() if torch_module.cuda.is_available() else None
        ),
    }
    torch_module.save(payload, temporary)
    temporary.replace(path)
    latest = path.parent / "latest.json"
    latest.write_text(
        json.dumps({"checkpoint": str(path), "progress": asdict(progress)}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_training_checkpoint(
    path: Path,
    *,
    model: CdicTrainingAdapter,
    optimizer: Any,
    expected_config_fingerprint: str,
    torch_module: Any,
) -> TrainingProgress:
    if not path.is_file():
        raise FileNotFoundError(f"training checkpoint does not exist: {path}")
    try:
        payload = torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch_module.load(path, map_location="cpu")
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
    torch_module.set_rng_state(payload["torch_rng_state"])
    cuda_rng_state = payload.get("cuda_rng_state_all")
    if cuda_rng_state is not None and torch_module.cuda.is_available():
        torch_module.cuda.set_rng_state_all(cuda_rng_state)
    return TrainingProgress(
        epoch=int(progress["epoch"]),
        next_episode_position=int(progress["next_episode_position"]),
        global_step=int(progress["global_step"]),
    )
