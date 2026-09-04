from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cdic_repro.config import RetrievalConfig, SupportOrder
from cdic_repro.icae_adapter import IcaeV1AdapterConfig


@dataclass(frozen=True, slots=True)
class MscDataConfig:
    root: Path
    split: str = "train"
    session_id: int = 4
    max_episodes: int | None = None
    max_turns_per_episode: int | None = None
    strict_pairs: bool = False

    def __post_init__(self) -> None:
        if self.split not in {"train", "valid", "test"}:
            raise ValueError("data.split must be train, valid, or test")
        if not 2 <= self.session_id <= 5:
            raise ValueError("data.session_id must be between 2 and 5")
        if self.split == "train" and self.session_id == 5:
            raise ValueError("official MSC session 5 has no training split")
        if self.max_episodes is not None and self.max_episodes < 1:
            raise ValueError("data.max_episodes must be positive")
        if self.max_turns_per_episode is not None and self.max_turns_per_episode < 1:
            raise ValueError("data.max_turns_per_episode must be positive")


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    output_dir: Path
    epochs: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    seed: int = 42
    shuffle: bool = True
    max_grad_norm: float | None = None
    save_every_steps: int = 50
    keep_last_checkpoints: int | None = 2
    save_final_checkpoint: bool = True
    log_every_steps: int = 1
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("training.epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("training.weight_decay must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("training.max_grad_norm must be positive")
        if self.save_every_steps < 1:
            raise ValueError("training.save_every_steps must be positive")
        if self.keep_last_checkpoints is not None and self.keep_last_checkpoints < 1:
            raise ValueError("training.keep_last_checkpoints must be positive")
        if self.log_every_steps < 1:
            raise ValueError("training.log_every_steps must be positive")


@dataclass(frozen=True, slots=True)
class CdicMscTrainingConfig:
    model: IcaeV1AdapterConfig
    data: MscDataConfig
    retrieval: RetrievalConfig
    training: OptimizationConfig

    def to_dict(self) -> dict[str, object]:
        serialized = asdict(self)
        return _serialize_paths(serialized)

    def fingerprint(self) -> str:
        serialized = self.to_dict()
        training = dict(serialized["training"])  # type: ignore[arg-type]
        training["resume_from"] = None
        serialized["training"] = training
        payload = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_training_config(path: Path) -> CdicMscTrainingConfig:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError("training config must contain a JSON object")

    model = _mapping(payload, "model")
    data = _mapping(payload, "data")
    retrieval = _mapping(payload, "retrieval")
    training = _mapping(payload, "training")
    return CdicMscTrainingConfig(
        model=IcaeV1AdapterConfig(
            model_path=Path(_required_string(model, "model_path")),
            checkpoint_path=Path(_required_string(model, "checkpoint_path")),
            device=str(model.get("device", "cuda:0")),
            devices=_device_tuple(model.get("devices")),
            dtype=str(model.get("dtype", "bfloat16")),
            memory_size=int(model.get("memory_size", 128)),
            max_turn_tokens=int(model.get("max_turn_tokens", 512)),
            max_new_tokens=int(model.get("max_new_tokens", 128)),
            lora_alpha=int(model.get("lora_alpha", 32)),
            lora_dropout=float(model.get("lora_dropout", 0.05)),
            lora_rank=_optional_int(model.get("lora_rank")),
            seed=int(training.get("seed", 42)),
            use_ft_markers=bool(model.get("use_ft_markers", True)),
            turn_template=str(
                model.get("turn_template", "<s>[INST] {query} [/INST] {response} </s>")
            ),
            gradient_checkpointing=bool(model.get("gradient_checkpointing", True)),
        ),
        data=MscDataConfig(
            root=Path(_required_string(data, "root")),
            split=str(data.get("split", "train")),
            session_id=int(data.get("session_id", 4)),
            max_episodes=_optional_int(data.get("max_episodes")),
            max_turns_per_episode=_optional_int(data.get("max_turns_per_episode")),
            strict_pairs=bool(data.get("strict_pairs", False)),
        ),
        retrieval=RetrievalConfig(
            threshold=float(retrieval.get("threshold", 0.8)),
            decay=float(retrieval.get("decay", 0.05)),
            support_order=SupportOrder(str(retrieval.get("support_order", "score_desc"))),
            max_retrieved=_optional_int(retrieval.get("max_retrieved")),
        ),
        training=OptimizationConfig(
            output_dir=Path(_required_string(training, "output_dir")),
            epochs=int(training.get("epochs", 2)),
            learning_rate=float(training.get("learning_rate", 2e-4)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            seed=int(training.get("seed", 42)),
            shuffle=bool(training.get("shuffle", True)),
            max_grad_norm=_optional_float(training.get("max_grad_norm")),
            save_every_steps=int(training.get("save_every_steps", 50)),
            keep_last_checkpoints=_optional_int(training.get("keep_last_checkpoints", 2)),
            save_final_checkpoint=bool(training.get("save_final_checkpoint", True)),
            log_every_steps=int(training.get("log_every_steps", 1)),
            resume_from=_optional_path(training.get("resume_from")),
        ),
    )


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"training config field {key!r} must be an object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"training config field {key!r} must be a non-empty string")
    return value


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("resume_from must be null or a non-empty path")
    return Path(value)


def _device_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("model.devices must be a non-empty list of device strings")
    return tuple(value)


def _serialize_paths(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_paths(item) for item in value]
    if isinstance(value, SupportOrder):
        return value.value
    return value
