from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
FRAMEWORK_VERSION = "v1"
GROWTH_ACTIONS = (0, 8, 16)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    schema_version: int = SCHEMA_VERSION
    framework_version: str = FRAMEWORK_VERSION
    model_name_or_path: str = "/data/bywei/models/meta-llama/Llama-2-7b-chat-hf"
    model_revision: str | None = None
    teacher_model_name_or_path: str = "/data/bywei/models/meta-llama/Llama-2-7b-chat-hf"
    teacher_model_revision: str | None = None
    distill_mode: str = "token_kl"
    distill_temperature: float = 1.0
    reader_lora_rank: int = 16
    reader_lora_alpha: int = 32
    reader_lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    reader_lora_dropout: float = 0.0
    d_mem: int = 512
    num_layers: int = 3
    num_heads: int = 8
    ffn_dim: int = 2048
    cell_tokens: int = 64
    update_chunk_cells: tuple[int, ...] = (1, 2, 4)
    k_init: int = 16
    k_limit: int = 512
    growth_actions: tuple[int, ...] = GROWTH_ACTIONS
    exploration_probs: tuple[float, ...] = (0.70, 0.25, 0.05)
    data_seed: int = 20260907
    model_seed: int = 42
    train_episodes: int = 256
    dev_episodes: int = 64
    test_episodes: int = 64
    min_episode_tokens: int = 512
    max_episode_tokens: int = 2048
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    bptt_cells: int = 8
    supervise_every_cells: int = 4
    probes_per_prefix: int = 3
    lambda_distill: float = 0.1
    lambda_partition: float = 0.1
    partition_every_episodes: int = 4
    capacity_label_roots: int = 64
    capacity_horizon_cells: int = 4
    capacity_score_offsets: tuple[int, ...] = (0, 2, 4)
    cost_reference_slots: int = 64
    state_cost_weight: float = 0.05
    write_cost_weight: float = 0.01
    read_cost_weight: float = 0.05
    outer_rounds: int = 2
    max_new_tokens: int = 64

    def __post_init__(self) -> None:
        _validate_config(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
        field_names = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - field_names)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {unknown}")

        normalized = dict(raw)
        for name in (
            "reader_lora_target_modules",
            "update_chunk_cells",
            "growth_actions",
            "exploration_probs",
            "capacity_score_offsets",
        ):
            if name in normalized:
                value = normalized[name]
                if not isinstance(value, (list, tuple)):
                    raise TypeError(f"{name} must be an array")
                normalized[name] = tuple(value)
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("The configuration root must be a JSON object")
    return ExperimentConfig.from_mapping(raw)


def write_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _validate_config(config: ExperimentConfig) -> None:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if config.framework_version != FRAMEWORK_VERSION:
        raise ValueError(f"framework_version must be {FRAMEWORK_VERSION!r}")
    for name in ("model_name_or_path", "teacher_model_name_or_path"):
        value = getattr(config, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    for name in ("model_revision", "teacher_model_revision"):
        value = getattr(config, name)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{name} must be null or a non-empty string")
    if config.teacher_model_name_or_path != config.model_name_or_path:
        raise ValueError("v1 teacher and student must use the same base model")
    if config.teacher_model_revision != config.model_revision:
        raise ValueError("v1 teacher and student must use the same model revision")
    if config.distill_mode != "token_kl":
        raise ValueError("v1 only supports distill_mode='token_kl'")

    positive_ints = (
        "reader_lora_rank",
        "reader_lora_alpha",
        "d_mem",
        "num_layers",
        "num_heads",
        "ffn_dim",
        "cell_tokens",
        "k_init",
        "k_limit",
        "train_episodes",
        "dev_episodes",
        "test_episodes",
        "min_episode_tokens",
        "max_episode_tokens",
        "bptt_cells",
        "supervise_every_cells",
        "probes_per_prefix",
        "partition_every_episodes",
        "capacity_label_roots",
        "capacity_horizon_cells",
        "cost_reference_slots",
        "outer_rounds",
        "max_new_tokens",
    )
    for name in positive_ints:
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if config.d_mem % config.num_heads != 0:
        raise ValueError("d_mem must be divisible by num_heads")
    if config.k_init != 16:
        raise ValueError("v1 k_init must be 16")
    if config.k_init > config.k_limit:
        raise ValueError("k_init must not exceed k_limit")
    if config.k_limit % 8 != 0:
        raise ValueError("k_limit must be divisible by 8")
    if config.min_episode_tokens > config.max_episode_tokens:
        raise ValueError("min_episode_tokens must not exceed max_episode_tokens")
    if config.supervise_every_cells > config.bptt_cells:
        raise ValueError("supervise_every_cells must not exceed bptt_cells")

    _require_positive_tuple("update_chunk_cells", config.update_chunk_cells)
    if tuple(sorted(set(config.update_chunk_cells))) != config.update_chunk_cells:
        raise ValueError("update_chunk_cells must be sorted and unique")

    if config.growth_actions != GROWTH_ACTIONS:
        raise ValueError("v1 growth_actions must be [0, 8, 16]")
    if len(config.exploration_probs) != len(config.growth_actions):
        raise ValueError("exploration_probs must align with growth_actions")
    if any(
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or probability < 0.0
        for probability in config.exploration_probs
    ):
        raise ValueError("exploration_probs must contain finite non-negative numbers")
    if not math.isclose(sum(config.exploration_probs), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("exploration_probs must sum to 1")

    if not config.reader_lora_target_modules or any(
        not isinstance(module, str) or not module for module in config.reader_lora_target_modules
    ):
        raise ValueError("reader_lora_target_modules must contain non-empty names")
    if len(set(config.reader_lora_target_modules)) != len(config.reader_lora_target_modules):
        raise ValueError("reader_lora_target_modules must be unique")

    non_negative_floats = (
        "reader_lora_dropout",
        "weight_decay",
        "lambda_distill",
        "lambda_partition",
        "state_cost_weight",
        "write_cost_weight",
        "read_cost_weight",
    )
    positive_floats = ("distill_temperature", "learning_rate", "gradient_clip")
    for name in non_negative_floats:
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")
    for name in positive_floats:
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")
    if config.reader_lora_dropout >= 1.0:
        raise ValueError("reader_lora_dropout must be less than 1")

    offsets = config.capacity_score_offsets
    if not offsets or offsets[0] != 0:
        raise ValueError("capacity_score_offsets must start at 0")
    if any(type(offset) is not int or offset < 0 for offset in offsets):
        raise ValueError("capacity_score_offsets must contain non-negative integers")
    if tuple(sorted(set(offsets))) != offsets:
        raise ValueError("capacity_score_offsets must be sorted and unique")
    if offsets[-1] > config.capacity_horizon_cells:
        raise ValueError("capacity_score_offsets cannot exceed capacity_horizon_cells")


def _require_positive_tuple(name: str, values: tuple[int, ...]) -> None:
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
