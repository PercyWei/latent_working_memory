from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

GROWTH_ACTIONS = (0, 8, 16)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf"
    model_revision: str | None = None
    reader_lora_rank: int = 16
    reader_lora_alpha: int = 32
    reader_lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    reader_lora_dropout: float = 0.0
    d_mem: int = 512
    num_layers: int = 3
    num_heads: int = 8
    ffn_dim: int = 2048
    dynamic_k_first: int = 16
    k_limit: int = 4096
    growth_actions: tuple[int, ...] = GROWTH_ACTIONS
    exploration_probs: tuple[float, ...] = (0.70, 0.25, 0.05)
    bptt_tokens: int = 1024
    pretrain_dataset: str = "HuggingFaceFW/fineweb"
    pretrain_subset: str = "sample-10BT"
    split_fractions: tuple[float, ...] = (0.90, 0.05, 0.05)
    input_length_bounds: tuple[int, ...] = (32, 128, 512, 1024)
    input_length_weights: tuple[float, ...] | None = None
    input_length_weights_end: tuple[float, ...] | None = None
    input_length_curriculum_steps: int = 0
    pretrain_compression_ratios: tuple[int, ...] = (2, 4, 8)
    pretrain_k_min: int = 1
    ratio_weights_start: tuple[float, ...] = (0.45, 0.45, 0.10)
    ratio_weights_end: tuple[float, ...] = (0.20, 0.30, 0.50)
    ratio_curriculum_steps: int = 1000
    max_input_tokens: int = 1024
    max_continuation_tokens: int = 256
    write_context_tokens: int = 4096
    read_context_tokens: int = 4096
    ae_prompt: str = "Reconstruct the text stored in memory:\n"
    lm_prompt: str = "Continue the text stored in memory:\n"
    ae_weight: float = 1.0
    lm_weight: float = 1.0
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0001
    warmup_steps: int = 0
    lr_decay_steps: int = 0
    min_lr_fraction: float = 0.1
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    gradient_checkpointing: bool = True
    data_seed: int = 20260907
    model_seed: int = 42
    eval_every: int = 100
    eval_examples: int = 16
    eval_generation_examples: int = 4
    eval_generation_every: int = 1000

    def __post_init__(self) -> None:
        non_negative_ints = {
            "data_seed",
            "model_seed",
            "eval_generation_examples",
            "input_length_curriculum_steps",
            "warmup_steps",
            "lr_decay_steps",
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type == "int":
                minimum = 0 if field.name in non_negative_ints else 1
                if type(value) is not int or value < minimum:
                    raise ValueError(f"{field.name} must be an integer >= {minimum}")
        for name in (
            "model_name_or_path",
            "pretrain_dataset",
            "pretrain_subset",
            "ae_prompt",
            "lm_prompt",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.model_revision is not None and not isinstance(self.model_revision, str):
            raise ValueError("model_revision must be null or a string")
        if self.pretrain_dataset != "HuggingFaceFW/fineweb":
            raise ValueError("pretrain_dataset must be HuggingFaceFW/fineweb")
        if self.d_mem % self.num_heads:
            raise ValueError("d_mem must be divisible by num_heads")
        if max(self.dynamic_k_first, self.pretrain_k_min) > self.k_limit:
            raise ValueError("initial capacities must not exceed k_limit")
        if self.max_input_tokens + 1 > self.write_context_tokens:
            raise ValueError("max_input_tokens plus BOS exceeds write_context_tokens")
        if self.growth_actions != GROWTH_ACTIONS:
            raise ValueError("growth_actions must be [0, 8, 16]")
        ratios = self.pretrain_compression_ratios
        if not ratios or any(type(r) is not int or r <= 0 for r in ratios):
            raise ValueError("pretrain_compression_ratios must contain positive integers")
        if tuple(sorted(set(ratios))) != ratios:
            raise ValueError("pretrain_compression_ratios must be sorted and unique")
        for name, size in (
            ("split_fractions", 3),
            ("exploration_probs", 3),
            ("ratio_weights_start", len(ratios)),
            ("ratio_weights_end", len(ratios)),
        ):
            weights = getattr(self, name)
            if (
                len(weights) != size
                or any(
                    isinstance(w, bool)
                    or not isinstance(w, (int, float))
                    or not math.isfinite(w)
                    or w < 0
                    for w in weights
                )
                or not math.isclose(sum(weights), 1.0, abs_tol=1e-8)
            ):
                raise ValueError(f"{name} must have {size} non-negative weights that sum to 1")
        if min(self.split_fractions) <= 0:
            raise ValueError("split_fractions must allocate train, dev and test")
        bounds = self.input_length_bounds
        if (
            not bounds
            or tuple(sorted(set(bounds))) != bounds
            or any(type(bound) is not int or bound <= 0 for bound in bounds)
        ):
            raise ValueError("input_length_bounds must be increasing positive integers")
        if self.input_length_weights is not None:
            weights = self.input_length_weights
            if (
                bounds[-1] < self.max_input_tokens
                or len(weights) != len(bounds)
                or any(
                    type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights
                )
                or not math.isclose(sum(weights), 1.0, abs_tol=1e-8)
            ):
                raise ValueError("input_length_weights must sum to 1 and cover the input budget")
        if self.input_length_weights_end is not None:
            weights = self.input_length_weights_end
            if (
                self.input_length_weights is None
                or self.input_length_curriculum_steps <= 0
                or len(weights) != len(bounds)
                or any(not math.isfinite(w) or w < 0 for w in weights)
                or not math.isclose(sum(weights), 1)
            ):
                raise ValueError(
                    "length curriculum requires valid start/end weights and positive steps"
                )
        if not 0 <= self.min_lr_fraction <= 1 or (
            self.lr_decay_steps and self.lr_decay_steps <= self.warmup_steps
        ):
            raise ValueError("invalid learning rate schedule")
        for name in (
            "reader_lora_dropout",
            "weight_decay",
            "ae_weight",
            "lm_weight",
            "learning_rate",
            "gradient_clip",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if min(self.learning_rate, self.gradient_clip) <= 0 or self.ae_weight + self.lm_weight <= 0:
            raise ValueError("learning rate, gradient clip and total task weight must be positive")
        if self.reader_lora_dropout >= 1:
            raise ValueError("reader_lora_dropout must be less than 1")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")
        if not self.reader_lora_target_modules or any(
            not isinstance(name, str) or not name for name in self.reader_lora_target_modules
        ):
            raise ValueError("reader_lora_target_modules must contain module names")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        values = dict(raw)
        for field in fields(cls):
            if field.type.startswith("tuple[") and field.name in values:
                if values[field.name] is None and "| None" in field.type:
                    continue
                if not isinstance(values[field.name], (list, tuple)):
                    raise TypeError(f"{field.name} must be an array")
                values[field.name] = tuple(values[field.name])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("The configuration root must be a JSON object")
    return ExperimentConfig.from_mapping(raw)


def write_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n")
