"""Configuration for dynamic-memory experiments."""

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path


@dataclass(frozen=True)
class DynamicConfig:
    capacities: tuple[int, ...] = (64, 128, 256, 512, 1024)
    ratios: tuple[int, ...] = (2, 4, 8)
    micro_epochs_per_capacity: int = 1
    samples_per_micro_epoch: int = 100
    stage_ends: tuple[float, ...] = (0.3, 0.7, 1.0)
    ratio_weights: tuple[tuple[float, ...], ...] = (
        (0.6, 0.3, 0.1),
        (0.3, 0.4, 0.3),
        (0.1, 0.3, 0.6),
    )
    new_count: int = 1
    history_count: int = 1
    max_visits: int = 2
    bptt_unit: str = "tokens"
    bptt_span: int = 0
    global_batch_size: int = 2
    qa_activation_checkpointing: bool = True
    learning_rate: float = 0.00003
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    generation_tokens: int = 64
    seed: int = 42
    epochs: int = 3
    save_every: int = 100
    eval_every: int = 100
    eval_generation_every: int = 250
    eval_texts_per_ratio: int = 2
    eval_reads_per_kind: int = 2

    def __post_init__(self):
        # JSON arrays have one canonical in-memory representation.
        for name in ("capacities", "ratios", "stage_ends"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "ratio_weights", tuple(tuple(w) for w in self.ratio_weights))
        for name in ("capacities", "ratios"):
            values = getattr(self, name)
            if (
                not values
                or len(set(values)) != len(values)
                or any(type(v) is not int or v <= 0 for v in values)
            ):
                raise ValueError(f"{name} must contain unique positive integers")
        for name in (
            "micro_epochs_per_capacity",
            "samples_per_micro_epoch",
            "global_batch_size",
            "max_visits",
            "generation_tokens",
            "epochs",
            "save_every",
            "eval_every",
            "eval_generation_every",
            "eval_texts_per_ratio",
            "eval_reads_per_kind",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("new_count", "history_count", "bptt_span", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.samples_per_micro_epoch < self.global_batch_size:
            raise ValueError("samples_per_micro_epoch must contain a complete batch")
        if self.new_count + self.history_count == 0:
            raise ValueError("empty reading policy")
        if self.bptt_unit not in {"tokens", "updates"}:
            raise ValueError("bptt_unit must be tokens or updates")
        if type(self.qa_activation_checkpointing) is not bool:
            raise ValueError("qa_activation_checkpointing must be boolean")
        if (
            not self.stage_ends
            or self.stage_ends[-1] != 1
            or any(not math.isfinite(v) or v <= 0 for v in self.stage_ends)
            or tuple(sorted(set(self.stage_ends))) != self.stage_ends
            or len(self.ratio_weights) != len(self.stage_ends)
            or any(
                len(w) != len(self.ratios)
                or any(not math.isfinite(v) or v < 0 for v in w)
                or not math.isclose(sum(w), 1.0)
                for w in self.ratio_weights
            )
        ):
            raise ValueError("invalid compression curriculum stages or weights")
        if not (
            0 < self.learning_rate < float("inf")
            and 0 < self.gradient_clip < float("inf")
            and 0 <= self.weight_decay < float("inf")
        ):
            raise ValueError("invalid optimizer parameters")

    @property
    def micro_epochs_per_epoch(self):
        return len(self.capacities) * self.micro_epochs_per_capacity

    @property
    def steps_per_micro_epoch(self):
        return self.samples_per_micro_epoch // self.global_batch_size

    def capacity_order(self, epoch):
        order = list(self.capacities) * self.micro_epochs_per_capacity
        random.Random(f"{self.seed}:capacity:{epoch}").shuffle(order)
        return order

    def weights(self, epoch, epochs):
        return next(
            weights
            for end, weights in zip(self.stage_ends, self.ratio_weights, strict=True)
            if epoch / max(epochs - 1, 1) <= end
        )


def load_dynamic_config(path: Path) -> DynamicConfig:
    return DynamicConfig(**json.loads(path.read_text()))
