from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Protocol

from torch import Tensor

from cdic_repro.credit import CreditPlan
from cdic_repro.memory_state import ThreadState


@dataclass(frozen=True, slots=True)
class CompressedTurn:
    latent: object
    retrieval_key: object
    provenance: tuple[str, ...] = ()
    gradient_depth: int = 1

    def __post_init__(self) -> None:
        if self.gradient_depth < 1:
            raise ValueError("gradient_depth must be positive")


@dataclass(frozen=True, slots=True)
class TrainingLoss:
    """One teacher-forced response loss and its effective token denominator."""

    value: Tensor
    token_count: int
    token_nll: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.token_count < 1:
            raise ValueError("token_count must be positive")
        if self.token_nll is not None and len(self.token_nll) != self.token_count:
            raise ValueError("token_nll must match the response token denominator")


class CdicModelAdapter(Protocol):
    """Boundary between the paper state machine and a concrete ICAE runtime."""

    def encode_query(self, query: str) -> object: ...

    def generate(self, supports: tuple[ThreadState, ...], query: str) -> str: ...

    def compress(
        self,
        supports: tuple[ThreadState, ...],
        query: str,
        response: str,
    ) -> CompressedTurn: ...


class TrainableStateAdapter(Protocol):
    """Boundary for saving and restoring the trainable model subset."""

    def trainable_state_dict(self) -> Mapping[str, object]: ...

    def load_trainable_state_dict(
        self,
        state_dict: Mapping[str, object],
        *,
        strict: bool = True,
    ) -> None: ...


class CdicTrainingAdapter(Protocol):
    """Differentiable boundary used by retrieval-aware C-DIC training."""

    def encode_query(self, query: str) -> object: ...

    @property
    def gradient_window_size(self) -> int: ...

    def response_loss(
        self,
        supports: tuple[ThreadState, ...],
        query: str,
        response: str,
        credit: CreditPlan,
    ) -> TrainingLoss: ...

    def compress_gold(
        self,
        supports: tuple[ThreadState, ...],
        query: str,
        response: str,
        credit: CreditPlan,
    ) -> CompressedTurn: ...

class CdicEvaluationAdapter(Protocol):
    """Teacher-forced inference boundary used by held-out evaluation."""

    def encode_query(self, query: str) -> object: ...

    def response_loss(
        self,
        supports: tuple[ThreadState, ...],
        query: str,
        response: str,
        credit: CreditPlan,
    ) -> TrainingLoss: ...

    def compress_gold(
        self,
        supports: tuple[ThreadState, ...],
        query: str,
        response: str,
        credit: CreditPlan,
    ) -> CompressedTurn: ...
