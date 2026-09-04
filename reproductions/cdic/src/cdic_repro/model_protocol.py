from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import Protocol

from cdic_repro.credit import CreditPlan
from cdic_repro.memory_state import ThreadState


@dataclass(frozen=True, slots=True)
class CompressedTurn:
    latent: object
    retrieval_key: object
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingLoss:
    """One teacher-forced response loss and its effective token denominator."""

    value: object
    token_count: int

    def __post_init__(self) -> None:
        if self.token_count < 1:
            raise ValueError("token_count must be positive")


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


class CdicTrainingAdapter(Protocol):
    """Differentiable boundary used by retrieval-aware C-DIC training."""

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
    ) -> CompressedTurn: ...

    def backward(self, loss: object, *, scale: float) -> None: ...

    def loss_requires_grad(self, loss: object) -> bool: ...

    def loss_to_float(self, loss: object) -> float: ...

    def trainable_parameters(self) -> Iterable[object]: ...

    def trainable_state_dict(self) -> Mapping[str, object]: ...

    def load_trainable_state_dict(
        self,
        state_dict: Mapping[str, object],
        *,
        strict: bool = True,
    ) -> None: ...
