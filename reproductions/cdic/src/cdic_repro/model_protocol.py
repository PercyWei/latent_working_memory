from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cdic_repro.memory_state import ThreadState


@dataclass(frozen=True, slots=True)
class CompressedTurn:
    latent: object
    retrieval_key: object
    provenance: tuple[str, ...] = ()


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
