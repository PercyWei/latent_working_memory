from __future__ import annotations

from dataclasses import dataclass

from cdic_repro.memory_state import ThreadState
from cdic_repro.retrieval import RetrievalResult


@dataclass(frozen=True, slots=True)
class CreditPlan:
    """Current-turn retrieval edge eligible for gradient credit."""

    connected_state_id: str | None
    detached_state_ids: tuple[str, ...]

    def is_connected(self, state_id: str) -> bool:
        return state_id == self.connected_state_id


@dataclass(frozen=True, slots=True)
class CompressionGradientPlan:
    """Gradient connection retained while constructing the next latent state."""

    retained_state_id: str | None
    new_state_gradient_depth: int


def build_credit_plan(retrieval: RetrievalResult) -> CreditPlan:
    connected_state_id = retrieval.best_state_id if retrieval.on_topic else None
    detached_state_ids = tuple(
        state_id for state_id in retrieval.selected_state_ids if state_id != connected_state_id
    )
    return CreditPlan(
        connected_state_id=connected_state_id,
        detached_state_ids=detached_state_ids,
    )


def build_compression_gradient_plan(
    supports: tuple[ThreadState, ...],
    credit: CreditPlan,
    gradient_window_size: int,
) -> CompressionGradientPlan:
    """Bound the revision-chain graph used to create the next latent state.

    The window counts compressor calls represented in a state's graph. Once the
    limit is reached, the predecessor is detached and a new graph segment starts.
    """

    if gradient_window_size < 1:
        raise ValueError("gradient_window_size must be positive")
    if credit.connected_state_id is None:
        return CompressionGradientPlan(
            retained_state_id=None,
            new_state_gradient_depth=1,
        )

    connected_state = next(
        (state for state in supports if state.state_id == credit.connected_state_id),
        None,
    )
    if connected_state is None:
        raise ValueError("connected state must be present in compression supports")
    if connected_state.gradient_depth >= gradient_window_size:
        return CompressionGradientPlan(
            retained_state_id=None,
            new_state_gradient_depth=1,
        )
    return CompressionGradientPlan(
        retained_state_id=connected_state.state_id,
        new_state_gradient_depth=connected_state.gradient_depth + 1,
    )
