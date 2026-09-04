from __future__ import annotations

from dataclasses import dataclass

from cdic_repro.retrieval import RetrievalResult


@dataclass(frozen=True, slots=True)
class CreditPlan:
    """One-hop gradient connectivity prescribed by paper Eq. 8."""

    connected_state_id: str | None
    detached_state_ids: tuple[str, ...]

    def is_connected(self, state_id: str) -> bool:
        return state_id == self.connected_state_id


def build_credit_plan(retrieval: RetrievalResult) -> CreditPlan:
    connected_state_id = retrieval.best_state_id if retrieval.on_topic else None
    detached_state_ids = tuple(
        state_id for state_id in retrieval.selected_state_ids if state_id != connected_state_id
    )
    return CreditPlan(
        connected_state_id=connected_state_id,
        detached_state_ids=detached_state_ids,
    )
