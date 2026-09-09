from __future__ import annotations

import pytest

from cdic_repro.config import RetrievalConfig
from cdic_repro.credit import CreditPlan, build_compression_gradient_plan, build_credit_plan
from cdic_repro.memory_state import MemoryBank
from cdic_repro.retrieval import retrieve


def test_on_topic_credit_connects_only_argmax() -> None:
    memory = MemoryBank()
    first = memory.insert(latent="a", retrieval_key="a", turn=0)
    second = memory.insert(latent="b", retrieval_key="b", turn=0)
    similarities = {"a": 0.95, "b": 0.9}
    retrieval = retrieve(
        memory,
        query_key="query",
        turn=1,
        similarity=lambda _query, key: similarities[str(key)],
        config=RetrievalConfig(decay=0.0),
    )

    plan = build_credit_plan(retrieval)

    assert plan.connected_state_id == first.state_id
    assert plan.detached_state_ids == (second.state_id,)
    assert plan.is_connected(first.state_id)
    assert not plan.is_connected(second.state_id)


def test_off_topic_fallback_is_detached() -> None:
    memory = MemoryBank()
    state = memory.insert(latent="a", retrieval_key="a", turn=0)
    retrieval = retrieve(
        memory,
        query_key="query",
        turn=1,
        similarity=lambda _query, _key: 0.2,
        config=RetrievalConfig(decay=0.0),
    )

    plan = build_credit_plan(retrieval)

    assert plan.connected_state_id is None
    assert plan.detached_state_ids == (state.state_id,)


def test_compression_gradient_window_resets_at_configured_depth() -> None:
    memory = MemoryBank()
    first = memory.insert(
        latent="a",
        retrieval_key="a",
        turn=0,
        gradient_depth=1,
    )
    credit = CreditPlan(connected_state_id=first.state_id, detached_state_ids=())

    continued = build_compression_gradient_plan((first,), credit, gradient_window_size=3)
    reset = build_compression_gradient_plan((first,), credit, gradient_window_size=1)

    assert continued.retained_state_id == first.state_id
    assert continued.new_state_gradient_depth == 2
    assert reset.retained_state_id is None
    assert reset.new_state_gradient_depth == 1


def test_compression_gradient_window_rejects_missing_connected_state() -> None:
    credit = CreditPlan(connected_state_id="missing", detached_state_ids=())

    with pytest.raises(ValueError, match="present in retrieved states"):
        build_compression_gradient_plan((), credit, gradient_window_size=2)
