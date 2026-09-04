from __future__ import annotations

from cdic_repro.config import RetrievalConfig
from cdic_repro.credit import build_credit_plan
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
