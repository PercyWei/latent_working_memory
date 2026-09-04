from __future__ import annotations

from cdic_repro.config import RetrievalConfig
from cdic_repro.memory_state import MemoryBank
from cdic_repro.retrieval import retrieve
from cdic_repro.writeback import NewStatePayload, WriteAction, apply_write_back


def keyed_similarity(query: object, key: object) -> float:
    return 1.0 if query == key else 0.0


def test_empty_memory_initializes_first_thread() -> None:
    memory = MemoryBank()
    retrieval = retrieve(
        memory,
        query_key="topic-a",
        turn=1,
        similarity=keyed_similarity,
        config=RetrievalConfig(),
    )

    result = apply_write_back(
        memory,
        retrieval=retrieval,
        payload=NewStatePayload(latent="latent-a", retrieval_key="topic-a"),
        turn=1,
    )

    assert result.action is WriteAction.INITIALIZE
    assert result.new_state.thread_id == "thread-000001"
    assert result.new_state.revision == 0
    assert result.memory_before == ()
    assert result.memory_after == (result.new_state.state_id,)


def test_on_topic_write_replaces_only_argmax_and_preserves_thread_identity() -> None:
    memory = MemoryBank()
    original = memory.insert(latent="old", retrieval_key="topic-a", turn=1)
    other = memory.insert(latent="other", retrieval_key="topic-a", turn=1)
    retrieval = retrieve(
        memory,
        query_key="topic-a",
        turn=2,
        similarity=keyed_similarity,
        config=RetrievalConfig(decay=0.0),
    )

    result = apply_write_back(
        memory,
        retrieval=retrieval,
        payload=NewStatePayload(latent="new", retrieval_key="topic-a"),
        turn=2,
    )

    assert result.action is WriteAction.REPLACE
    assert result.replaced_state_id == original.state_id
    assert result.new_state.state_id != original.state_id
    assert result.new_state.thread_id == original.thread_id
    assert result.new_state.revision == 1
    assert result.new_state.parent_state_id == original.state_id
    assert memory.state_ids == (result.new_state.state_id, other.state_id)
    assert memory.get(other.state_id).last_retrieved_turn == 2


def test_off_topic_fallback_is_marked_retrieved_before_new_thread_insert() -> None:
    memory = MemoryBank()
    existing = memory.insert(latent="old", retrieval_key="topic-a", turn=1)
    retrieval = retrieve(
        memory,
        query_key="topic-b",
        turn=3,
        similarity=keyed_similarity,
        config=RetrievalConfig(),
    )

    result = apply_write_back(
        memory,
        retrieval=retrieval,
        payload=NewStatePayload(latent="new", retrieval_key="topic-b"),
        turn=3,
    )

    assert result.action is WriteAction.INSERT
    assert result.new_state.thread_id != existing.thread_id
    assert memory.get(existing.state_id).last_retrieved_turn == 3
    assert len(memory) == 2


def test_stale_retrieval_cannot_mutate_changed_memory() -> None:
    memory = MemoryBank()
    memory.insert(latent="a", retrieval_key="topic-a", turn=1)
    retrieval = retrieve(
        memory,
        query_key="topic-a",
        turn=2,
        similarity=keyed_similarity,
        config=RetrievalConfig(),
    )
    memory.insert(latent="b", retrieval_key="topic-b", turn=2)

    try:
        apply_write_back(
            memory,
            retrieval=retrieval,
            payload=NewStatePayload(latent="new", retrieval_key="topic-a"),
            turn=2,
        )
    except ValueError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("stale retrieval should be rejected")
