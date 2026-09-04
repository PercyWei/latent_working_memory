from __future__ import annotations

import math

import pytest

from cdic_repro.config import RetrievalConfig, SupportOrder
from cdic_repro.memory_state import MemoryBank
from cdic_repro.retrieval import retrieve


def cosine(left: object, right: object) -> float:
    left_vector = tuple(float(value) for value in left)  # type: ignore[union-attr]
    right_vector = tuple(float(value) for value in right)  # type: ignore[union-attr]
    numerator = sum(a * b for a, b in zip(left_vector, right_vector, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    return numerator / (left_norm * right_norm)


def test_empty_memory_retrieves_nothing() -> None:
    result = retrieve(
        MemoryBank(),
        query_key=(1.0, 0.0),
        turn=1,
        similarity=cosine,
        config=RetrievalConfig(),
    )

    assert result.selected_state_ids == ()
    assert result.best_state_id is None
    assert result.peak_score is None
    assert not result.used_fallback


def test_threshold_retrieval_selects_multiple_states_in_score_order() -> None:
    memory = MemoryBank()
    first = memory.insert(latent="a", retrieval_key=(0.9, 0.4358899), turn=0)
    second = memory.insert(latent="b", retrieval_key=(1.0, 0.0), turn=0)
    memory.insert(latent="c", retrieval_key=(0.0, 1.0), turn=0)

    result = retrieve(
        memory,
        query_key=(1.0, 0.0),
        turn=1,
        similarity=cosine,
        config=RetrievalConfig(threshold=0.8, decay=0.0),
    )

    assert result.on_topic
    assert result.selected_state_ids == (second.state_id, first.state_id)
    assert result.best_state_id == second.state_id
    assert not result.used_fallback


def test_off_topic_retrieval_uses_top_one_fallback() -> None:
    memory = MemoryBank()
    first = memory.insert(latent="a", retrieval_key=(0.4, 0.9165151), turn=0)
    second = memory.insert(latent="b", retrieval_key=(0.2, 0.9797959), turn=0)

    result = retrieve(
        memory,
        query_key=(1.0, 0.0),
        turn=1,
        similarity=cosine,
        config=RetrievalConfig(threshold=0.8, decay=0.0),
    )

    assert not result.on_topic
    assert result.used_fallback
    assert result.best_state_id == first.state_id
    assert result.selected_state_ids == (first.state_id,)
    assert second.state_id not in result.selected_state_ids


def test_recency_decay_prefers_recent_equal_content() -> None:
    memory = MemoryBank()
    old = memory.insert(latent="old", retrieval_key=(1.0, 0.0), turn=0)
    recent = memory.insert(latent="recent", retrieval_key=(1.0, 0.0), turn=9)

    result = retrieve(
        memory,
        query_key=(1.0, 0.0),
        turn=10,
        similarity=cosine,
        config=RetrievalConfig(threshold=0.0, decay=0.1),
    )

    assert result.best_state_id == recent.state_id
    scores = {score.state_id: score for score in result.scores}
    assert scores[old.state_id].score == pytest.approx(math.exp(-1.0))
    assert scores[recent.state_id].score == pytest.approx(math.exp(-0.1))


def test_equality_is_on_topic() -> None:
    memory = MemoryBank()
    state = memory.insert(latent="a", retrieval_key="a", turn=0)

    result = retrieve(
        memory,
        query_key="query",
        turn=1,
        similarity=lambda _query, _key: 0.8,
        config=RetrievalConfig(threshold=0.8, decay=0.0),
    )

    assert result.on_topic
    assert result.selected_state_ids == (state.state_id,)


def test_bounded_retrieval_takes_top_scores_before_support_ordering() -> None:
    memory = MemoryBank()
    first = memory.insert(latent="a", retrieval_key="a", turn=0)
    second = memory.insert(latent="b", retrieval_key="b", turn=0)
    memory.insert(latent="c", retrieval_key="c", turn=0)
    similarities = {"a": 0.9, "b": 0.95, "c": 0.85}

    result = retrieve(
        memory,
        query_key="query",
        turn=1,
        similarity=lambda _query, key: similarities[str(key)],
        config=RetrievalConfig(
            threshold=0.8,
            decay=0.0,
            support_order=SupportOrder.MEMORY_ORDER,
            max_retrieved=2,
        ),
    )

    assert result.selected_state_ids == (first.state_id, second.state_id)


def test_similarity_outside_cosine_range_is_rejected() -> None:
    memory = MemoryBank()
    memory.insert(latent="a", retrieval_key="a", turn=0)

    with pytest.raises(ValueError, match="cosine range"):
        retrieve(
            memory,
            query_key="query",
            turn=1,
            similarity=lambda _query, _key: 2.0,
            config=RetrievalConfig(),
        )
