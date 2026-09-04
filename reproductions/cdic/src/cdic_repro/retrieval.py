from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from cdic_repro.config import RetrievalConfig, SupportOrder
from cdic_repro.memory_state import MemoryBank


SimilarityFunction = Callable[[object, object], float]


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    state_id: str
    thread_id: str
    memory_index: int
    raw_similarity: float
    recency_turns: int
    decay_weight: float
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    turn: int
    memory_state_ids: tuple[str, ...]
    scores: tuple[ScoreRecord, ...]
    selected_state_ids: tuple[str, ...]
    best_state_id: str | None
    peak_score: float | None
    on_topic: bool
    used_fallback: bool


def retrieve(
    memory: MemoryBank,
    *,
    query_key: object,
    turn: int,
    similarity: SimilarityFunction,
    config: RetrievalConfig,
) -> RetrievalResult:
    """Apply paper Eq. 3 and Algorithm 1 without differentiating selection."""

    if turn < 0:
        raise ValueError("turn must be non-negative")
    if len(memory) == 0:
        return RetrievalResult(
            turn=turn,
            memory_state_ids=(),
            scores=(),
            selected_state_ids=(),
            best_state_id=None,
            peak_score=None,
            on_topic=False,
            used_fallback=False,
        )

    records: list[ScoreRecord] = []
    for memory_index, state in enumerate(memory.states):
        raw_similarity = float(similarity(query_key, state.retrieval_key))
        if not math.isfinite(raw_similarity):
            raise ValueError(f"non-finite similarity for {state.state_id}")
        if not -1.000001 <= raw_similarity <= 1.000001:
            raise ValueError(f"similarity outside cosine range for {state.state_id}")
        recency_turns = state.recency_at(turn)
        decay_weight = math.exp(-config.decay * recency_turns)
        records.append(
            ScoreRecord(
                state_id=state.state_id,
                thread_id=state.thread_id,
                memory_index=memory_index,
                raw_similarity=raw_similarity,
                recency_turns=recency_turns,
                decay_weight=decay_weight,
                score=raw_similarity * decay_weight,
            )
        )

    best = max(records, key=lambda record: record.score)
    on_topic = best.score >= config.threshold
    if on_topic:
        selected = [record for record in records if record.score >= config.threshold]
    else:
        selected = [best]

    if config.max_retrieved is not None:
        selected.sort(key=lambda record: (-record.score, record.memory_index))
        selected = selected[: config.max_retrieved]
    if config.support_order is SupportOrder.SCORE_DESC:
        selected.sort(key=lambda record: (-record.score, record.memory_index))
    else:
        selected.sort(key=lambda record: record.memory_index)

    return RetrievalResult(
        turn=turn,
        memory_state_ids=memory.state_ids,
        scores=tuple(records),
        selected_state_ids=tuple(record.state_id for record in selected),
        best_state_id=best.state_id,
        peak_score=best.score,
        on_topic=on_topic,
        used_fallback=not on_topic,
    )
