from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cdic_repro.memory_state import MemoryBank, ThreadState
from cdic_repro.retrieval import RetrievalResult


class WriteAction(str, Enum):
    INITIALIZE = "initialize"
    INSERT = "insert"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class NewStatePayload:
    latent: object
    retrieval_key: object
    provenance: tuple[str, ...] = ()
    graph_connected: bool = False


@dataclass(frozen=True, slots=True)
class WriteBackResult:
    action: WriteAction
    new_state: ThreadState
    replaced_state_id: str | None
    memory_before: tuple[str, ...]
    memory_after: tuple[str, ...]


def apply_write_back(
    memory: MemoryBank,
    *,
    retrieval: RetrievalResult,
    payload: NewStatePayload,
    turn: int,
) -> WriteBackResult:
    """Apply paper Eq. 6 while retaining deterministic state lineage."""

    memory_before = memory.state_ids
    if retrieval.memory_state_ids != memory_before:
        raise ValueError("retrieval result is stale for the current memory bank")
    if retrieval.turn != turn:
        raise ValueError("retrieval turn does not match write-back turn")
    if retrieval.on_topic and retrieval.best_state_id is None:
        raise ValueError("on-topic retrieval requires a best state")

    memory.mark_retrieved(retrieval.selected_state_ids, turn=turn)
    if not memory_before:
        action = WriteAction.INITIALIZE
        replaced_state_id = None
        new_state = memory.insert(
            latent=payload.latent,
            retrieval_key=payload.retrieval_key,
            turn=turn,
            provenance=payload.provenance,
            graph_connected=payload.graph_connected,
        )
    elif not retrieval.on_topic:
        action = WriteAction.INSERT
        replaced_state_id = None
        new_state = memory.insert(
            latent=payload.latent,
            retrieval_key=payload.retrieval_key,
            turn=turn,
            provenance=payload.provenance,
            graph_connected=payload.graph_connected,
        )
    else:
        action = WriteAction.REPLACE
        replaced_state_id = retrieval.best_state_id
        new_state = memory.replace(
            replaced_state_id,
            latent=payload.latent,
            retrieval_key=payload.retrieval_key,
            turn=turn,
            provenance=payload.provenance,
            graph_connected=payload.graph_connected,
        )

    return WriteBackResult(
        action=action,
        new_state=new_state,
        replaced_state_id=replaced_state_id,
        memory_before=memory_before,
        memory_after=memory.state_ids,
    )
