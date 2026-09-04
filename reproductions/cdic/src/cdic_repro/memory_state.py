from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ThreadState:
    """One version of a persistent compressed dialogue thread."""

    state_id: str
    thread_id: str
    revision: int
    latent: object
    retrieval_key: object
    created_turn: int
    written_turn: int
    last_retrieved_turn: int
    parent_state_id: str | None = None
    provenance: tuple[str, ...] = ()
    graph_connected: bool = False

    def recency_at(self, turn: int) -> int:
        if turn < self.last_retrieved_turn:
            raise ValueError("turn cannot precede last_retrieved_turn")
        return turn - self.last_retrieved_turn


class MemoryBank:
    """Ordered memory bank with stable thread identity and versioned states."""

    def __init__(self) -> None:
        self._states: list[ThreadState] = []
        self._next_state_index = 1
        self._next_thread_index = 1

    def __len__(self) -> int:
        return len(self._states)

    @property
    def states(self) -> tuple[ThreadState, ...]:
        return tuple(self._states)

    @property
    def state_ids(self) -> tuple[str, ...]:
        return tuple(state.state_id for state in self._states)

    def get(self, state_id: str) -> ThreadState:
        for state in self._states:
            if state.state_id == state_id:
                return state
        raise KeyError(f"unknown state_id: {state_id}")

    def select(self, state_ids: Iterable[str]) -> tuple[ThreadState, ...]:
        return tuple(self.get(state_id) for state_id in state_ids)

    def insert(
        self,
        *,
        latent: object,
        retrieval_key: object,
        turn: int,
        provenance: tuple[str, ...] = (),
        graph_connected: bool = False,
    ) -> ThreadState:
        self._validate_turn(turn)
        state = ThreadState(
            state_id=self._allocate_state_id(),
            thread_id=self._allocate_thread_id(),
            revision=0,
            latent=latent,
            retrieval_key=retrieval_key,
            created_turn=turn,
            written_turn=turn,
            last_retrieved_turn=turn,
            provenance=provenance,
            graph_connected=graph_connected,
        )
        self._states.append(state)
        return state

    def replace(
        self,
        state_id: str,
        *,
        latent: object,
        retrieval_key: object,
        turn: int,
        provenance: tuple[str, ...] = (),
        graph_connected: bool = False,
    ) -> ThreadState:
        self._validate_turn(turn)
        index = self._index_of(state_id)
        previous = self._states[index]
        if turn < previous.written_turn:
            raise ValueError("replacement turn cannot precede the previous write")
        state = ThreadState(
            state_id=self._allocate_state_id(),
            thread_id=previous.thread_id,
            revision=previous.revision + 1,
            latent=latent,
            retrieval_key=retrieval_key,
            created_turn=previous.created_turn,
            written_turn=turn,
            last_retrieved_turn=turn,
            parent_state_id=previous.state_id,
            provenance=provenance,
            graph_connected=graph_connected,
        )
        self._states[index] = state
        return state

    def mark_retrieved(self, state_ids: Iterable[str], *, turn: int) -> None:
        self._validate_turn(turn)
        requested = set(state_ids)
        unknown = requested.difference(self.state_ids)
        if unknown:
            raise KeyError(f"unknown retrieved state IDs: {sorted(unknown)}")
        self._states = [
            replace(state, last_retrieved_turn=turn) if state.state_id in requested else state
            for state in self._states
        ]

    def _index_of(self, state_id: str) -> int:
        for index, state in enumerate(self._states):
            if state.state_id == state_id:
                return index
        raise KeyError(f"unknown state_id: {state_id}")

    def _allocate_state_id(self) -> str:
        state_id = f"state-{self._next_state_index:06d}"
        self._next_state_index += 1
        return state_id

    def _allocate_thread_id(self) -> str:
        thread_id = f"thread-{self._next_thread_index:06d}"
        self._next_thread_index += 1
        return thread_id

    @staticmethod
    def _validate_turn(turn: int) -> None:
        if turn < 0:
            raise ValueError("turn must be non-negative")
