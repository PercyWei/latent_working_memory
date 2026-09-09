from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ThreadState:
    """Memory thread state, 多个 state 共同组成一个紧凑的记忆库。

    注意:
    - 一个语义 thread 可以随着对话推进被多次修订.
    - ThreadState 是不可变对象, MemoryBank 需要更新字段时会创建新的对象, 而不会原地修改.

    状态变化规则:
    - 插入新 thread:
        分配新的 state_id 和 thread_id
        revision=0
        parent_state_id=None
        created_turn, written_turn 和 last_retrieved_turn 均设为当前轮次
    - 仅检索:
        last_retrieved_turn 更新为当前轮次
        其余字段均保持不变
    - 检索并更新:
        分配新的 state_id
        沿用原有的 thread_id 和 created_turn
        revision += 1
        written_turn 和 last_retrieved_turn 更新为当前轮次
        parent_state_id=被替换的旧 state_id
        latent, retrieval_key, provenance 及计算图信息替换为本次压缩产生的新值

    属性：
        state_id: 唯一标识.
        thread_id: 所属 thread 的唯一标识.
        revision: 该 state 在所属 thread 中的修订序号, 表示该 thread 的更新次数.
        latent: 记忆压缩向量, 用于存储某一主题的历史信息, 形状为 [memory_size, hidden_size].
        retrieval_key: 检索向量, 用于在记忆库中进行检索, 形状为 [hidden_size].
        created_turn: 所属 thread 首次插入 MemoryBank 的轮次, 表示整个 thread 的创建时间.
        written_turn: 所属 thread 最近一次更新的轮次.
        last_retrieved_turn: 该 state 最近一次被检索到的轮次.
        parent_state_id: 该 state 的父 state ID, 用于更新时追踪历史 state 版本.
        provenance: 描述压缩内容来源的审计信息, 例如来自生成回答或 gold response.
        graph_connected: 标记该 latent 是否按训练路径保留计算图连接.
        gradient_depth: 当前 latent 计算图中连续 compression 的深度, 用于执行 bounded TBPTT.
            达到配置上限后，前驱 latent 会被 detach, 新图段从深度 1 重新开始.
    """

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
    gradient_depth: int = 1

    def __post_init__(self) -> None:
        if self.gradient_depth < 1:
            raise ValueError("gradient_depth must be positive")

    def recency_at(self, turn: int) -> int:
        """返回指定轮次距离该 state 最近一次被检索所经过的轮次数."""
        if turn < self.last_retrieved_turn:
            raise ValueError("turn cannot precede last_retrieved_turn")
        return turn - self.last_retrieved_turn


class MemoryBank:

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
        latent: object,
        retrieval_key: object,
        turn: int,
        provenance: tuple[str, ...] = (),
        graph_connected: bool = False,
        gradient_depth: int = 1,
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
            gradient_depth=gradient_depth,
        )
        self._states.append(state)
        return state

    def replace(
        self,
        state_id: str,
        latent: object,
        retrieval_key: object,
        turn: int,
        provenance: tuple[str, ...] = (),
        graph_connected: bool = False,
        gradient_depth: int = 1,
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
            gradient_depth=gradient_depth,
        )
        self._states[index] = state
        return state

    def mark_retrieved(self, state_ids: Iterable[str], turn: int) -> None:
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
