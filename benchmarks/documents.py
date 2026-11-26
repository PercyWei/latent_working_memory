"""把同一正文分配给同一 worker，按批准备正文并独立解码各题。"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class QuestionInput:
    id: str
    prompt: str
    max_new_tokens: int


@dataclass
class PreparedQuestion:
    sample: Any
    inputs: Any


@dataclass
class PreparedConversation:
    state: Any
    questions: list[PreparedQuestion]
    statistics: dict = field(default_factory=dict)


class DocumentAdapter(Protocol):
    def prepare_contexts(self, documents) -> list[PreparedConversation]:
        """按输入顺序为每个 (正文, 问题列表) 准备一份只读状态。"""
        ...

    def answer_batch(self, states, questions, complete):
        """按行从对应的状态分支，并行解码，完成一题即回调。"""
        ...


def group_by_document(items, key):
    groups = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return list(groups.values())


def shard_groups(groups, workers):
    """整组分配，在 worker 之间按待处理条数平衡。"""
    if not groups:
        return []
    if workers < 1:
        raise ValueError("workers must be positive")
    shards = [[] for _ in range(min(workers, len(groups)))]
    for group in sorted(groups, key=len, reverse=True):
        min(shards, key=len).extend(group)
    return shards


def _prepare(adapter, documents):
    prepared = adapter.prepare_contexts(documents)
    for (_, samples), document in zip(documents, prepared, strict=True):
        for sample, question in zip(samples, document.questions, strict=True):
            if question.sample.id != sample.id:
                raise ValueError(f"adapter returned questions out of order for {sample.id}")
            yield document.state, question, document.statistics


def _answer(adapter, batch, commit):
    states, questions, stats = zip(*batch)
    pending = {q.sample.id: s for q, s in zip(questions, stats)}

    def completed(question, answer):
        if question.sample.id not in pending:
            raise ValueError(f"duplicate or unexpected answer: {question.sample.id}")
        commit(question.sample, answer, pending.pop(question.sample.id))

    adapter.answer_batch(states, questions, completed)
    if pending:
        raise ValueError(f"adapter did not answer: {sorted(pending)}")


def evaluate_documents(documents, adapter: DocumentAdapter, batch_size, commit, before_batch=None):
    """用上篇正文剩余的问题优先填批，只为本批首次遇到的正文准备状态。"""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    documents = iter(documents)
    ready = deque()
    while True:
        available = len(ready)
        new_documents = []
        while available < batch_size:
            document = next(documents, None)
            if document is None:
                break
            new_documents.append(document)
            available += len(document[1])
        if not available:
            return
        if before_batch is not None:
            before_batch()
        if new_documents:
            ready.extend(_prepare(adapter, new_documents))
        _answer(adapter, [ready.popleft() for _ in range(min(batch_size, len(ready)))], commit)
