"""方法无关的会话分片、同正文组批与逐题提交。"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from benchmarks.data import BenchmarkSample, locomo_history


@dataclass
class PreparedQuestion:
    sample: Any
    inputs: Any
    batch_key: tuple


@dataclass
class PreparedConversation:
    state: Any
    questions: list[PreparedQuestion]
    statistics: dict = field(default_factory=dict)


class ConversationAdapter(Protocol):
    def prepare_context(
        self, history: str, questions: list[BenchmarkSample],
    ) -> PreparedConversation:
        """准备只读正文状态；batch_key 标识可一起解码的问题。"""
        ...

    def answer_batch(
        self, state: Any, questions: list[PreparedQuestion],
        complete: Callable[[PreparedQuestion, dict], None],
    ) -> None:
        """从正文状态为各题独立解码，完成一题就回调一次。"""
        ...


def conversations(samples):
    groups = {}
    for sample in samples:
        groups.setdefault(sample.metadata["conversation_id"], []).append(sample)
    return list(groups.values())


def shard_conversations(samples, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    groups = conversations(samples)
    shards = [[] for _ in range(min(workers, len(groups)))]
    for group in sorted(groups, key=len, reverse=True):
        min(shards, key=len).extend(group)
    return shards


def evaluate_conversations(samples, adapter: ConversationAdapter, batch_size, commit):
    """adapter 返回的正文状态在各问题间只读；独立的回答分支由 adapter 创建。"""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    for questions in conversations(samples):
        prepared = adapter.prepare_context(locomo_history(questions[0]), questions)
        groups = defaultdict(list)
        for question in prepared.questions:
            groups[question.batch_key].append(question)
        pending = {sample.id for sample in questions}
        statistics = prepared.statistics

        def completed(question, answer):
            if question.sample.id not in pending:
                raise ValueError(f"duplicate or unexpected answer: {question.sample.id}")
            commit(question.sample, answer, statistics)
            pending.remove(question.sample.id)

        for group in groups.values():
            for start in range(0, len(group), batch_size):
                adapter.answer_batch(prepared.state, group[start:start + batch_size], completed)
        if pending:
            raise ValueError(f"adapter did not answer: {sorted(pending)}")
        del prepared
