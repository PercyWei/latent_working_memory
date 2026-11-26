"""按完整会话正文分片，并使用公共 QA 批处理调度器。"""

from benchmarks.data import locomo_history
from benchmarks.documents import evaluate_documents, group_by_document, shard_groups


def conversations(samples):
    return group_by_document(samples, locomo_history)


def shard_conversations(samples, workers):
    return shard_groups(conversations(samples), workers)


def evaluate_conversations(samples, adapter, batch_size, commit, before_batch=None):
    documents = ((locomo_history(group[0]), group) for group in conversations(samples))
    evaluate_documents(documents, adapter, batch_size, commit, before_batch)
