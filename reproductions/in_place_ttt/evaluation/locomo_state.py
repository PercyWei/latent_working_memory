"""In-Place TTT 的 KV、尾块和快速权重按问题独立复制。"""

from copy import copy, deepcopy


def fork_cache(prefix, batch_size):
    cache = copy(prefix)
    cache.rows = [deepcopy(prefix.rows[0]) for _ in range(batch_size)]
    cache.seen = prefix.seen * batch_size
    cache.finished = prefix.finished.new_zeros(batch_size)
    cache.generated_lengths = prefix.generated_lengths.new_zeros(batch_size)
    cache.prompt_width = None
    cache.stop_trackers = []
    return cache


def state_tensors(cache):
    for row in cache.rows:
        for layer in row.layers:
            yield layer.keys
            yield layer.values
        for state in row.ttt_states:
            yield from state
