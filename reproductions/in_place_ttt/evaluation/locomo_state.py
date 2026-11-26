"""按来源行复制 In-Place TTT 正文的 KV、尾块和快速权重。"""

from copy import copy


def fork_cache(sources):
    prefix, _ = sources[0]
    branch = copy(prefix)
    branch.rows = []
    for cache, row in sources:
        original = cache.rows[row]
        # KV 追加和快速权重更新都返回新张量，仅列表和 layer 对象会就地更新。
        selected = copy(original)
        selected.layers = [copy(layer) for layer in original.layers]
        selected.ttt_states = list(original.ttt_states)
        branch.rows.append(selected)
    branch.seen = [cache.seen[row] for cache, row in sources]
    branch.physical_seen = max(branch.seen)
    branch.finished = prefix.finished.new_zeros(len(sources))
    branch.generated_lengths = prefix.generated_lengths.new_zeros(len(sources))
    branch.prompt_width = None
    branch.stop_trackers = []
    return branch


def state_tensors(cache):
    for row in cache.rows:
        for layer in row.layers:
            yield layer.keys
            yield layer.values
        for state in row.ttt_states:
            yield from state
