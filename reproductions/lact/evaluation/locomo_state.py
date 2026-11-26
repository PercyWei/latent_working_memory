"""按来源行分支 LaCT 正文的局部 KV、快速权重和未满块。"""

from copy import copy

import torch


def fork_cache(sources):
    prefix, _ = sources[0]
    branch = copy(prefix)
    branch.batch_size = len(sources)
    branch.physical_seen = max(cache.layers[0].seen[row] for cache, row in sources)
    branch._next_positions = torch.stack([cache._next_positions[row] for cache, row in sources])
    branch._layout = branch._generated_mask = branch._generated_mask_version = None
    branch.layers = []
    for index, original in enumerate(prefix.layers):
        layers = [(cache.layers[index], row) for cache, row in sources]
        layer = copy(original)
        # attention_forward_padded 为新 token 分配 KV，并替换列表元素；原 KV 可共享。
        layer.keys = [source.keys[row] for source, row in layers]
        layer.values = [source.values[row] for source, row in layers]
        layer.key_lengths = layer.key_offsets = None
        ttt = copy(original.ttt)
        ttt.batch_size = len(sources)
        ttt._rows = list(range(len(sources)))
        states = [(source.ttt, row) for source, row in layers]
        for name in (*ttt._components, "w0_norm", "w1_norm", "w2_norm"):
            if getattr(original.ttt, name) is not None:
                setattr(ttt, name, torch.cat([getattr(state, name).narrow(0, row * ttt.heads, ttt.heads)
                                             for state, row in states]))
        for name in ttt._pending:
            if getattr(original.ttt, name) is not None:
                setattr(ttt, name, torch.stack([getattr(state, name)[row] for state, row in states]))
        for name in ("seen", "n_updates", "pending_length"):
            setattr(ttt, name, [getattr(state, name)[row] for state, row in states])
        ttt._dtypes = {name: [state._dtypes[name][row] for state, row in states]
                       for name in original.ttt._dtypes}
        layer.ttt, layer.seen = ttt, ttt.seen
        branch.layers.append(layer)
    return branch


def state_tensors(cache):
    for layer in cache.layers:
        yield from layer.keys
        yield from layer.values
        ttt = layer.ttt
        for name in (*ttt._components, *ttt._pending, "w0_norm", "w1_norm", "w2_norm"):
            yield getattr(ttt, name)
