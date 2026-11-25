"""LaCT 共享正文状态的独立批量分支。"""

from copy import copy


def fork_cache(prefix, batch_size):
    cache = copy(prefix)
    cache.batch_size = batch_size
    cache._next_positions = prefix._next_positions.repeat(batch_size)
    cache._layout = cache._generated_mask = cache._generated_mask_version = None
    cache.layers = []
    for original in prefix.layers:
        layer = copy(original)
        layer.keys = [original.keys[0].clone() for _ in range(batch_size)]
        layer.values = [original.values[0].clone() for _ in range(batch_size)]
        layer.key_lengths = layer.key_offsets = None
        ttt = copy(original.ttt)
        ttt.batch_size = batch_size
        ttt._rows = list(range(batch_size))
        for name in (*ttt._components, "w0_norm", "w1_norm", "w2_norm"):
            value = getattr(original.ttt, name)
            setattr(ttt, name, None if value is None else value.repeat(batch_size, 1, 1))
        for name in ttt._pending:
            value = getattr(original.ttt, name)
            setattr(ttt, name, None if value is None else value.repeat(batch_size, 1, 1, 1))
        for name in ("seen", "n_updates", "pending_length"):
            setattr(ttt, name, getattr(original.ttt, name) * batch_size)
        ttt._dtypes = {name: values * batch_size for name, values in original.ttt._dtypes.items()}
        layer.ttt, layer.seen = ttt, ttt.seen
        cache.layers.append(layer)
    return cache


def state_tensors(cache):
    for layer in cache.layers:
        yield from layer.keys
        yield from layer.values
        ttt = layer.ttt
        for name in (*ttt._components, *ttt._pending, "w0_norm", "w1_norm", "w2_norm"):
            yield getattr(ttt, name)
