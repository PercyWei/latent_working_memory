"""异长源前缀的左侧 padding、独立 KV 打包与有效位置拆分。"""

import torch
from torch.nn import functional as F
from transformers import DynamicCache


def pad_inputs(rows):
    if not rows:
        raise ValueError("a model batch must contain at least one row")
    lengths = [len(row) for row in rows]
    if min(lengths) < 1:
        raise ValueError("batched model inputs must be nonempty")
    width = max(lengths)
    inputs = torch.stack([F.pad(row, (0, 0, width - len(row), 0)) for row in rows])
    sizes = torch.tensor(lengths, device=inputs.device, dtype=torch.long)
    mask = torch.arange(width, device=inputs.device)[None] >= width - sizes[:, None]
    return inputs, lengths, mask


def pack_caches(caches):
    """新建 packed KV，不修改任一输入缓存；旧 key 保持已有 RoPE 编码。"""
    lengths = [cache.get_seq_length() for cache in caches]
    width = max(lengths)
    layers = []
    for layer_index in range(len(caches[0].layers)):
        keys, values = [], []
        for cache, length in zip(caches, lengths, strict=True):
            layer = cache.layers[layer_index]
            padding = (0, 0, width - length, 0)
            keys.append(F.pad(layer.keys, padding))
            values.append(F.pad(layer.values, padding))
        layers.append((torch.cat(keys), torch.cat(values)))
    packed = DynamicCache(ddp_cache_data=layers)
    sizes = torch.tensor(lengths, device=layers[0][0].device, dtype=torch.long)
    mask = torch.arange(width, device=sizes.device)[None] >= width - sizes[:, None]
    return packed, lengths, width, mask


def unpack_caches(cache, past_lengths, new_lengths, past_width, new_width):
    """分别抽取有效 past 与 new，去掉两段各自的左 padding。"""
    results = []
    for row, (past, new) in enumerate(zip(past_lengths, new_lengths, strict=True)):
        layers = []
        for layer in cache.layers:
            keys = torch.cat((
                layer.keys[row:row + 1, :, past_width - past:past_width],
                layer.keys[row:row + 1, :, past_width + new_width - new:past_width + new_width],
            ), dim=-2)
            values = torch.cat((
                layer.values[row:row + 1, :, past_width - past:past_width],
                layer.values[row:row + 1, :, past_width + new_width - new:past_width + new_width],
            ), dim=-2)
            layers.append((keys, values))
        results.append(DynamicCache(ddp_cache_data=layers))
    return results
