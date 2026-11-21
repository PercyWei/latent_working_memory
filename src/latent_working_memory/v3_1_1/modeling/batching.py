"""异长源前缀的左侧 padding、独立 KV 打包与有效位置拆分。"""

import torch
from torch.nn import functional as F
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer


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


def _layer_from_tensors(keys, values):
    """接管已分配的独立 KV 张量，初始化层状态而不再次拼接或复制。"""
    layer = DynamicLayer()
    layer.lazy_initialization(keys)
    layer.keys, layer.values = keys, values
    return layer


def pack_caches(caches):
    """直接分配独立 packed KV，不修改输入缓存或已有 key 的 RoPE 编码。"""
    lengths = [cache.get_seq_length() for cache in caches]
    width = max(lengths)
    packed = DynamicCache()
    for layer_index in range(len(caches[0].layers)):
        source = caches[0].layers[layer_index]
        keys = source.keys.new_zeros((
            len(caches), source.keys.shape[1], width, source.keys.shape[-1],
        ))
        values = source.values.new_zeros((
            len(caches), source.values.shape[1], width, source.values.shape[-1],
        ))
        for row, (cache, length) in enumerate(zip(caches, lengths, strict=True)):
            layer = cache.layers[layer_index]
            keys[row:row + 1, :, width - length:].copy_(layer.keys)
            values[row:row + 1, :, width - length:].copy_(layer.values)
        packed.layers.append(_layer_from_tensors(keys, values))
    sizes = torch.tensor(lengths, device=packed.layers[0].keys.device, dtype=torch.long)
    mask = torch.arange(width, device=sizes.device)[None] >= width - sizes[:, None]
    return packed, lengths, width, mask


def unpack_caches(cache, past_lengths, new_lengths, past_width, new_width):
    """消费临时 packed KV，逐层去掉两段左 padding，返回各行独立存储。"""
    lengths = list(zip(past_lengths, new_lengths, strict=True))
    results = [DynamicCache() for _ in lengths]
    for layer in cache.layers:
        for row, (past, new) in enumerate(lengths):
            keys = layer.keys.new_empty((1, layer.keys.shape[1], past + new, layer.keys.shape[-1]))
            values = layer.values.new_empty((1, layer.values.shape[1], past + new, layer.values.shape[-1]))
            keys[:, :, :past].copy_(layer.keys[row:row + 1, :, past_width - past:past_width])
            keys[:, :, past:].copy_(
                layer.keys[row:row + 1, :, past_width + new_width - new:past_width + new_width],
            )
            values[:, :, :past].copy_(layer.values[row:row + 1, :, past_width - past:past_width])
            values[:, :, past:].copy_(
                layer.values[row:row + 1, :, past_width + new_width - new:past_width + new_width],
            )
            results[row].layers.append(_layer_from_tensors(keys, values))
        # 仅释放本函数拥有的临时 packed 层；调用者的旧前缀缓存仍用于边界提交。
        layer.keys = layer.values = None
        layer.is_initialized = False
    return results
