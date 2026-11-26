"""正文独立预填、只读前缀分支与混长问题的贪心解码。"""

from copy import copy
from inspect import signature

import torch


def _slice_cache(cache, rows, clone=False):
    """从相同长度的预填结果中取出指定行；保留各层自身的滑窗状态。"""
    def select(tensor):
        selected = tensor[rows]
        return selected.clone() if clone else selected

    if isinstance(cache, tuple):
        return tuple(tuple(select(t) for t in layer) for layer in cache)
    result = copy(cache)
    result.layers = [copy(layer) for layer in cache.layers]
    for layer in result.layers:
        layer.keys = select(layer.keys)
        layer.values = select(layer.values)
    return result


@torch.no_grad()
def prefill_prefixes(model, prefixes, chunk_size=None):
    """并行预填独立的 token ID 或嵌入序列。已结束正文退出批次，不给滑窗 KV 写入 padding。"""
    lengths = [len(row) for row in prefixes]
    active = sorted(range(len(prefixes)), key=lengths.__getitem__, reverse=True)
    result = [None] * len(prefixes)
    cache = None
    start = 0
    while active:
        end = lengths[active[-1]]
        if chunk_size is not None:
            end = min(end, start + chunk_size)
        inputs = torch.stack([prefixes[i][start:end] for i in active])
        field = "input_ids" if inputs.ndim == 2 else "inputs_embeds"
        cache = model.base_model(**{field: inputs}, past_key_values=cache,
                                 use_cache=True, return_dict=True).past_key_values
        # 最短正文先完成；仍运行的行连续排列，可直接切片而无需逐轮拷贝其长 KV。
        keep = len(active)
        while keep and lengths[active[keep - 1]] == end:
            row = keep - 1
            result[active[row]] = cache if len(active) == 1 else _slice_cache(cache, slice(row, row + 1), clone=True)
            keep -= 1
        if keep < len(active):
            active = active[:keep]
            if keep:
                cache = _slice_cache(cache, slice(keep))
        start = end
    return result


def _join_prefixes(caches, lengths):
    """按行左填充 KV。浅复制可变的 DynamicCache 容器，避免覆盖同正文后续问题的前缀。"""
    first = caches[0]
    same = all(cache is first for cache in caches)

    def join(tensors):
        if same:
            return tensors[0].expand(len(tensors), -1, -1, -1)
        width = max(t.size(-2) for t in tensors)
        if all(t.size(-2) == width for t in tensors):
            return torch.cat(tensors, dim=0)
        joined = tensors[0].new_zeros((len(tensors), tensors[0].size(1), width, tensors[0].size(-1)))
        for row, tensor in enumerate(tensors):
            joined[row, :, -tensor.size(-2):, :].copy_(tensor[0])
        return joined

    if isinstance(first, tuple):
        return tuple(tuple(join(tensors) for tensors in zip(*layers, strict=True))
                     for layers in zip(*caches, strict=True))
    result = copy(first)
    result.layers = []
    for layers in zip(*(cache.layers for cache in caches), strict=True):
        layer = copy(layers[0])
        layer.keys = join([source.keys for source in layers])
        layer.values = join([source.values for source in layers])
        if layer.is_sliding:
            layer.cumulative_length = max(lengths)
        result.layers.append(layer)
    return result


@torch.no_grad()
def read_prefixes(model, caches, prefix_lengths, suffix_embeddings, budgets, eos_id, complete,
                  vocab_size=None, native_logits=False):
    """续写独立的缓存分支；滑窗问题的有效 token 必须在物理 KV 位置上也连续。"""
    cache = _join_prefixes(caches, prefix_lengths)
    suffix_lengths = [len(row) for row in suffix_embeddings]
    longest = max(suffix_lengths)
    sliding = (getattr(model.config, "sliding_window", None) is not None or
               (not isinstance(cache, tuple) and any(cache.is_sliding)))
    device = suffix_embeddings[0].device
    if sliding and min(suffix_lengths) < longest:
        # 中间插入 padding 会挤出本该可见的窗口。先一起读到最短问题的末端，
        # 随后短行生成答案、长行继续读问题，每一步都使用连续的真实位置。
        width = min(suffix_lengths)
        embeddings = torch.stack([row[:width] for row in suffix_embeddings])
        query_mask = torch.ones((len(caches), width), device=device, dtype=torch.bool)
        pending_ends = suffix_lengths
    else:
        width = longest
        embeddings = torch.stack([torch.nn.functional.pad(row, (0, 0, width - len(row), 0))
                                  for row in suffix_embeddings])
        query_mask = torch.arange(width, device=device) >= width - torch.tensor(suffix_lengths, device=device)[:, None]
        pending_ends = [width] * len(caches)
    lengths = torch.tensor(prefix_lengths, device=device)
    prefix_width = max(prefix_lengths)
    prefix_mask = torch.arange(prefix_width, device=device) >= prefix_width - lengths[:, None]
    mask = torch.cat((prefix_mask, query_mask), dim=1)
    pos = (lengths[:, None] + query_mask.long().cumsum(1) - 1).masked_fill_(~query_mask, 0)
    generated = [[] for _ in budgets]
    active = set(range(len(budgets)))
    consumed = width
    # 原始基座保留原生 LM forward 内的 logit 变换。部分架构提供只投影末位的接口。
    decoder = model if native_logits else model.base_model
    forward_options = {}
    if native_logits and "logits_to_keep" in signature(model.forward).parameters:
        forward_options["logits_to_keep"] = 1
    while active:
        outputs = decoder(inputs_embeds=embeddings, past_key_values=cache, attention_mask=mask,
                          position_ids=pos, use_cache=True, return_dict=True, **forward_options)
        cache = outputs.past_key_values
        logits = (outputs.logits[:, -1] if native_logits else
                  model.get_output_embeddings()(outputs.last_hidden_state[:, -1]))
        if vocab_size is not None:
            logits = logits[:, :vocab_size]
        predicted = logits.argmax(dim=-1)
        for i, token in enumerate(predicted.tolist()):
            if i not in active or consumed < pending_ends[i]:
                continue
            generated[i].append(token)
            reason = ("eos" if token == eos_id else
                      "max_new_tokens" if len(generated[i]) == budgets[i] else None)
            if reason:
                complete(i, {"generated_token_ids": generated[i], "stop_reason": reason})
                active.remove(i)
        if active:
            embeddings = model.get_input_embeddings()(predicted[:, None])
            for i in active:
                if consumed < pending_ends[i]:
                    embeddings[i, 0] = suffix_embeddings[i][consumed]
            mask = torch.cat((mask, mask.new_ones((len(caches), 1))), dim=1)
            pos = pos[:, -1:] + torch.tensor([i in active for i in range(len(caches))], device=device)[:, None]
            consumed += 1
