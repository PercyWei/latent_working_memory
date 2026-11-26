"""ICAE Mistral Reader：各问题共享只读正文 KV，单独续写问题和回答。"""

from itertools import groupby

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_with_kvcache


def _read(model, hidden, positions, groups, suffix_caches, suffix_lengths):
    batch, width, _ = hidden.shape
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    cos, sin = model.model.rotary_emb(hidden, positions)
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        inputs = layer.input_layernorm(hidden)
        q = attention.q_proj(inputs).view(batch, width, num_heads, attention.head_dim)
        k = attention.k_proj(inputs).view(batch, width, num_kv_heads, attention.head_dim)
        v = attention.v_proj(inputs).view_as(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # 后缀内部遵循因果掩码；问题中的每个有效位置均可看见整个正文。
        output, suffix_lse = flash_attn_with_kvcache(
            q, *suffix_caches[index], k=k, v=v, cache_seqlens=suffix_lengths,
            causal=True, return_softmax_lse=True,
        )
        for rows, prefix in groups:
            keys, values = prefix[index]
            queries = q[rows]
            # 同正文问题合并查询，直接读取原 KV 的转置视图。
            prefix_output, prefix_lse = flash_attn_with_kvcache(
                queries.reshape(1, -1, num_heads, attention.head_dim),
                keys.transpose(1, 2), values.transpose(1, 2), return_softmax_lse=True,
            )
            # 以两段的 softmax 分母合并，相当于对正文和后缀一起做一次 softmax。
            prefix_lse = prefix_lse.transpose(1, 2).reshape(queries.shape[:-1])
            weight = (prefix_lse - suffix_lse[rows].transpose(1, 2)).sigmoid().unsqueeze(-1)
            output[rows] = torch.lerp(output[rows].float(), prefix_output.reshape_as(queries).float(), weight)

        hidden = hidden + attention.o_proj(output.reshape(batch, width, -1))
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return model.model.norm(hidden)


@torch.inference_mode()
def read_mistral_prefixes(model, caches, prefix_lengths, suffix_embeddings, budgets, eos_id, complete,
                          vocab_size):
    groups = []
    start = 0
    for _, run in groupby(caches, key=id):
        rows = list(run)
        end = start + len(rows)
        groups.append((slice(start, end), rows[0]))
        start = end

    lengths = [len(embeds) for embeds in suffix_embeddings]
    capacity = max(n + budget for n, budget in zip(lengths, budgets, strict=True))
    # 为单 token GQA 内核保留同一 KV head 下各查询头所需的空间。
    capacity = max(capacity, model.config.num_attention_heads // model.config.num_key_value_heads)
    batch = len(caches)
    suffix_caches = []
    for keys, values in caches[0]:
        shape = (batch, capacity, keys.size(1), keys.size(-1))
        suffix_caches.append((keys.new_empty(shape), values.new_empty(shape)))

    inputs = pad_sequence(suffix_embeddings, batch_first=True)
    device = inputs.device
    prefix_positions = torch.tensor(prefix_lengths, device=device)
    question_lengths = torch.tensor(lengths, device=device)
    offsets = torch.arange(inputs.size(1), device=device).clamp(max=question_lengths[:, None] - 1)
    suffix_lengths = torch.zeros(batch, dtype=torch.int32, device=device)
    hidden = _read(model, inputs, prefix_positions[:, None] + offsets, groups, suffix_caches, suffix_lengths)
    last = hidden[torch.arange(batch, device=device), question_lengths - 1]
    # 第一轮右填充的问题从各自的真实末尾续写，直接覆盖填充位置。
    suffix_lengths = question_lengths.int()
    generated = [[] for _ in budgets]
    active = [True] * batch
    while True:
        predictions = model.get_output_embeddings()(last)[:, :vocab_size].argmax(dim=-1)
        for i, token in enumerate(predictions.tolist()):
            if active[i]:
                generated[i].append(token)
                if token == eos_id or len(generated[i]) == budgets[i]:
                    complete(i, {"generated_token_ids": generated[i],
                                 "stop_reason": "eos" if token == eos_id else "max_new_tokens"})
                    active[i] = False
        if not any(active):
            return
        inputs = model.get_input_embeddings()(predictions[:, None])
        positions = (prefix_positions + suffix_lengths)[:, None]
        last = _read(model, inputs, positions, groups, suffix_caches, suffix_lengths)[:, 0]
        suffix_lengths += torch.tensor(active, device=device, dtype=torch.int32)
