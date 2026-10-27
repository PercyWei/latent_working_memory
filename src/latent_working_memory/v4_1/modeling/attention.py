"""无可变 HF cache 的 Qwen3 layer，供流式执行及 checkpoint 重算共用。"""

import math

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


def decoder_forward(layer, hidden, old_keys, old_values, memory_key, memory_value,
                    memory_bias, cosine, sine):
    residual = hidden
    hidden = layer.input_layernorm(hidden)
    attention = layer.self_attn
    batch, length, _ = hidden.shape
    shape = (batch, length, -1, attention.head_dim)
    query = attention.q_norm(attention.q_proj(hidden).view(shape)).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(hidden).view(shape)).transpose(1, 2)
    value = attention.v_proj(hidden).view(shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, cosine, sine)
    raw_keys = torch.cat((old_keys, key), dim=2)
    raw_values = torch.cat((old_values, value), dim=2)
    # 正式读取遵循基座精度；FP32 memory KV 另用于局部 AM。
    keys = torch.cat((memory_key[None].to(key.dtype), raw_keys), dim=2)
    values = torch.cat((memory_value[None].to(value.dtype), raw_values), dim=2)
    groups = query.shape[1] // keys.shape[1]
    keys = keys.repeat_interleave(groups, dim=1)
    values = values.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query.float(), keys.float().transpose(-1, -2))
    scores = scores / math.sqrt(attention.head_dim)
    memory_length, raw_length = memory_key.shape[1], raw_keys.shape[2]
    bias = torch.cat((memory_bias, memory_bias.new_zeros(memory_bias.shape[0], raw_length)), -1)
    scores = scores + bias.repeat_interleave(groups, dim=0)[None, :, None, :]
    # 所有 memory 条目可见；原始 KV 只可见旧前缀和当前 chunk 的因果前缀。
    columns = torch.arange(keys.shape[2], device=hidden.device)
    rows = memory_length + old_keys.shape[2] + torch.arange(length, device=hidden.device)
    scores = scores.masked_fill(columns[None, :] > rows[:, None], float("-inf"))
    weights = scores.softmax(-1).to(query.dtype)
    output = torch.matmul(weights, values).transpose(1, 2).reshape(batch, length, -1)
    hidden = residual + attention.o_proj(output)
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden, raw_keys, raw_values, query
