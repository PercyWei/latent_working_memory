"""离线 teacher 的三段 attention：共享原文 KV，复用压缩历史的局部输出。"""

from itertools import accumulate
import math

import torch
from flashinfer import (
    BatchPrefillWithRaggedKVCacheWrapper,
    merge_state,
    single_prefill_with_kv_cache,
)
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


class SharedPrefixAttention:
    """只在响应生成期间替换 attention；原文 prefill 与运行时回放使用原模块。"""

    def __init__(self, model, prefix, compressed_end, on_read):
        self.model = model
        self.prefix = prefix
        self.compressed_end = compressed_end
        self.on_read = on_read
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.originals = [layer.self_attn for layer in model.model.layers]
        self.workspace = torch.empty(128 * 1024 * 1024, device=self.device, dtype=torch.uint8)
        self.suffix_attention = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace, kv_layout="NHD", backend="fa2",
        )

    def __enter__(self):
        for layer, original in zip(self.model.model.layers, self.originals):
            layer.self_attn = _SharedPrefixLayer(original, self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for layer, original in zip(self.model.model.layers, self.originals):
            layer.self_attn = original

    def plan(self, query_lengths, query_width, suffix_lengths, suffix_width):
        """去掉左填充；每次前向只规划一次，所有层复用相同的 ragged 布局。"""
        def indices(lengths, width):
            rows, columns = [], []
            for row, length in enumerate(lengths):
                rows.extend([row] * length)
                columns.extend(range(width - length, width))
            return tuple(torch.tensor(values, device=self.device, dtype=torch.long)
                         for values in (rows, columns))

        self.query_indices = indices(query_lengths, query_width)
        self.suffix_indices = indices(suffix_lengths, suffix_width)
        # plan 在 CPU 读取长度；直接传 CPU indptr，避免从 GPU 同步取回。
        qo_indptr = torch.tensor([0, *accumulate(query_lengths)], dtype=torch.int32)
        kv_indptr = torch.tensor([0, *accumulate(suffix_lengths)], dtype=torch.int32)
        config = self.model.config
        self.suffix_attention.plan(
            qo_indptr, kv_indptr, config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, causal=True, pos_encoding_mode="NONE",
            sm_scale=self.originals[0].scaling, q_data_type=self.dtype, kv_data_type=self.dtype,
        )

    def read(self, layer, queries, keys, values, scaling):
        # Q/K 已经完成 Qwen3 的 norm 和 RoPE；FlashInfer 不再施加位置编码。
        rows, columns = self.query_indices
        query = queries.transpose(1, 2)[rows, columns]
        prefix_k, prefix_v = self.prefix[layer]
        output, mass = single_prefill_with_kv_cache(
            query, prefix_k[0, :, :self.compressed_end], prefix_v[0, :, :self.compressed_end],
            kv_layout="HND", causal=False, pos_encoding_mode="NONE", sm_scale=scaling,
            return_lse=True, backend="fa2",
        )
        # FlashInfer 0.2.5 返回 log2 LSE；AM 使用自然对数，合并仍使用原 log2 值。
        self.on_read(layer, output, mass * math.log(2))
        recent_output, recent_mass = single_prefill_with_kv_cache(
            query, prefix_k[0, :, self.compressed_end:], prefix_v[0, :, self.compressed_end:],
            kv_layout="HND", causal=False, pos_encoding_mode="NONE", sm_scale=scaling,
            return_lse=True, backend="fa2",
        )
        output, mass = merge_state(output, mass, recent_output, recent_mass)
        kv_rows, kv_columns = self.suffix_indices
        suffix_output, suffix_mass = self.suffix_attention.run(
            query, keys.transpose(1, 2)[kv_rows, kv_columns],
            values.transpose(1, 2)[kv_rows, kv_columns], return_lse=True,
        )
        output, _ = merge_state(output, mass, suffix_output, suffix_mass)
        batch, heads, width, dim = queries.shape
        padded = output.new_zeros(batch, width, heads, dim)
        padded[rows, columns] = output
        return padded


class _SharedPrefixLayer(torch.nn.Module):
    """保留原 Qwen3 投影，只将 suffix KV 写入分支 DynamicCache。"""

    def __init__(self, original, attention):
        super().__init__()
        self.original = original
        self.attention = attention
        self.train(original.training)

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values, cache_position=None, **kwargs):
        module = self.original
        shape = (*hidden_states.shape[:-1], -1, module.head_dim)
        queries = module.q_norm(module.q_proj(hidden_states).view(shape)).transpose(1, 2)
        keys = module.k_norm(module.k_proj(hidden_states).view(shape)).transpose(1, 2)
        values = module.v_proj(hidden_states).view(shape).transpose(1, 2)
        queries, keys = apply_rotary_pos_emb(queries, keys, *position_embeddings)
        keys, values = past_key_values.update(keys, values, module.layer_idx)
        output = self.attention.read(module.layer_idx, queries, keys, values, module.scaling)
        output = output.reshape(*hidden_states.shape[:-1], -1)
        return module.o_proj(output), None
