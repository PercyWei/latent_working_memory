"""生成时采集紧凑统计；按原提示顺序维护固定候选池，不保存逐位置向量。"""

import math

import torch
from transformers.models.qwen3.modeling_qwen3 import rotate_half


def _local_moments(queries, keys, values):
    """GQA 共享 KV；返回每个 query/head 的输出通道均值、方差和 log-mass。"""
    query_heads, head_dim = queries.shape[1:]
    kv_heads = keys.shape[0]
    grouped = queries.reshape(len(queries), kv_heads, query_heads // kv_heads, head_dim)
    scores = torch.einsum("nkgd,ktd->nkgt", grouped, keys) / math.sqrt(head_dim)
    log_mass = scores.logsumexp(-1).reshape(len(queries), query_heads)
    outputs = torch.einsum("nkgt,ktd->nkgd", scores.softmax(-1), values)
    outputs = outputs.reshape(len(queries), query_heads, head_dim)
    variance, mean = torch.var_mean(outputs.double(), dim=-1, correction=0)
    return torch.stack((mean, variance, log_mass.double()), dim=-1)


class GenerationStatistics:
    """CPU 保留候选池三标量，GPU 只保留当前批统计和至多 16 个 query 的局部读取。"""

    def __init__(self, model, prefix, compressed_end, pool_size, rng):
        self.model = model
        self.prefix = prefix
        self.compressed_end = compressed_end
        self.pool_size = pool_size
        self.rng = rng
        self.device = next(model.parameters()).device
        self.layers = model.config.num_hidden_layers
        self.heads = model.config.num_attention_heads
        self.pool = torch.empty(self.layers, pool_size, self.heads, 3, dtype=torch.float64)
        self.positions = []
        self.count = 0
        self.handles = []
        self.batch = None
        self.rotary = None
        self.selection = None

    def __enter__(self):
        for index, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(
                self._capture_rotary, with_kwargs=True,
            ))
            self.handles.append(layer.self_attn.q_norm.register_forward_hook(self._query_hook(index)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.batch = self.rotary = self.selection = None

    def start_batch(self, prompt_indices, capacities):
        self.prompt_indices = tuple(prompt_indices)
        self.offsets = {}
        size = 0
        for index, capacity in zip(prompt_indices, capacities):
            self.offsets[index] = size
            size += capacity
        self.batch = torch.empty(
            self.layers, size, self.heads, 3, device=self.device, dtype=torch.float64,
        )

    def select_positions(self, rows, columns, prompt_indices, sequence_positions):
        destinations = [self.offsets[index] + position
                        for index, position in zip(prompt_indices, sequence_positions)]
        self.selection = tuple(torch.tensor(values, device=self.device, dtype=torch.long)
                               for values in (rows, columns, destinations))

    def _capture_rotary(self, module, args, kwargs):
        self.rotary = kwargs["position_embeddings"]

    def _query_hook(self, layer_index):
        @torch.no_grad()
        def capture(module, args, output):
            rows, columns, destinations = self.selection
            cosine, sine = self.rotary
            keys, values = self.prefix[layer_index]
            # 每层只转换一次原始 KV，不扩展到 query-head 数量，也不跨层保留 FP32 副本。
            keys = keys[0, :, :self.compressed_end].float()
            values = values[0, :, :self.compressed_end].float()
            for start in range(0, len(rows), 16):
                selected_rows, selected_columns = rows[start:start + 16], columns[start:start + 16]
                query = output[selected_rows, selected_columns]
                cos = cosine[selected_rows, selected_columns].unsqueeze(1)
                sin = sine[selected_rows, selected_columns].unsqueeze(1)
                query = (query * cos + rotate_half(query) * sin).float()
                moments = _local_moments(query, keys, values)
                self.batch[layer_index].index_copy_(0, destinations[start:start + 16], moments)
        return capture

    def finish_batch(self, lengths):
        # 按原提示／位置顺序采样；一批内同槽被多次替换时只拷贝最终值。
        replacements = {}
        for prompt_index, length in zip(self.prompt_indices, lengths):
            for position in range(length):
                self.count += 1
                if len(self.positions) < self.pool_size:
                    slot = len(self.positions)
                    self.positions.append((prompt_index, position))
                else:
                    slot = self.rng.randrange(self.count)
                    if slot >= self.pool_size:
                        continue
                    self.positions[slot] = (prompt_index, position)
                replacements[slot] = self.offsets[prompt_index] + position
        if replacements:
            slots = torch.tensor(list(replacements), dtype=torch.long)
            sources = torch.tensor(list(replacements.values()), dtype=torch.long, device=self.device)
            selected = self.batch.index_select(1, sources).cpu()
            self.pool.index_copy_(1, slots, selected)
        self.batch = self.rotary = self.selection = None

    def finish(self, floor):
        if self.count < self.pool_size:
            raise ValueError(
                f"teacher generated {self.count} candidate positions, below required "
                f"pool_size={self.pool_size}; increase prompt/response coverage rather than duplicating positions"
            )
        # 全方差公式：跨位置/通道方差 = 通道内方差均值 + 通道均值的跨位置方差。
        output_variance = self.pool[..., 1].mean(dim=1) + self.pool[..., 0].var(dim=1, correction=0)
        mass_variance = self.pool[..., 2].var(dim=1, correction=0)
        scales = {
            "scales_o": output_variance.sqrt().float().clamp_min(floor),
            "scales_z": mass_variance.sqrt().float().clamp_min(floor),
        }
        return torch.tensor(self.positions, dtype=torch.int64), self.count, scales
