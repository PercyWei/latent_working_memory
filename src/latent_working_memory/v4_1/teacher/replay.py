"""独立冻结 teacher 的原文 prefill 与提示分支回放；不持有 student 状态。"""

from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import rotate_half

from ..modeling.am import local_read


def load_teacher(model_config, device):
    return AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        dtype=getattr(torch, model_config.backbone_dtype),
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)


class TeacherReplay:
    def __init__(self, model, prefill_chunk_size, batch_size):
        if model.config.model_type != "qwen3":
            raise ValueError("v4.1 teacher currently supports native Qwen3 only")
        self.model = model.eval().requires_grad_(False)
        self.device = next(model.parameters()).device
        self.prefill_chunk_size = prefill_chunk_size
        self.batch_size = batch_size

    @torch.no_grad()
    def forward(self, tokens, cache, logits=False):
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        if logits:
            return self.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        return self.model.model(input_ids=ids, past_key_values=cache, use_cache=True)

    @torch.no_grad()
    def advance(self, context_ids, cache, processed_end):
        """仅追加该 episode 中新到达的原文；跨边界保留完整 teacher KV。"""
        for start in range(cache.get_seq_length(), processed_end, self.prefill_chunk_size):
            self.forward(context_ids[start:min(start + self.prefill_chunk_size, processed_end)], cache)

    @torch.no_grad()
    def replay(self, prefix_cache, sequences, positions, compressed_end, on_read):
        """回调逐层接收 pool 索引和 q/o/z；全部为普通的无梯度 tensors。

        positions 是 [N,2] 的 (prompt_index, sequence_position)。每条分支只回放
        至其最远选中位置，包含该位置本身；原始 prefix cache 不被分支修改。
        """
        selected = defaultdict(list)
        for pool_index, (prompt_index, position) in enumerate(positions.tolist()):
            selected[prompt_index].append((pool_index, position))
        # 相近长度合批，减少右侧 padding。顺序变化不改变 pool 中的索引。
        branches = sorted(selected, key=lambda index: max(p for _, p in selected[index]))
        prefix = tuple((layer.keys, layer.values) for layer in prefix_cache.layers)
        prefix_length = prefix_cache.get_seq_length()
        bias = torch.zeros(self.model.config.num_key_value_heads, compressed_end, device=self.device)
        rotary, current = {}, {}

        def capture_rotary(module, args, kwargs):
            rotary["cos"], rotary["sin"] = kwargs["position_embeddings"]

        def query_hook(layer_index):
            def capture(module, args, output):
                rows, columns = current["rows"], current["columns"]
                query = output[rows, columns]
                cosine = rotary["cos"][rows, columns].unsqueeze(1)
                sine = rotary["sin"][rows, columns].unsqueeze(1)
                query = (query * cosine + rotate_half(query) * sine).float()
                keys, values = prefix[layer_index]
                outputs, log_mass = local_read(
                    query, keys[0, :, :compressed_end], values[0, :, :compressed_end], bias
                )
                on_read(layer_index, current["pool_indices"], query, outputs, log_mass)
            return capture

        handles = []
        for index, layer in enumerate(self.model.model.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(capture_rotary, with_kwargs=True))
            handles.append(layer.self_attn.q_norm.register_forward_hook(query_hook(index)))
        try:
            for start in range(0, len(branches), self.batch_size):
                indices = branches[start:start + self.batch_size]
                lengths = [max(position for _, position in selected[index]) + 1 for index in indices]
                max_length, batch_size = max(lengths), len(indices)
                if prefix_length + max_length > self.model.config.max_position_embeddings:
                    raise ValueError("teacher branch exceeds the backbone position limit")
                ids = torch.zeros((batch_size, max_length), dtype=torch.long)
                rows, columns, pool_indices = [], [], []
                for row, (index, length) in enumerate(zip(indices, lengths)):
                    ids[row, :length] = torch.tensor(sequences[index][:length])
                    for pool_index, position in selected[index]:
                        rows.append(row)
                        columns.append(position)
                        pool_indices.append(pool_index)
                current.update({name: torch.tensor(value, device=self.device, dtype=torch.long)
                                for name, value in (("rows", rows), ("columns", columns),
                                                    ("pool_indices", pool_indices))})
                branch_cache = DynamicCache(ddp_cache_data=tuple(
                    (keys.expand(batch_size, -1, -1, -1), values.expand(batch_size, -1, -1, -1))
                    for keys, values in prefix
                ))
                lengths_tensor = torch.tensor(lengths, device=self.device)
                attention_mask = torch.cat((
                    torch.ones((batch_size, prefix_length), device=self.device, dtype=torch.bool),
                    torch.arange(max_length, device=self.device)[None] < lengths_tensor[:, None],
                ), dim=1)
                position_ids = torch.arange(
                    prefix_length, prefix_length + max_length, device=self.device
                )[None].expand(batch_size, -1)
                self.model.model(
                    input_ids=ids.to(self.device), past_key_values=branch_cache, attention_mask=attention_mask,
                    position_ids=position_ids, use_cache=True,
                )
                del branch_cache
        finally:
            for handle in handles:
                handle.remove()
