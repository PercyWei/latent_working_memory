"""复用官方 Llama 层和 FlashAttention，允许同一 batch 每行在不同 KV 长度处续写。"""

from auto_compressor import LlamaAutoCompressorModel
from flash_attn import flash_attn_with_kvcache
import torch


class BatchedLlamaAutoCompressor(LlamaAutoCompressorModel):
    def _read(self, ids, caches, lengths, scale):
        hidden = self.get_input_embeddings()(ids)
        batch, width = ids.shape
        for layer, (key_cache, value_cache) in zip(self.model.layers, caches, strict=True):
            attn = layer.self_attn
            inputs = layer.input_layernorm(hidden)
            q = attn.q_proj(inputs).view(batch, width, attn.num_heads, attn.head_dim)
            k = attn.k_proj(inputs).view(batch, width, attn.num_key_value_heads, attn.head_dim)
            v = attn.v_proj(inputs).view_as(k)
            rotary = attn.rotary_emb
            rotary._update_cos_sin_cache(key_cache.size(1), device=q.device, dtype=q.dtype)
            values = flash_attn_with_kvcache(
                q, key_cache, value_cache, k=k, v=v, cache_seqlens=lengths,
                rotary_cos=rotary._cos_cached, rotary_sin=rotary._sin_cached,
                rotary_interleaved=rotary.interleaved, causal=True, softmax_scale=scale,
            )
            hidden = hidden + attn.o_proj(values.reshape(batch, width, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.model.norm(hidden)

    @torch.inference_mode()
    def read_questions(self, states, suffixes, budgets, eos_id, complete):
        batch = len(states)
        prefixes = [state[0][1] for state in states]
        lengths = torch.tensor(prefixes, device=self.device, dtype=torch.int32)
        suffix_sizes = [len(suffix) for suffix in suffixes]
        width = max(suffix_sizes)
        capacity = max(max(prefixes) + width,
                       max(p + s + b for p, s, b in zip(prefixes, suffix_sizes, budgets, strict=True)))
        caches = []
        for layer, rows in zip(self.model.layers, zip(*states, strict=True), strict=True):
            attn = layer.self_attn
            shape = (batch, capacity, attn.num_key_value_heads, attn.head_dim)
            keys = rows[0][0].new_empty(shape)
            values = torch.empty_like(keys)
            for i, (kv, length) in enumerate(rows):
                # 官方前缀内重复了 GQA 的 KV heads，内核接受原始 KV heads。
                keys[i, :length] = kv[0, :length, 0, ::attn.num_key_value_groups]
                values[i, :length] = kv[0, :length, 1, ::attn.num_key_value_groups]
            caches.append((keys, values))
        inputs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(row, device=self.device) for row in suffixes], batch_first=True,
        )
        scale = float(1.0 / self.model.layers[0].self_attn.norm_factor)
        hidden = self._read(inputs, caches, lengths, scale)
        last = torch.tensor(suffix_sizes, device=self.device)
        hidden = hidden[torch.arange(batch, device=self.device), last - 1]
        # 下一 token 会覆盖问句末尾的 padding，始终从该行的真实位置续写。
        lengths += last.to(torch.int32)
        tokens = [[] for _ in states]
        active = set(range(batch))
        while active:
            predicted = self.get_output_embeddings()(hidden).argmax(dim=-1)
            for i, token in enumerate(predicted.tolist()):
                if i not in active:
                    continue
                tokens[i].append(token)
                reason = ("eos" if token == eos_id else
                          "max_new_tokens" if len(tokens[i]) == budgets[i] else None)
                if reason:
                    complete(i, {"generated_token_ids": tokens[i], "stop_reason": reason})
                    active.remove(i)
            if active:
                hidden = self._read(predicted[:, None], caches, lengths, scale)[:, 0]
                lengths[list(active)] += 1
