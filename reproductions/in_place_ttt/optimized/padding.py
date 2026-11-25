"""按有效序列长度分组计算 attention，并维护各样本的 KV 和 TTT 状态。"""

from dataclasses import dataclass

from inference_model.hf_qwen3.modeling_qwen3 import (
    Qwen3Model,
    TTTDynamicCache,
    apply_rotary_pos_emb,
)
import torch
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .ttt import ttt_forward


@dataclass
class PaddingStep:
    width: int
    lengths: list[int]
    position_ids: torch.Tensor


class PaddedTTTDynamicCache(TTTDynamicCache):
    def __init__(self, batch_size, device):
        super().__init__()
        self.rows = [TTTDynamicCache() for _ in range(batch_size)]
        self.seen = [0] * batch_size
        self.physical_seen = 0
        self.finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
        self.generated_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.prompt_width = None
        self.stop_trackers = []

    def get_seq_length(self, layer_idx=0):
        # HF uses the padded width to slice newly appended input IDs.
        return self.physical_seen

    def prepare_step(self, input_ids, attention_mask, position_ids):
        batch, width = input_ids.shape
        if batch != len(self.rows):
            raise ValueError("The input batch must match the cached sample states")
        if self.physical_seen and width != 1:
            raise ValueError("Cached optimized inference accepts one token per row")
        if attention_mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            if (attention_mask.ndim != 2 or attention_mask.shape[0] != batch
                    or attention_mask.shape[1] != self.physical_seen + width):
                raise ValueError("attention_mask must match the full padded input history")
            mask = attention_mask[:, -width:]
        valid = mask.to(torch.bool) & ~self.finished[:, None]
        host_mask = valid.detach().cpu().tolist()
        lengths = []
        for row in host_mask:
            length = sum(row)
            if row != [False] * (width - length) + [True] * length:
                raise ValueError("Optimized inference requires contiguous left padding")
            if not self.physical_seen and length == 0:
                raise ValueError("Every prefill row must contain real tokens")
            lengths.append(length)
        if position_ids is None:
            offsets = torch.tensor(
                [seen + length - width for seen, length in zip(self.seen, lengths, strict=True)],
                device=input_ids.device, dtype=torch.long,
            )
            position_ids = (offsets[:, None] + torch.arange(width, device=input_ids.device)).clamp_min(0)
        elif position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match the current input batch")
        return PaddingStep(width, lengths, position_ids)

    def advance(self, step):
        self.physical_seen += step.width
        for index, length in enumerate(step.lengths):
            self.seen[index] += length

    def finish_rows(self, stopped, sequence_width):
        newly_finished = stopped & ~self.finished
        self.generated_lengths = torch.where(
            newly_finished, sequence_width - self.prompt_width, self.generated_lengths
        )
        self.finished = self.finished | stopped


class CacheStoppingCriteriaList(StoppingCriteriaList):
    def __init__(self, criteria, cache):
        super().__init__(criteria)
        self.cache = cache

    def __call__(self, input_ids, scores, **kwargs):
        stopped = super().__call__(input_ids, scores, **kwargs)
        finished = stopped.clone()
        # OpenCompass tracks rows separately but returns an all-rows scalar.
        # Observe its existing matches without changing HF's stop decision.
        for tracker in self.cache.stop_trackers:
            finished = finished | torch.tensor(
                tracker, device=finished.device, dtype=torch.bool
            )
        self.cache.finish_rows(finished, input_ids.shape[1])
        return stopped


def attention_forward(layer, hidden_states, position_embeddings, cache, step):
    batch, width, _ = hidden_states.shape
    shape = (batch, width, -1, layer.head_dim)
    query = layer.q_norm(layer.q_proj(hidden_states).view(shape)).transpose(1, 2)
    key = layer.k_norm(layer.k_proj(hidden_states).view(shape)).transpose(1, 2)
    value = layer.v_proj(hidden_states).view(shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    groups = {}
    for sample, length in enumerate(step.lengths):
        if not length:
            continue
        current_q = query[sample : sample + 1, :, -length:]
        current_k = key[sample : sample + 1, :, -length:].clone()
        current_v = value[sample : sample + 1, :, -length:].clone()
        keys, values = cache.rows[sample].update(current_k, current_v, layer.layer_idx)
        groups.setdefault((length, keys.shape[-2]), []).append((sample, current_q, keys, values))
    output = query.new_zeros((batch, width, query.shape[1], layer.head_dim))
    attention = ALL_ATTENTION_FUNCTIONS[layer.config._attn_implementation]
    for (query_length, _), group in groups.items():
        # Separate rows share a kernel only when both sequence dimensions agree.
        if len(group) == 1:
            _, queries, keys, values = group[0]
        else:
            queries = torch.cat([item[1] for item in group], dim=0)
            keys = torch.cat([item[2] for item in group], dim=0)
            values = torch.cat([item[3] for item in group], dim=0)
        attended, _ = attention(
            layer, queries, keys, values, attention_mask=None,
            dropout=0.0, scaling=layer.scaling, is_causal=query_length > 1,
        )
        for row, (sample, _, _, _) in enumerate(group):
            output[sample, -query_length:] = attended[row]
    return layer.o_proj(output.reshape(batch, width, -1))


class PaddedQwen3Model(Qwen3Model):
    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, use_cache=None,
        cache_position=None, **kwargs,
    ):
        if self.training or input_ids is None or inputs_embeds is not None:
            raise ValueError("Optimized inference accepts input_ids in evaluation mode")
        if not use_cache:
            raise ValueError("Optimized inference requires use_cache=True")
        if self.config._attn_implementation not in ("flash_attention_2", "sdpa") or self.has_sliding_layers:
            raise ValueError("Optimized inference requires full attention with FlashAttention 2 or SDPA")
        if self.rotary_emb.rope_type not in ("default", "linear", "yarn", "llama3"):
            raise ValueError("Optimized mixed batches require a sequence-independent RoPE configuration")
        if kwargs.get("output_attentions") or kwargs.get("output_hidden_states"):
            raise ValueError("Optimized inference returns logits and its independent sample cache")
        cache = past_key_values
        if cache is None:
            cache = PaddedTTTDynamicCache(input_ids.shape[0], input_ids.device)
        if not isinstance(cache, PaddedTTTDynamicCache):
            raise ValueError("Optimized inference requires PaddedTTTDynamicCache")
        step = cache.prepare_step(input_ids, attention_mask, position_ids)
        inputs_embeds = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(inputs_embeds, step.position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers[: self.config.num_hidden_layers]:
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            hidden_states = residual + attention_forward(
                layer.self_attn, normalized, position_embeddings, cache, step
            )
            residual = hidden_states
            normalized = layer.post_attention_layernorm(hidden_states)
            if hasattr(layer.mlp, "ttt_conv"):
                target = self._resolve_ttt_target_states(layer, inputs_embeds)
                if target is None:
                    target = normalized
                states = [row.ttt_states[layer.layer_idx] for row in cache.rows]
                hidden_states, states = ttt_forward(
                    layer.mlp, normalized, target, states, step.lengths
                )
                for row, state in zip(cache.rows, states, strict=True):
                    row.TTT_update(state, layer.layer_idx)
            else:
                hidden_states = layer.mlp(normalized)
            hidden_states = residual + hidden_states
        hidden_states = self.norm(hidden_states)
        cache.advance(step)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=cache)
