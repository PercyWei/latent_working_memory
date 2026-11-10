"""Incremental unpadded batch inference alongside the unchanged upstream model."""

from dataclasses import dataclass

from einops import rearrange
from flash_attn import flash_attn_varlen_func
from lact_model import LaCTForCausalLM
from lact_model.ttt_operation import l2_norm
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from .ttt import TTTState


@dataclass
class LayerCache:
    seen: int = 0
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    ttt: TTTState | None = None
    query_offsets: torch.Tensor | None = None
    key_offsets: torch.Tensor | None = None
    query_length: int = 0
    key_length: int = 0


class LaCTCache:
    def __init__(self, num_layers):
        self.layers = [LayerCache() for _ in range(num_layers)]

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].seen


def project_each_sample(projection, hidden_states):
    """Keep sensitive projections at the B1 GEMM shape under the caller's autocast.

    Small control projections and the vocabulary head can choose different numerical
    kernels when the sample dimension changes. Evaluate their independent rows with
    the original single-sample shape, then restore the batch dimension.
    """
    if hidden_states.shape[0] == 1:
        return projection(hidden_states)
    return torch.cat(
        [projection(hidden_states[index:index + 1]) for index in range(hidden_states.shape[0])],
        dim=0,
    )


def attention_forward(layer, hidden_states, cache):
    batch_size, q_len, _ = hidden_states.shape
    q, k, v = layer.qkv(hidden_states).chunk(3, dim=-1)
    if layer.attn_qk_norm:
        q, k = layer.q_norm(q), layer.k_norm(k)
    fast_q, fast_k = layer._rescale_qk(q, k)
    fast_v = v
    q = rearrange(q, "b s (h d) -> b s h d", d=layer.head_dim)
    k = rearrange(k, "b s (h d) -> b s h d", d=layer.head_dim)
    v = rearrange(v, "b s (h d) -> b s h d", d=layer.head_dim)
    max_seqlen = cache.seen + q_len
    if layer.max_position_embeddings is not None:
        max_seqlen = max(max_seqlen, layer.max_position_embeddings)
    q, k = layer.rotary(q, k, seqlen_offset=cache.seen, max_seqlen=max_seqlen)
    if cache.keys is not None:
        k = torch.cat((cache.keys, k), dim=1)
        v = torch.cat((cache.values, v), dim=1)
    kv_len = k.shape[1]
    # Every row has the same length; offsets keep samples independent in varlen attention.
    if cache.query_length != q_len or cache.query_offsets.numel() != batch_size + 1:
        cache.query_offsets = torch.tensor(
            [index * q_len for index in range(batch_size + 1)],
            dtype=torch.int32,
            device=q.device,
        )
        cache.query_length = q_len
    if cache.key_length != kv_len or cache.key_offsets.numel() != batch_size + 1:
        cache.key_offsets = torch.tensor(
            [index * kv_len for index in range(batch_size + 1)],
            dtype=torch.int32,
            device=q.device,
        )
        cache.key_length = kv_len
    output = flash_attn_varlen_func(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        cu_seqlens_q=cache.query_offsets,
        cu_seqlens_k=cache.key_offsets,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        causal=True,
        window_size=(-1, -1) if layer.window_size is None else (layer.window_size - 1, 0),
    ).reshape(batch_size, q_len, -1)
    keep = kv_len if layer.window_size is None else min(kv_len, layer.window_size - 1)
    cache.keys = k[:, kv_len - keep :].clone()
    cache.values = v[:, kv_len - keep :].clone()

    fast_q = rearrange(fast_q, "b s (h d) -> (b h) s d", h=layer.num_fw_heads)
    fast_k = rearrange(fast_k, "b s (h d) -> (b h) s d", h=layer.num_fw_heads)
    fast_v = rearrange(fast_v, "b s (h d) -> (b h) s d", h=layer.num_fw_heads)
    if layer.qkv_silu:
        fast_q, fast_k = F.silu(fast_q), F.silu(fast_k)
        if not layer.no_v_silu:
            fast_v = F.silu(fast_v)
    fast_q, fast_k = l2_norm(fast_q), l2_norm(fast_k)
    if not layer.ttt_nope:
        fast_q = rearrange(fast_q, "(b h) s d -> b s (h d)", h=layer.num_fw_heads)
        fast_k = rearrange(fast_k, "(b h) s d -> b s (h d)", h=layer.num_fw_heads)
        fast_q = rearrange(fast_q, "b s (h d) -> b s h d", h=layer.num_heads)
        fast_k = rearrange(fast_k, "b s (h d) -> b s h d", h=layer.num_heads)
        fast_q, fast_k = layer.rotary(
            fast_q, fast_k, seqlen_offset=cache.seen, max_seqlen=max_seqlen
        )
        fast_q = rearrange(fast_q, "b s h d -> b s (h d)")
        fast_k = rearrange(fast_k, "b s h d -> b s (h d)")
        fast_q = rearrange(fast_q, "b s (h d) -> (b h) s d", h=layer.num_fw_heads)
        fast_k = rearrange(fast_k, "b s (h d) -> (b h) s d", h=layer.num_fw_heads)

    if cache.ttt is None:
        w0 = layer.w0() if layer.w0_w2_low_rank > 0 else layer.w0
        w2 = layer.w2() if layer.w0_w2_low_rank > 0 else layer.w2
        w0 = w0.repeat(batch_size, 1, 1)
        w1 = layer.w1.repeat(batch_size, 1, 1)
        w2 = w2.repeat(batch_size, 1, 1)
        if layer.fp32_states:
            w0, w1, w2 = w0.float(), w1.float(), w2.float()
        cache.ttt = TTTState(
            w0,
            w1,
            w2,
            layer.lact_chunk_size,
            layer.ttt_prenorm,
            layer.use_muon,
            layer.use_momentum,
        )
    if layer.lr_parameterization != "mamba":
        raise ValueError("Incremental inference requires the checkpoint's mamba LR rule")
    lr = F.softplus(project_each_sample(layer.lr_proj, hidden_states).float() + layer.base_lr_inv)
    lr = rearrange(lr, "b s (h d) -> (b h) s d", h=layer.num_fw_heads)
    lr0, lr1, lr2 = lr.chunk(3, dim=-1)
    momentum = None
    if layer.use_momentum:
        momentum = rearrange(
            project_each_sample(layer.momentum_proj, hidden_states).float(),
            "b s (h d) -> (b h) s d",
            h=layer.num_fw_heads,
        )
    fast_output = cache.ttt.forward(fast_q, fast_k, fast_v, lr0, lr1, lr2, momentum)
    fast_output = layer.ttt_norm(fast_output)
    if layer.learnable_ttt_scale:
        scale = F.silu(project_each_sample(layer.ttt_scale_proj, hidden_states), inplace=False)
        scale = rearrange(scale, "b s (h d) -> (b h) s d", h=layer.num_fw_heads)
        fast_output = fast_output * scale
    fast_output = rearrange(fast_output, "(b h) s d -> b s (h d)", h=layer.num_fw_heads)
    cache.seen += q_len
    return layer.o_proj(output + fast_output)


class IncrementalLaCTForCausalLM(LaCTForCausalLM):
    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=True,
        logits_to_keep=None,
        **kwargs,
    ):
        if not use_cache:
            return super().prepare_inputs_for_generation(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        if inputs_embeds is not None:
            raise ValueError("Incremental generation accepts input_ids")
        if past_key_values is not None:
            input_ids = input_ids[:, past_key_values.get_seq_length() :]
        return {
            "input_ids": input_ids.contiguous(),
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
            "logits_to_keep": logits_to_keep if logits_to_keep is not None else 1,
        }

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        logits_to_keep=0,
        **kwargs,
    ):
        if not use_cache:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        if self.training or labels is not None or inputs_embeds is not None:
            raise ValueError("Incremental LaCT is an input_ids inference path")
        if output_attentions or output_hidden_states or return_dict is False:
            raise ValueError("Incremental LaCT returns last logits and its cache")
        if input_ids is None or input_ids.ndim != 2 or 0 in input_ids.shape:
            raise ValueError("Incremental LaCT requires a nonempty unpadded batch")
        if self.config.use_fused_kernel:
            raise ValueError("Incremental LaCT preserves the non-fused TTT implementation")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("Packed sequences are not supported by incremental LaCT")
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != input_ids.shape[0]:
                raise ValueError("Attention mask must match the input batch")
            if not bool(attention_mask.all()):
                raise ValueError("Padding is not supported by this incremental inference path")
        cache = past_key_values
        if cache is None:
            cache = LaCTCache(len(self.model.layers))
        elif (
            cache.layers[0].keys is not None
            and cache.layers[0].keys.shape[0] != input_ids.shape[0]
        ):
            raise ValueError("Input batch size differs from the existing cache")
        hidden_states = self.model.embeddings(input_ids)
        for block, layer_cache in zip(self.model.layers, cache.layers, strict=True):
            residual = hidden_states
            hidden_states = block.attn_norm(hidden_states)
            hidden_states = attention_forward(block.attn, hidden_states, layer_cache)
            if self.config.fuse_norm:
                hidden_states, residual = block.mlp_norm(hidden_states, residual, True)
            else:
                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = block.mlp_norm(hidden_states)
            hidden_states = residual + block.mlp(hidden_states)
        hidden_states = self.model.norm(hidden_states)
        indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        return CausalLMOutputWithPast(
            logits=project_each_sample(self.lm_head, hidden_states[:, indices]),
            past_key_values=cache,
        )
