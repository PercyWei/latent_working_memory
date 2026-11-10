"""Equal-length batch inference without changing the upstream model registry."""

from einops import rearrange
from flash_attn import flash_attn_func
import torch
import torch.nn.functional as F

from e2_ttt.layers.e2_ttt_mlp import E2TTTMLP
from e2_ttt.layers.e2_ttt_swiglu import E2TTTSwiGLU
from e2_ttt.models.e2_ttt_mlp.modeling import E2TTTMLPForCausalLM, E2TTTMLPModel
from e2_ttt.models.e2_ttt_swiglu.modeling import E2TTTSwiGLUForCausalLM, E2TTTSwiGLUModel

from . import mlp, swiglu


class EqualLengthModel:
    def forward(
        self, input_ids=None, attention_mask=None, inputs_embeds=None,
        past_key_values=None, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None, **kwargs,
    ):
        if self.training:
            raise ValueError("The optimized model is an inference-only implementation")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("The optimized model requires separate, unpacked batch rows")
        if attention_mask is not None:
            if attention_mask.ndim != 2 or not bool(torch.all(attention_mask == 1)):
                raise ValueError("The optimized model requires equal-length inputs without padding")
        # The upstream layer packs every non-None mask into a single row. A
        # validated all-one mask can be removed while preserving each B row.
        return super().forward(
            input_ids=input_ids, attention_mask=None, inputs_embeds=inputs_embeds,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, **kwargs,
        )


class BatchedMLPModel(EqualLengthModel, E2TTTMLPModel):
    pass


class BatchedSwiGLUModel(EqualLengthModel, E2TTTSwiGLUModel):
    pass


class BatchedLayer:
    """Native eval forward with the true B dimension and local TTT kernels."""

    def forward(
        self, hidden_states, attention_mask=None, past_key_values=None,
        use_cache=False, output_attentions=False, **kwargs,
    ):
        if attention_mask is not None or kwargs.get("cu_seqlens") is not None:
            raise ValueError("The optimized layer accepts only unpacked, unpadded batch rows")
        _, q_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        q, k = self.q_proj(hidden_states), self.k_proj(hidden_states)
        v = F.silu(self.v_proj(hidden_states))
        q, k = (
            rearrange(value, "... (h d) -> ... h d", d=self.head_k_dim)
            for value in (q, k)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.use_local_pos_encoding:
            seqlen_offset, max_seqlen = 0, q_len
            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                max_seqlen = q.shape[1] + seqlen_offset
            if self.max_position_embeddings is not None:
                max_seqlen = max(max_seqlen, self.max_position_embeddings)
            q_kv, k_kv = self.rotary(
                q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen,
                cu_seqlens=None,
            )
        else:
            q_kv, k_kv = q, k
        v_kv = v
        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=[k_kv.flatten(-2, -1), v_kv.flatten(-2, -1)],
                layer_idx=self.layer_idx, offset=q_len,
                cache_kwargs=dict(window_size=self.chunk_size),
            )["attn_state"]
            if cache_has_content:
                k_kv = rearrange(k_cached, "... (h d) -> ... h d", d=self.head_k_dim)
                v_kv = rearrange(v_cached, "... (h d) -> ... h d", d=self.head_v_dim)
        o_kv = None
        if not self._swiglu or self.use_window_attention:
            o_kv = flash_attn_func(
                q_kv, k_kv, v_kv, causal=True,
                window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size - 1, 0),
            )

        q, k = F.silu(q), F.silu(k)
        q, k = (F.normalize(value, p=2, dim=-1, eps=1e-8) for value in (q, k))
        lr = F.sigmoid(self.lr_proj(hidden_states).float()) * self.base_lr
        lr = lr.unsqueeze(-1)
        log_decay = None
        if self.use_ttt_decay:
            decay = F.sigmoid(self.decay_proj(hidden_states).float()) * self.base_decay
            log_decay = torch.log1p(-lr * decay.unsqueeze(-1))
        log_momentum = F.logsigmoid(self.momentum_proj(hidden_states).float()) / 16
        log_momentum = log_momentum.unsqueeze(-1)
        if log_decay is None:
            log_decay = torch.zeros_like(lr).float()
        lr = lr * (1 / self.ttt_chunk_size)
        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        kernels = swiglu if self._swiglu else mlp
        if recurrent_state is None:
            recurrent_state = (self.w0, self.w1, self.w2) if self._swiglu else (self.w0, self.w1)
            fn_ttt = kernels.chunk_ttt_momentum
        else:
            fn_ttt = kernels.chunk_ttt_momentum_with_cache
        kernel_kwargs = dict(use_post_norm=self.use_post_norm) if self._swiglu else dict(use_grad_clip=self.use_grad_clip)
        if last_state is None or last_state["recurrent_state"] is None:
            kernel_kwargs["is_training"] = False
        o_fw, recurrent_state = fn_ttt(
            q, k, v, lr, log_decay, log_momentum, recurrent_state,
            self.ttt_norm_weight, self.ttt_norm_bias,
            chunk_size=self.ttt_chunk_size, verbose=False,
            use_closed_form=self.use_closed_form, **kernel_kwargs,
        )
        if self.use_memory_gate:
            mg = rearrange(
                self.memory_gate_proj(hidden_states), "... (h d) -> ... h d",
                d=self.head_v_dim,
            ).sigmoid()
            o = mg * o_fw + (1 - mg) * o_kv
        else:
            o = o_fw + o_kv if o_kv is not None else o_fw
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=None,
                layer_idx=self.layer_idx, offset=0,
            )
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        return self.o_proj(o), None, past_key_values


class BatchedMLPLayer(BatchedLayer, E2TTTMLP):
    _swiglu = False


class BatchedSwiGLULayer(BatchedLayer, E2TTTSwiGLU):
    _swiglu = True


def optimize_model(model):
    """Adapt one loaded model in place; parameters and state_dict keys stay intact."""
    if isinstance(model, E2TTTMLPForCausalLM):
        backbone_class, original_layer, layer_class = BatchedMLPModel, E2TTTMLP, BatchedMLPLayer
    elif isinstance(model, E2TTTSwiGLUForCausalLM):
        backbone_class, original_layer, layer_class = BatchedSwiGLUModel, E2TTTSwiGLU, BatchedSwiGLULayer
    else:
        raise ValueError("Optimized batching supports the released E²-TTT MLP/SwiGLU models")
    for block in model.model.layers:
        layer = block.attn
        if not isinstance(layer, original_layer):
            raise ValueError("Optimized batching does not support hybrid attention layers")
        if layer.use_short_conv:
            raise ValueError("Optimized batching supports the released models without short convolution")
        if not layer.use_ttt_momentum or not layer.use_closed_form:
            raise ValueError("Optimized batching requires the released closed-form momentum configuration")
        if isinstance(layer, E2TTTSwiGLU) and layer.use_fused_ttt_kernel:
            raise ValueError("Optimized batching retains the reference SwiGLU kernel")
    model.model.__class__ = backbone_class
    for block in model.model.layers:
        block.attn.__class__ = layer_class
    return model
