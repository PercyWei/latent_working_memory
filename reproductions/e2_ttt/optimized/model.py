"""Batch inference with independent per-sample memory and left padding."""

from fla.models.utils import Cache
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from e2_ttt.layers.e2_ttt_mlp import E2TTTMLP
from e2_ttt.layers.e2_ttt_swiglu import E2TTTSwiGLU
from e2_ttt.models.e2_ttt_mlp.modeling import E2TTTMLPForCausalLM, E2TTTMLPModel
from e2_ttt.models.e2_ttt_swiglu.modeling import E2TTTSwiGLUForCausalLM, E2TTTSwiGLUModel

from .padding import padding_forward, padding_layout


class BatchedModel:
    def forward(
        self, input_ids=None, attention_mask=None, inputs_embeds=None,
        past_key_values=None, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None, **kwargs,
    ):
        if self.training:
            raise ValueError("The optimized model is an inference-only implementation")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("The optimized model requires separate, unpacked batch rows")
        shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        if attention_mask is not None:
            if (attention_mask.ndim != 2 or attention_mask.shape[0] != shape[0]
                    or attention_mask.shape[1] < shape[1]
                    or not bool(((attention_mask == 0) | (attention_mask == 1)).all())):
                raise ValueError("attention_mask must be a binary matrix matching the input batch")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        batch, width, _ = inputs_embeds.shape
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise ValueError("Optimized inference requires its FLA Cache")
        if use_cache and past_key_values is None:
            past_key_values = Cache()
        previous_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        if previous_seen and width != 1:
            raise ValueError("Padded continuation accepts one token per row")
        mask = (
            torch.ones((batch, width), device=inputs_embeds.device, dtype=torch.bool)
            if attention_mask is None else attention_mask[:, -width:].bool()
        )
        if not previous_seen:
            if not bool(mask.any(-1).all()) or bool((mask[:, :-1] & ~mask[:, 1:]).any()):
                raise ValueError("Prefill requires nonempty rows with standard left padding")
        layout = padding_layout(mask)
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        hidden_states = inputs_embeds
        all_hidden = () if output_hidden_states else None
        for block in self.layers:
            if output_hidden_states:
                all_hidden += (hidden_states,)
            residual = hidden_states
            hidden_states = padding_forward(
                block.attn, block.attn_norm(hidden_states), layout, past_key_values, previous_seen,
            )
            if self.config.fuse_norm:
                hidden_states, residual = block.mlp_norm(hidden_states, residual, True)
            else:
                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = block.mlp_norm(hidden_states)
            hidden_states = residual + block.mlp(hidden_states, **kwargs)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden += (hidden_states,)
        if not return_dict:
            return tuple(value for value in (hidden_states, past_key_values, all_hidden) if value is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values, hidden_states=all_hidden,
        )


class BatchedMLPModel(BatchedModel, E2TTTMLPModel):
    pass


class BatchedSwiGLUModel(BatchedModel, E2TTTSwiGLUModel):
    pass


def optimize_model(model):
    """Adapt one loaded model in place; parameters and state_dict keys stay intact."""
    if isinstance(model, E2TTTMLPForCausalLM):
        backbone_class, original_layer = BatchedMLPModel, E2TTTMLP
    elif isinstance(model, E2TTTSwiGLUForCausalLM):
        backbone_class, original_layer = BatchedSwiGLUModel, E2TTTSwiGLU
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
        block.attn._swiglu = isinstance(block.attn, E2TTTSwiGLU)
    return model
