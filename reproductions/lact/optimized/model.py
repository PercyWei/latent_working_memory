"""保留原始 LaCT 公式的统一批量推理，支持等长与左 padding 输入。"""

from lact_model import LaCTForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .padding import PaddedLaCTCache, attention_forward_padded


class OptimizedLaCTForCausalLM(LaCTForCausalLM):
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
            raise ValueError("Optimized generation accepts input_ids")
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
            raise ValueError("Optimized LaCT is an input_ids inference path")
        if output_attentions or output_hidden_states or return_dict is False:
            raise ValueError("Optimized LaCT returns last logits and its cache")
        if input_ids is None or input_ids.ndim != 2 or 0 in input_ids.shape:
            raise ValueError("Optimized LaCT requires a nonempty input batch")
        if self.config.use_fused_kernel:
            raise ValueError("Optimized LaCT preserves the non-fused TTT implementation")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("Packed sequences are not supported by optimized LaCT")
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != input_ids.shape[0]:
                raise ValueError("Attention mask must match the input batch")
        cache = past_key_values
        if cache is None:
            cache = PaddedLaCTCache(len(self.model.layers), input_ids.shape[0])
        step = cache.prepare_step(input_ids, attention_mask)
        hidden_states = self.model.embeddings(input_ids)
        for block, layer_cache in zip(self.model.layers, cache.layers, strict=True):
            residual = hidden_states
            hidden_states = block.attn_norm(hidden_states)
            hidden_states = attention_forward_padded(block.attn, hidden_states, layer_cache, step)
            if self.config.fuse_norm:
                hidden_states, residual = block.mlp_norm(hidden_states, residual, True)
            else:
                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = block.mlp_norm(hidden_states)
            hidden_states = residual + block.mlp(hidden_states)
        hidden_states = self.model.norm(hidden_states)
        cache.physical_seen += input_ids.shape[1]
        indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        return CausalLMOutputWithPast(
            logits=self.lm_head(hidden_states[:, indices]),
            past_key_values=cache,
        )
