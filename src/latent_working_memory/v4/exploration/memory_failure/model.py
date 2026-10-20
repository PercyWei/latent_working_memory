"""保持正式流式路径的四条件推理干预。"""

import torch
from transformers import GenerationConfig

from latent_working_memory.v4.model import StreamingMemoryLM


class DiagnosticMemoryLM(StreamingMemoryLM):
    def __init__(self, config, backbone=None):
        super().__init__(config, backbone)
        self.condition = "ttt"
        self.boundary_events = 0
        self.memory_update_events = 0

    def _before_attention(self, index, module, args, kwargs):
        if self.condition != "no-read":
            return super()._before_attention(index, module, args, kwargs)
        if self._active_state is not None:
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            self._source_chunks[index] = self.memory_layers[index].source(hidden)

    def _write(self, state, differentiable, active_rows=None):
        events = state.memories[0].shape[0] if active_rows is None else len(active_rows)
        self.boundary_events += events
        if self.condition != "fixed-initial":
            self.memory_update_events += events
            return super()._write(state, differentiable, active_rows)
        # 与正式 _write 的删除范围一致；保留初始 slots，不伪造未计算的 AM loss。
        count = self.config.pending_size
        state.sources = [
            source[:, count:].clone(memory_format=torch.contiguous_format)
            for source in state.sources
        ]
        for layer in state.cache.layers:
            layer.keys = layer.keys[..., count:, :].clone(memory_format=torch.contiguous_format)
            layer.values = layer.values[..., count:, :].clone(memory_format=torch.contiguous_format)
        state.write_events += events

    def generate_condition(self, ids, condition, tokenizer, max_new_tokens):
        self.condition = condition
        self.boundary_events = 0
        self.memory_update_events = 0
        self._active_state = None
        self._corrections = {}
        self._source_chunks = {}
        if condition != "full-context":
            return self.generate(ids, max_new_tokens, tokenizer.eos_token_id)
        generation = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            ),
            bos_token_id=tokenizer.bos_token_id,
            use_cache=True,
        )
        inputs = ids.unsqueeze(0)
        return self.backbone.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            generation_config=generation,
            use_model_defaults=False,
        )[0]
