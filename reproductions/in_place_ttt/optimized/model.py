"""保留官方计算公式和参数结构的 Qwen3 混长批量推理。"""

from inference_model.hf_qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
)
from torch import nn
from transformers.generation import GenerationMixin

from .padding import CacheStoppingCriteriaList, PaddedQwen3Model, PaddedTTTDynamicCache
from .ttt import grouped_mlp_forward


class OptimizedQwen3MLP(Qwen3MLP):
    def __init__(self, original):
        # 复用构造阶段的原模块，不再创建或初始化任何参数。
        nn.Module.__init__(self)
        self.config = original.config
        self.hidden_size = original.hidden_size
        self.intermediate_size = original.intermediate_size
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj
        self.act_fn = original.act_fn
        self.layer_idx = original.layer_idx
        if hasattr(original, "ttt_conv"):
            self.ttt_chunk = original.ttt_chunk
            self.ttt_proj = original.ttt_proj
            self.ttt_lr = original.ttt_lr
            self.ttt_conv = original.ttt_conv
        self.mlp_chunk_size = 0

    def forward(self, x, t=None, past_w=None):
        if self.mlp_chunk_size == 0 or x.shape[1] <= self.mlp_chunk_size:
            return super().forward(x, t, past_w)
        if not hasattr(self, "ttt_conv"):
            return grouped_mlp_forward(self, x)
        # The upstream decoder uses the original single-row MLP signature.
        # Mixed batches enter ttt_forward with independent sample states.
        if x.shape[0] != 1:
            raise ValueError("TTT batches require the optimized sample-state scheduler")
        values, weights = grouped_mlp_forward(
            self, x, t, None if past_w is None else past_w.unsqueeze(0)
        )
        return values, weights[0]


class OptimizedQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            layer.mlp = OptimizedQwen3MLP(layer.mlp)
        self.compact_tails = False

    def set_optimization(self, compact_tails: bool, mlp_chunk_size: int):
        if mlp_chunk_size < 0:
            raise ValueError("mlp_chunk_size must be nonnegative")
        for layer in self.model.layers:
            if (
                mlp_chunk_size
                and hasattr(layer.mlp, "ttt_conv")
                and mlp_chunk_size % layer.mlp.ttt_chunk
            ):
                raise ValueError("mlp_chunk_size must be a multiple of each TTT chunk size")
        self.compact_tails = compact_tails
        self.model.__class__ = PaddedQwen3Model if compact_tails else Qwen3Model
        for layer in self.model.layers:
            layer.mlp.mlp_chunk_size = mlp_chunk_size

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def _get_stopping_criteria(self, generation_config, stopping_criteria, tokenizer=None):
        criteria = super()._get_stopping_criteria(generation_config, stopping_criteria, tokenizer)
        if self.compact_tails:
            return CacheStoppingCriteriaList(criteria, self._generation_cache)
        return criteria

    def generate(self, **kwargs):
        if not self.compact_tails:
            return super().generate(**kwargs)
        if kwargs.get("past_key_values") is not None:
            raise ValueError("Each optimized generate call starts with fresh sample states")
        input_ids = kwargs["input_ids"]
        generation_config = kwargs.get("generation_config", self.generation_config)
        if (kwargs.get("num_beams", generation_config.num_beams) != 1
                or kwargs.get("num_return_sequences", generation_config.num_return_sequences) != 1
                or kwargs.get("do_sample", generation_config.do_sample)):
            raise ValueError("Optimized evaluation supports greedy, single-beam generation")
        cache = PaddedTTTDynamicCache(input_ids.shape[0], input_ids.device)
        cache.prompt_width = input_ids.shape[1]
        kwargs["past_key_values"] = cache
        self._generation_cache = cache
        try:
            # Invoke the HF loop with our sample cache, bypassing the upstream B1 guard.
            return GenerationMixin.generate(self, **kwargs)
        finally:
            del self._generation_cache
