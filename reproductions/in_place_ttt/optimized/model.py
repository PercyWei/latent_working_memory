"""保持官方更新公式和 checkpoint key 的 Qwen3 推理内存优化。"""

from einops import rearrange
from inference_model.hf_qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3MLP,
    TTTDynamicCache,
)
from opt_einsum import contract
import torch
from torch import nn


class CompactTTTDynamicCache(TTTDynamicCache):
    def TTT_update(self, ttt_state, layer_idx):
        past_h, past_t, past_w = ttt_state
        if past_h is not None:
            if past_h.untyped_storage().nbytes() > past_h.numel() * past_h.element_size():
                past_h = past_h.clone()
            if past_t.untyped_storage().nbytes() > past_t.numel() * past_t.element_size():
                past_t = past_t.clone()
        super().TTT_update((past_h, past_t, past_w), layer_idx)


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
        seq_len = x.shape[1]
        if self.mlp_chunk_size == 0 or seq_len <= self.mlp_chunk_size:
            return super().forward(x, t, past_w)

        ttt = hasattr(self, "ttt_conv")
        if not ttt or t is None or seq_len < self.ttt_chunk:
            current_w = (
                (self.down_proj.weight.clone() if past_w is None else past_w) if ttt else None
            )
            output = None
            for start in range(0, seq_len, self.mlp_chunk_size):
                end = min(start + self.mlp_chunk_size, seq_len)
                part = x[:, start:end, :]
                h = self.act_fn(self.gate_proj(part)) * self.up_proj(part)
                y = (
                    nn.functional.linear(h, current_w, self.down_proj.bias)
                    if ttt
                    else self.down_proj(h)
                )
                if output is None:
                    output = y.new_empty((x.shape[0], seq_len, self.hidden_size))
                output[:, start:end, :] = y
            return (output, current_w) if ttt else output

        if x.shape[0] != 1:
            raise ValueError("TTT inference supports batch size 1")
        current_w = self.down_proj.weight.clone() if past_w is None else past_w
        # 保持官方卷积输入形状，一次计算各自独立、以零填充边界的 TTT chunk。
        t_padded = self.padding(t)
        bs, chunk_num, chunk_size, _ = t_padded.shape
        t_conv = (
            self.ttt_conv(t_padded.transpose(-1, -2).reshape(bs * chunk_num, -1, chunk_size))
            .transpose(-1, -2)
            .reshape(bs, chunk_num, chunk_size, -1)
        )
        y = torch.zeros_like(t_conv)
        for start in range(0, seq_len, self.mlp_chunk_size):
            end = min(start + self.mlp_chunk_size, seq_len)
            part = x[:, start:end, :]
            h = self.act_fn(self.gate_proj(part)) * self.up_proj(part)
            h_padded = self.padding(h)
            first_chunk = start // self.ttt_chunk
            for offset, current_h in enumerate(h_padded[0]):
                index = first_chunk + offset
                current_t = t_conv[0][index]
                current_y = contract("d h, c h -> c d", current_w, current_h)
                y[0][index] = current_y
                if seq_len % self.ttt_chunk == 0 or index != chunk_num - 1:
                    if self.ttt_proj is not None:
                        dw = (
                            contract(
                                "c h, c d, d e -> e h", current_h, current_t, self.ttt_proj.weight
                            )
                            * self.ttt_lr
                        )
                    else:
                        dw = contract("c h, c d -> d h", current_h, current_t) * self.ttt_lr
                    current_w = current_w + dw
        out = rearrange(y, "b t c d -> b (t c) d")[:, :seq_len, :]
        return out, current_w


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
        for layer in self.model.layers:
            layer.mlp.mlp_chunk_size = mlp_chunk_size

    def generate(self, **kwargs):
        if self.compact_tails and self.model.ttt_mode:
            if kwargs.get("past_key_values") is not None:
                raise ValueError("Each compact-tail generate call starts with a fresh cache")
            kwargs["past_key_values"] = CompactTTTDynamicCache()
        return super().generate(**kwargs)
