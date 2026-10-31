"""慢参数定义 memory 接口；inner 仅更新当前样本的 latent 状态。"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import rotate_half

from .am import local_read, matching_loss, teacher_scales


class LayerMemory(nn.Module):
    def __init__(self, config, num_kv_heads, head_dim):
        super().__init__()
        self.config = config
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.initial_slots = nn.Parameter(torch.empty(config.num_slots, config.memory_dim))
        nn.init.normal_(self.initial_slots, std=config.init_std)
        self.decoder = nn.ModuleDict({
            "input": nn.Linear(config.memory_dim, config.decoder_hidden_dim),
            "key": nn.Linear(config.decoder_hidden_dim, num_kv_heads * head_dim),
            "value": nn.Linear(config.decoder_hidden_dim, num_kv_heads * head_dim),
            "bias": nn.Linear(config.decoder_hidden_dim, num_kv_heads),
        })
        for module in self.decoder.values():
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        self.raw_step_size = nn.Parameter(torch.tensor(math.log(math.expm1(config.inner_lr_init))))

    def decode(self, memory, compressed_tokens, rotary):
        hidden = F.silu(self.decoder["input"](memory.float()))
        shape = (*memory.shape[:-1], self.num_kv_heads, self.head_dim)
        key = self.decoder["key"](hidden).view(shape).transpose(-3, -2)
        value = self.decoder["value"](hidden).view(shape).transpose(-3, -2)
        bias = self.decoder["bias"](hidden).transpose(-2, -1)
        positions = torch.arange(memory.shape[-2], device=memory.device, dtype=torch.float32)
        positions = positions * (compressed_tokens / memory.shape[-2])
        cosine, sine = rotary(key, positions[None])
        key = key * cosine + rotate_half(key) * sine
        return key, value, bias

    def _matching_loss(self, candidate, queries, teacher_outputs, teacher_mass,
                       scales_o, scales_z, compressed_tokens, rotary):
        outputs, mass = local_read(queries, *self.decode(candidate, compressed_tokens, rotary))
        return matching_loss(outputs, mass, teacher_outputs, teacher_mass, scales_o, scales_z)

    def _matching_gradient(self, candidate, *args):
        # 重算求导产生的高阶图；各样本的 inner 步长不随 batch 改变。
        with torch.enable_grad():
            loss = self._matching_loss(candidate, *args)
            gradient = torch.autograd.grad(loss.sum(), candidate, create_graph=True)[0]
        return gradient, loss.detach()

    def write(self, old_memory, teacher_kv, queries, compressed_tokens, rotary, differentiable):
        """Teacher/queries 轮内固定；clone 隔离局部偏导，不截断 outer 历史依赖。"""
        with torch.enable_grad():
            candidate = old_memory.clone() if differentiable else old_memory.detach().clone()
            candidate.requires_grad_(True)
            recompute = differentiable and self.config.gradient_checkpointing
            if recompute:
                teacher_outputs, teacher_mass = checkpoint(
                    local_read, queries, *teacher_kv, use_reentrant=False
                )
            else:
                teacher_outputs, teacher_mass = local_read(queries, *teacher_kv)
            scales_o, scales_z = teacher_scales(
                teacher_outputs, teacher_mass, self.config.am_std_floor
            )
            step_size = F.softplus(self.raw_step_size)
            before = None
            for _ in range(self.config.inner_steps):
                args = (
                    candidate, queries, teacher_outputs, teacher_mass,
                    scales_o, scales_z, compressed_tokens, rotary,
                )
                if recompute:
                    gradient, loss = checkpoint(self._matching_gradient, *args, use_reentrant=False)
                else:
                    loss = self._matching_loss(*args)
                    gradient = torch.autograd.grad(loss.sum(), candidate, create_graph=differentiable)[0]
                if before is None:
                    before = loss.detach()
                candidate = candidate - step_size * gradient
                if not differentiable:
                    candidate = candidate.detach().requires_grad_(True)
            with torch.no_grad():
                outputs, mass = local_read(
                    queries, *self.decode(candidate, compressed_tokens, rotary)
                )
                after = matching_loss(
                    outputs, mass, teacher_outputs, teacher_mass, scales_o, scales_z
                )
        return candidate if differentiable else candidate.detach(), (before, after)
