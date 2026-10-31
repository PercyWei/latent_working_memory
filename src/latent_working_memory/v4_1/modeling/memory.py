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
                       scales_o, scales_z, compressed_tokens, rotary, return_components=False):
        outputs, mass = local_read(queries, *self.decode(candidate, compressed_tokens, rotary))
        output_loss, mass_loss = matching_loss(
            outputs, mass, teacher_outputs, teacher_mass, scales_o, scales_z
        )
        loss = output_loss + mass_loss
        if return_components:
            return loss, output_loss.detach(), mass_loss.detach()
        return loss

    def _matching_gradient(self, candidate, *args, return_components=False):
        # 重算求导产生的高阶图；各样本的 inner 步长不随 batch 改变。
        with torch.enable_grad():
            result = self._matching_loss(candidate, *args, return_components=return_components)
            loss = result[0] if return_components else result
            gradient = torch.autograd.grad(loss.sum(), candidate, create_graph=True)[0]
        if return_components:
            return gradient, loss.detach(), result[1], result[2]
        return gradient, loss.detach()

    def write(self, old_memory, teacher_kv, queries, compressed_tokens, rotary, differentiable,
              diagnostics=None):
        """轮内参照固定；clone 隔离局部偏导，不截断 outer 历史依赖。

        可选 diagnostics 按 batch 行顺序追加逐样本记录。
        """
        step_records = [] if diagnostics is not None else None
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
                    result = checkpoint(
                        self._matching_gradient, *args, use_reentrant=False,
                        return_components=diagnostics is not None,
                    )
                    if diagnostics is not None:
                        gradient, loss, output_loss, mass_loss = result
                    else:
                        gradient, loss = result
                else:
                    result = self._matching_loss(*args, return_components=diagnostics is not None)
                    if diagnostics is not None:
                        loss, output_loss, mass_loss = result
                    else:
                        loss = result
                    gradient = torch.autograd.grad(loss.sum(), candidate, create_graph=differentiable)[0]
                if before is None:
                    before = loss.detach()
                if diagnostics is not None:
                    update = step_size * gradient
                    candidate = candidate - update
                    with torch.no_grad():
                        step_records.append(torch.stack((
                            loss.detach(), output_loss, mass_loss,
                            torch.linalg.vector_norm(gradient.detach().flatten(1), dim=1),
                            torch.linalg.vector_norm(update.detach().flatten(1), dim=1),
                        ), dim=-1))
                else:
                    candidate = candidate - step_size * gradient
                if not differentiable:
                    candidate = candidate.detach().requires_grad_(True)
            with torch.no_grad():
                outputs, mass = local_read(
                    queries, *self.decode(candidate, compressed_tokens, rotary)
                )
                output_loss, mass_loss = matching_loss(
                    outputs, mass, teacher_outputs, teacher_mass, scales_o, scales_z
                )
                after = output_loss + mass_loss
        if diagnostics is not None:
            scales_o_rows, scales_z_rows = scales_o.cpu().tolist(), scales_z.cpu().tolist()
            eta = float(step_size.detach())
            step_rows = torch.stack(step_records, dim=1).cpu().tolist()
            after_rows = torch.stack((after, output_loss, mass_loss), dim=-1).cpu().tolist()
            for row, (steps, final) in enumerate(zip(step_rows, after_rows)):
                diagnostics.append({
                    "step_size": eta,
                    "scales_o": scales_o_rows[row], "scales_z": scales_z_rows[row],
                    "steps": [
                        {"inner_step": index + 1, "loss": values[0],
                         "output_loss": values[1], "mass_loss": values[2],
                         "gradient_norm": values[3], "update_norm": values[4]}
                        for index, values in enumerate(steps)
                    ],
                    "after": {"inner_step": self.config.inner_steps, "loss": final[0],
                              "output_loss": final[1], "mass_loss": final[2]},
                })
        return candidate if differentiable else candidate.detach(), (before, after)
