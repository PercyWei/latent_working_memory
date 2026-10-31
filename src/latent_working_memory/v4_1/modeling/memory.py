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
        output_loss, mass_loss = matching_loss(
            outputs, mass, teacher_outputs, teacher_mass, scales_o, scales_z
        )
        return output_loss + mass_loss

    def _matching_gradient(self, candidate, *args):
        # 重算求导产生的高阶图；各样本的 inner 步长不随 batch 改变。
        with torch.enable_grad():
            loss = self._matching_loss(candidate, *args)
            gradient = torch.autograd.grad(loss.sum(), candidate, create_graph=True)[0]
        return gradient, loss.detach()

    def _matching_diagnostics(self, candidate, queries, teacher_outputs, teacher_mass,
                              scales_o, scales_z, compressed_tokens, rotary):
        # 独立局部图只求 memory 偏导；不经过 teacher/query 历史或 checkpoint 重算。
        probe = candidate.detach().requires_grad_(True)
        outputs, mass = local_read(
            queries.detach(), *self.decode(probe, compressed_tokens, rotary)
        )
        output_loss, mass_loss, output_heads, mass_heads = matching_loss(
            outputs, mass, teacher_outputs.detach(), teacher_mass.detach(), scales_o, scales_z,
            return_per_head=True,
        )
        output_gradient = torch.autograd.grad(
            output_loss.sum(), probe, retain_graph=True, create_graph=False,
        )[0].flatten(1).double()
        mass_gradient = torch.autograd.grad(
            mass_loss.sum(), probe, create_graph=False,
        )[0].flatten(1).double()
        return (
            output_loss.detach(), mass_loss.detach(), output_heads.detach(), mass_heads.detach(),
            torch.linalg.vector_norm(output_gradient, dim=1),
            torch.linalg.vector_norm(mass_gradient, dim=1),
            (output_gradient * mass_gradient).sum(dim=1),
        )

    def write(self, old_memory, teacher_kv, queries, compressed_tokens, rotary, differentiable,
              diagnostics=None):
        """轮内参照固定；clone 隔离局部偏导，不截断 outer 历史依赖。

        可选 diagnostics 按 batch 行顺序追加逐样本记录。
        """
        step_records = [] if diagnostics is not None else None
        head_records = [] if diagnostics is not None else None
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
                if diagnostics is not None:
                    components = self._matching_diagnostics(*args)
                if recompute:
                    gradient, loss = checkpoint(
                        self._matching_gradient, *args, use_reentrant=False,
                    )
                else:
                    loss = self._matching_loss(*args)
                    gradient = torch.autograd.grad(loss.sum(), candidate, create_graph=differentiable)[0]
                if before is None:
                    before = loss.detach()
                if diagnostics is not None:
                    update = step_size * gradient
                    with torch.no_grad():
                        step_records.append(torch.stack((
                            loss.detach(), components[0], components[1],
                            torch.linalg.vector_norm(gradient.detach().flatten(1), dim=1,
                                                     dtype=torch.float64),
                            torch.linalg.vector_norm(update.detach().flatten(1), dim=1,
                                                     dtype=torch.float64),
                            *components[4:],
                            torch.linalg.vector_norm(candidate.detach().flatten(1), dim=1,
                                                     dtype=torch.float64),
                        ), dim=-1))
                        head_records.append(torch.stack(components[2:4], dim=1))
                    candidate = candidate - update
                else:
                    candidate = candidate - step_size * gradient
                if not differentiable:
                    candidate = candidate.detach().requires_grad_(True)
            with torch.no_grad():
                outputs, mass = local_read(
                    queries, *self.decode(candidate, compressed_tokens, rotary)
                )
                final_components = matching_loss(
                    outputs, mass, teacher_outputs, teacher_mass, scales_o, scales_z,
                    return_per_head=diagnostics is not None,
                )
                output_loss, mass_loss = final_components[:2]
                after = output_loss + mass_loss
        if diagnostics is not None:
            scales_o_rows, scales_z_rows = scales_o.cpu().tolist(), scales_z.cpu().tolist()
            eta = float(step_size.detach())
            step_rows = torch.stack(step_records, dim=1).cpu().tolist()
            head_rows = torch.stack(head_records, dim=1).cpu().tolist()
            after_rows = torch.stack((after, output_loss, mass_loss), dim=-1).cpu().tolist()
            after_heads = torch.stack(final_components[2:], dim=1).cpu().tolist()
            for row, (steps, heads, final) in enumerate(zip(step_rows, head_rows, after_rows)):
                records = []
                for index, (values, head_losses) in enumerate(zip(steps, heads)):
                    output_norm, mass_norm, dot, memory_norm = values[5:]
                    cosine = (max(-1.0, min(1.0, dot / (output_norm * mass_norm)))
                              if output_norm > 0 and mass_norm > 0 else None)
                    records.append({
                        "inner_step": index + 1, "loss": values[0],
                        "output_loss": values[1], "mass_loss": values[2],
                        "output_loss_per_head": head_losses[0],
                        "mass_loss_per_head": head_losses[1],
                        "gradient_norm": values[3], "update_norm": values[4],
                        "output_gradient_norm": output_norm, "mass_gradient_norm": mass_norm,
                        "gradient_cosine_similarity": cosine,
                        "gradient_angle_degrees": (math.degrees(math.acos(cosine))
                                                   if cosine is not None else None),
                        "memory_norm": memory_norm,
                        "relative_update_norm": (values[4] / memory_norm
                                                 if memory_norm > 0 else None),
                    })
                diagnostics.append({
                    "step_size": eta,
                    "scales_o": scales_o_rows[row], "scales_z": scales_z_rows[row],
                    "steps": records,
                    "after": {"inner_step": self.config.inner_steps, "loss": final[0],
                              "output_loss": final[1], "mass_loss": final[2],
                              "output_loss_per_head": after_heads[row][0],
                              "mass_loss_per_head": after_heads[row][1]},
                })
        return candidate if differentiable else candidate.detach(), (before, after)
