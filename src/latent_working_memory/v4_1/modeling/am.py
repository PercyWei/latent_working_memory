"""局部 attention matching；显式 FP32 运算支持 inner 更新的高阶梯度。"""

import math

import torch


def local_read(queries, keys, values, bias):
    """q:[N,Hq,D]，k/v:[Hkv,K,D]，bias:[Hkv,K] → o:[N,Hq,D], z:[N,Hq]。"""
    groups = queries.shape[1] // keys.shape[0]
    keys = keys.float().repeat_interleave(groups, dim=0)
    values = values.float().repeat_interleave(groups, dim=0)
    bias = bias.float().repeat_interleave(groups, dim=0)
    scores = torch.einsum("nhd,hkd->nhk", queries.float(), keys)
    scores = scores / math.sqrt(queries.shape[-1]) + bias.unsqueeze(0)
    outputs = torch.einsum("nhk,hkd->nhd", scores.softmax(-1), values)
    return outputs, scores.logsumexp(-1)


def teacher_scales(outputs, log_mass, floor):
    """每个 query head 的固定尺度；统计量不参与元梯度。"""
    scale_o = outputs.float().std(dim=(0, 2), correction=0).clamp_min(floor)
    scale_z = log_mass.float().std(dim=0, correction=0).clamp_min(floor)
    return scale_o.detach(), scale_z.detach()


def matching_loss(outputs, log_mass, teacher_outputs, teacher_log_mass, scales_o, scales_z):
    output_loss = ((outputs - teacher_outputs) / scales_o[None, :, None]).square().mean()
    mass_loss = ((log_mass - teacher_log_mass) / scales_z[None, :]).square().mean()
    return output_loss + mass_loss
