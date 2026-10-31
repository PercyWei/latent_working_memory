"""局部 attention matching；显式 FP32 运算支持 inner 更新的高阶梯度。"""

import math

import torch


def local_read(queries, keys, values, bias):
    """q:[...,N,Hq,D]，k/v:[...,Hkv,K,D]，bias:[...,Hkv,K]，前置样本维独立。"""
    groups = queries.shape[-2] // keys.shape[-3]
    grouped = queries.float().unflatten(-2, (keys.shape[-3], groups))
    scores = torch.einsum("...nhgd,...hkd->...nhgk", grouped, keys.float())
    scores = scores / math.sqrt(queries.shape[-1]) + bias.float().unsqueeze(-3).unsqueeze(-2)
    outputs = torch.einsum("...nhgk,...hkd->...nhgd", scores.softmax(-1), values.float())
    return outputs.reshape(queries.shape), scores.logsumexp(-1).flatten(-2)


def teacher_scales(outputs, log_mass, floor):
    """每个 query head 的固定尺度；统计量不参与元梯度。"""
    scale_o = outputs.float().std(dim=(-3, -1), correction=0).clamp_min(floor)
    scale_z = log_mass.float().std(dim=-2, correction=0).clamp_min(floor)
    return scale_o.detach(), scale_z.detach()


def matching_loss(outputs, log_mass, teacher_outputs, teacher_log_mass, scales_o, scales_z,
                  return_per_head=False):
    """返回逐样本的标准化 MSE；可额外返回逐 query-head 分项，原损失归约不变。"""
    output_error = ((outputs - teacher_outputs) / scales_o[..., None, :, None]).square()
    mass_error = ((log_mass - teacher_log_mass) / scales_z[..., None, :]).square()
    output_loss = output_error.mean(dim=(-3, -2, -1))
    mass_loss = mass_error.mean(dim=(-2, -1))
    if return_per_head:
        return (output_loss, mass_loss, output_error.mean(dim=(-3, -1)),
                mass_error.mean(dim=-2))
    return output_loss, mass_loss
