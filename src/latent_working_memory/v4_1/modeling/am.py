"""局部 attention matching；显式 FP32 运算支持 inner 更新的高阶梯度。"""

import math

import torch


def _attention_scores(queries, keys, bias):
    groups = queries.shape[-2] // keys.shape[-3]
    grouped = queries.float().unflatten(-2, (keys.shape[-3], groups))
    scores = torch.einsum("...nhgd,...hkd->...nhgk", grouped, keys.float())
    return scores / math.sqrt(queries.shape[-1]) + bias.float().unsqueeze(-3).unsqueeze(-2)


def local_read(queries, keys, values, bias):
    """q:[...,N,Hq,D]，k/v:[...,Hkv,K,D]，bias:[...,Hkv,K]，前置样本维独立。"""
    scores = _attention_scores(queries, keys, bias)
    outputs = torch.einsum("...nhgk,...hkd->...nhgd", scores.softmax(-1), values.float())
    return outputs.reshape(queries.shape), scores.logsumexp(-1).flatten(-2)


def _teacher_keys_and_bias(memory_key, memory_bias, pending_key):
    keys = torch.cat((memory_key.float(), pending_key.float()), dim=2)
    bias = torch.cat((
        memory_bias.float(), memory_bias.new_zeros((*memory_bias.shape[:-1], pending_key.shape[2])),
    ), dim=-1)
    return keys, bias


def read_teacher(queries, memory_key, memory_value, memory_bias, pending_key, pending_value):
    """在重算边界内拼接 FP32 参照，checkpoint 输入保留原始 KV/query 存储。"""
    keys, bias = _teacher_keys_and_bias(memory_key, memory_bias, pending_key)
    values = torch.cat((memory_value.float(), pending_value.float()), dim=2)
    return local_read(queries, keys, values, bias)


def calibrate_memory_bias(queries, keys, bias, teacher_memory_key, teacher_memory_bias,
                          pending_key, floor):
    """按 KV head 拟合公共 log-mass 偏移；保留 query 与两侧 key/bias 的梯度。"""
    teacher_keys, teacher_bias = _teacher_keys_and_bias(
        teacher_memory_key, teacher_memory_bias, pending_key,
    )
    teacher_mass = _attention_scores(queries, teacher_keys, teacher_bias).logsumexp(-1)
    memory_mass = _attention_scores(queries, keys, bias).logsumexp(-1)
    scales = teacher_mass.std(dim=-3, correction=0).clamp_min(floor).detach()
    weights = scales.double().square().reciprocal()
    gap = (teacher_mass.double() - memory_mass.double()).mean(dim=-3)
    offset = (weights * gap).sum(dim=-1) / weights.sum(dim=-1)
    return bias + offset.to(bias.dtype).unsqueeze(-1)


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
