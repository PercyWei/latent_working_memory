"""冻结压缩器的 PPO：全批原生 GAE，独立 actor／critic 损失与固定目标。"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from verl.trainer.ppo.core_algos import compute_value_loss
from verl.workers.config import CriticConfig

from ..data import Document
from ..rollout import Rollout, sample_paths_batch
from .advantages import effective_gamma, generalized_advantages
from .rl import boundary_policy_loss, path_rewards, summarize_paths


@dataclass(frozen=True)
class PPORolloutGroup:
    document: Document
    paths: tuple[Rollout, ...]
    rewards: torch.Tensor
    old_values: torch.Tensor


@dataclass(frozen=True)
class PPOGroup(PPORolloutGroup):
    advantages: torch.Tensor
    value_targets: torch.Tensor


@torch.no_grad()
def collect_rollouts(model, documents, pending_stride, objective, generator, temperature,
                     pending_size, recent_size):
    """只收集固定旧策略的数据；优势在全部累积微批采样完成后计算。"""
    with model.forward_context():
        groups = sample_paths_batch(
            model, documents, pending_stride, objective["group_size"], generator,
            temperature=temperature, pending_size=pending_size, recent_size=recent_size,
            collect_features=True,
        )
        batches = []
        for document, paths in zip(documents, groups, strict=True):
            rewards = torch.stack([
                path_rewards(path, len(document.input_ids), objective, pending_stride).float() for path in paths
            ])
            old_values = model.value_head(torch.stack([path.event_features for path in paths])).float()
            batches.append(PPORolloutGroup(document, tuple(paths), rewards, old_values))
    return batches


@torch.no_grad()
def prepare_advantages(batches, objective, device):
    """整批真实事件共同白化；DP 仅交换奖励和旧价值，不传输模型特征。"""
    local = [(batch.rewards.cpu(), batch.old_values.cpu()) for batch in batches]
    if dist.is_initialized():
        shards = [None] * dist.get_world_size()
        dist.all_gather_object(shards, local)
        rank = dist.get_rank()
    else:
        shards, rank = [local], 0
    groups = [group for shard in shards for group in shard]
    reward_rows = [row.to(device) for rewards, _ in groups for row in rewards]
    value_rows = [row.to(device) for _, values in groups for row in values]
    rewards = pad_sequence(reward_rows, batch_first=True)
    old_values = pad_sequence(value_rows, batch_first=True)
    lengths = torch.tensor([row.numel() for row in reward_rows], device=device)
    event_mask = (torch.arange(rewards.shape[-1], device=device)[None, :] < lengths[:, None]).float()
    advantages, targets = generalized_advantages(
        rewards, old_values, event_mask, effective_gamma(objective), objective["gae_lambda"],
    )
    offset = sum(len(rewards) for shard in shards[:rank] for rewards, _ in shard)
    prepared = []
    for batch in batches:
        count, width = batch.rewards.shape
        prepared.append(PPOGroup(
            batch.document, batch.paths, batch.rewards, batch.old_values,
            advantages[offset:offset + count, :width].clone(),
            targets[offset:offset + count, :width].clone(),
        ))
        offset += count
    return prepared


def backward_critic_group(model, batch, objective, pending_stride):
    """只反传 value head；保留 verl 默认 value clipping 与已有损失权重。"""
    with model.forward_context():
        values = model.value_head(torch.stack([path.event_features for path in batch.paths])).float()
        value_loss, clip_fraction = compute_value_loss(
            vpreds=values,
            returns=batch.value_targets,
            values=batch.old_values,
            response_mask=torch.ones_like(values),
            cliprange_value=CriticConfig.cliprange_value,
            loss_agg_mode="seq-mean-token-mean",
        )
        loss = objective["value_loss_weight"] * value_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite PPO critic loss: {batch.document.id}")
    loss.backward()
    rows = values.new_zeros((len(batch.paths), 4))
    rows[:, 0] = values.new_tensor([path.qa_nll for path in batch.paths])
    _, metrics = summarize_paths(
        batch.document, batch.paths, batch.rewards.sum(dim=-1), batch.advantages[:, 1:-1],
        rows, objective, model.memory_slots, pending_stride, "boundary_only",
    )
    metrics.update({
        "value_loss": (float(value_loss.detach()), 1),
        "value_clip_fraction": (float(clip_fraction), 1),
    })
    return loss.detach(), metrics


def backward_actor_group(model, batch, objective, pending_stride, temperature):
    """只反传 boundary head；旧概率和全批白化优势在内轮更新期间固定。"""
    rows, clip_counts, dual_clip_counts, divergences, native_kl = [], [], [], [], []
    with model.forward_context():
        for index, path in enumerate(batch.paths):
            features = path.event_features
            mask = torch.tensor(path.decision_mask, device=features.device, dtype=torch.bool)
            if bool(mask.any()):
                logits = model.boundary(features[1:-1][mask]) / temperature
                actions = torch.tensor(path.actions, device=logits.device, dtype=torch.bool)[mask]
                log_probs = F.logsigmoid(torch.where(actions, logits, -logits)).float()
                policy_loss, policy_metrics = boundary_policy_loss(
                    log_probs, path.old_log_probs[mask], batch.advantages[index, 1:-1][mask], objective,
                )
                entropy = (F.softplus(logits) - logits * logits.sigmoid()).float().mean()
                log_ratio = log_probs - path.old_log_probs[mask].float()
                drift = log_ratio.abs().max()
                count = int(mask.sum())
                clip_counts.append(float(policy_metrics["actor/pg_clipfrac"]) * count)
                dual_clip_counts.append(float(policy_metrics["actor/pg_clipfrac_lower"]) * count)
                divergences.append(float((log_ratio.exp() - 1 - log_ratio).detach().sum()))
                native_kl.append(float(policy_metrics["actor/ppo_kl"]) * count)
            else:
                policy_loss = entropy = drift = features.new_zeros((), dtype=torch.float32)
            rows.append(torch.stack((policy_loss.new_tensor(path.qa_nll), policy_loss, entropy, drift)))
        rows = torch.stack(rows)
        loss = objective["policy_weight"] * rows[:, 1].mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite PPO actor loss: {batch.document.id}")
    if loss.requires_grad:
        loss.backward()
    _, metrics = summarize_paths(
        batch.document, batch.paths, batch.rewards.sum(dim=-1), batch.advantages[:, 1:-1],
        rows.detach(), objective, model.memory_slots, pending_stride, "boundary_only",
    )
    decisions = sum(sum(path.decision_mask) for path in batch.paths)
    metrics.update({
        "clip_fraction": (sum(clip_counts), decisions),
        "dual_clip_fraction": (sum(dual_clip_counts), decisions),
        "approx_kl": (sum(divergences), decisions),
        "ppo_kl": (sum(native_kl), decisions),
    })
    return loss.detach(), metrics
