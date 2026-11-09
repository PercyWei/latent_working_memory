"""固定组内采样结果，逐路径反传 QA 与停止梯度奖励的策略目标。"""

import torch

from .rollout import execute, qa_nll, sample_group


def _path_backward(model, document, objective, path, advantage, group_size):
    """反传后只返回无图统计，使本路径的中间张量在下一次前向前释放。"""
    blocks, _, log_probs, entropy, _ = execute(
        model, document.input_ids, objective["unit_tokens"], actions=path.actions,
    )
    task_loss = qa_nll(model, blocks, document)
    if len(log_probs):
        ratios = (log_probs - path.old_log_probs).exp()
        clipped = ratios.clamp(1 - objective["clip_epsilon"], 1 + objective["clip_epsilon"])
        policy_loss = -torch.minimum(ratios * advantage, clipped * advantage).mean()
        policy_entropy = entropy.mean()
        drift = (log_probs - path.old_log_probs).abs().max()
    else:
        policy_loss = task_loss.new_zeros(())
        policy_entropy = policy_loss
        drift = policy_loss
    loss = task_loss + objective["policy_weight"] * policy_loss
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"non-finite training loss: {document.id}")
    (loss / group_size).backward()
    return torch.stack((
        task_loss.detach(), policy_loss.detach(), policy_entropy.detach(), drift.detach(),
    ))


def post_backward(model, document, objective, generator):
    """采样完整组后逐路径累积梯度；文档归约与参数更新由 engine 执行。"""
    paths = sample_group(model, document, objective, generator)
    device = paths[0].old_log_probs.device
    rewards = torch.tensor([
        -path.qa_nll - objective["slot_cost"] * model.memory_slots * len(path.block_lengths)
        for path in paths
    ], device=device, dtype=torch.float32)
    std = rewards.std(unbiased=False)
    advantages = ((rewards - rewards.mean()) / (std + objective["advantage_epsilon"])).detach()
    group_size = len(paths)
    values = torch.stack([
        _path_backward(model, document, objective, path, advantage, group_size)
        for path, advantage in zip(paths, advantages, strict=True)
    ])
    qa_loss, policy_loss = values[:, :2].mean(dim=0)
    decisions = sum(len(path.actions) for path in paths)
    blocks = sum(len(path.block_lengths) for path in paths)
    metrics = {
        "qa_nll": (float(qa_loss), 1),
        "policy_loss": (float(policy_loss), 1),
        "reward": (float(rewards.mean()), 1),
        "reward_std": (float(std), 1),
        "zero_advantage_groups": (float(bool(std == 0)), 1),
        "blocks": (blocks, group_size),
        "memory_slots": (blocks * model.memory_slots, group_size),
        "segment_tokens": (len(document.input_ids) * group_size, blocks),
        "commit_rate": (sum(sum(path.actions) for path in paths), decisions),
        "policy_entropy": (
            float(values[:, 2].sum()) * len(paths[0].actions) if decisions else 0,
            decisions,
        ),
        "replay_log_probability_drift": (float(values[:, 3].sum()), group_size),
    }
    return qa_loss + objective["policy_weight"] * policy_loss, metrics
