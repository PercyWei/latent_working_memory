"""同批一次更新：QA 连续梯度与基于停止梯度奖励的组内策略梯度。"""

import torch
from torch.utils.checkpoint import checkpoint

from .rollout import execute, qa_nll, sample_group


def post_loss(model, document, objective, generator):
    paths = sample_group(model, document, objective, generator)
    device = paths[0].old_log_probs.device
    rewards = torch.tensor([
        -path.qa_nll - objective["slot_cost"] * model.memory_slots * len(path.block_lengths)
        for path in paths
    ], device=device, dtype=torch.float32)
    std = rewards.std(unbiased=False)
    advantages = ((rewards - rewards.mean()) / (std + objective["advantage_epsilon"])).detach()
    losses, policies, entropies, drifts = [], [], [], []
    for path, advantage in zip(paths, advantages, strict=True):
        # 默认参数绑定本条路径，避免 backward 重算时读取循环末尾的路径。
        def replay(old_log_probs, advantage, path=path):
            blocks, _, log_probs, entropy, _ = execute(
                model, document.input_ids, objective["unit_tokens"], actions=path.actions,
            )
            task_loss = qa_nll(model, blocks, document)
            if len(log_probs):
                ratios = (log_probs - old_log_probs).exp()
                clipped = ratios.clamp(1 - objective["clip_epsilon"], 1 + objective["clip_epsilon"])
                policy_loss = -torch.minimum(ratios * advantage, clipped * advantage).mean()
                policy_entropy = entropy.mean()
                drift = (log_probs - old_log_probs).abs().max()
            else:
                policy_loss = task_loss.new_zeros(())
                policy_entropy = policy_loss
                drift = policy_loss
            return task_loss, policy_loss, policy_entropy, drift

        task_loss, policy_loss, entropy, drift = checkpoint(
            replay, path.old_log_probs, advantage, use_reentrant=False,
        )
        losses.append(task_loss)
        policies.append(policy_loss)
        entropies.append(entropy.detach())
        drifts.append(drift.detach())
    qa_loss = torch.stack(losses).mean()
    policy_loss = torch.stack(policies).mean()
    group_size = len(paths)
    decisions = sum(len(path.actions) for path in paths)
    blocks = sum(len(path.block_lengths) for path in paths)
    metrics = {
        "qa_nll": (float(qa_loss.detach()), 1),
        "policy_loss": (float(policy_loss.detach()), 1),
        "reward": (float(rewards.mean()), 1),
        "reward_std": (float(std), 1),
        "zero_advantage_groups": (float(bool(std == 0)), 1),
        "blocks": (blocks, group_size),
        "memory_slots": (blocks * model.memory_slots, group_size),
        "segment_tokens": (len(document.input_ids) * group_size, blocks),
        "commit_rate": (sum(sum(path.actions) for path in paths), decisions),
        "policy_entropy": (
            float(torch.stack(entropies).sum()) * len(paths[0].actions) if decisions else 0,
            decisions,
        ),
        "replay_log_probability_drift": (float(torch.stack(drifts).sum()), group_size),
    }
    return qa_loss + objective["policy_weight"] * policy_loss, metrics
