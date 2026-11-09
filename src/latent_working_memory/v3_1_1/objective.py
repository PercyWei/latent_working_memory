"""固定组内采样结果，逐路径反传 QA 与停止梯度奖励的策略目标。"""

import torch

from .rollout import execute, sample_group


def _qa_backward(model, memory_leaves, qa, normalizer):
    """单题只反传到独立 memory 叶子；返回统计时释放本题的 decoder 图。"""
    loss = model.answer_nll(memory_leaves, qa.prompt_ids, qa.target_ids)
    (loss / normalizer).backward()
    return loss.detach()


def _path_backward(model, document, objective, path, advantage, group_size):
    """反传后只返回无图统计，使本路径的中间张量在下一次前向前释放。"""
    blocks, _, log_probs, entropy, _ = execute(
        model, document.input_ids, objective["unit_tokens"], actions=path.actions,
    )
    memory_leaves = tuple(block.detach().requires_grad_(True) for block in blocks)
    question_count = len(document.qas)
    question_losses = [None] * question_count
    # 旧 stack(...).mean().backward() 按后建图的题目先执行，保持同样的累加顺序。
    for index in reversed(range(question_count)):
        question_losses[index] = _qa_backward(
            model, memory_leaves, document.qas[index], question_count * group_size,
        )
    task_loss = torch.stack(question_losses).mean()
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
    outputs = list(blocks)
    gradients = [leaf.grad for leaf in memory_leaves]
    if policy_loss.requires_grad:
        # 零权重也连接策略图，保留原更新中 boundary 的显式零梯度。
        outputs.append(objective["policy_weight"] * policy_loss / group_size)
        gradients.append(None)
    # 同时注入所有直接 QA adjoint 与策略梯度，让共享 history 只反传一次。
    torch.autograd.backward(outputs, gradients)
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
