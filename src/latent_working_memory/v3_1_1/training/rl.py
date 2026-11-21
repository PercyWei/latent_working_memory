"""联合更新或仅边界更新；终局／逐步奖励在完整轨迹之后结算。"""

from dataclasses import dataclass

import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.core_algos import get_policy_loss_fn

from ..data import Document
from ..rollout import Rollout, execute_replay_batch, sample_paths_batch
from .advantages import relative_advantages


@dataclass(frozen=True)
class RLGroup:
    document: Document
    paths: tuple[Rollout, ...]
    rewards: torch.Tensor
    advantages: torch.Tensor
    baseline: torch.Tensor | None


def boundary_policy_loss(log_probs, old_log_probs, advantages, objective):
    """原生 vanilla loss 对自主动作求和；调用方按全部采样路径和文档平均。"""
    config = OmegaConf.create({
        "clip_ratio": objective["clip_epsilon"],
        "clip_ratio_low": None,
        "clip_ratio_high": None,
        "global_batch_info": {"global_batch_size": 1},
    })
    return get_policy_loss_fn("vanilla")(
        old_log_prob=old_log_probs.float().unsqueeze(0),
        log_prob=log_probs.float().unsqueeze(0),
        advantages=advantages.float().expand_as(log_probs).unsqueeze(0),
        response_mask=torch.ones_like(log_probs).unsqueeze(0),
        loss_agg_mode="seq-mean-token-sum",
        config=config,
    )


def path_rewards(path, document_length, rl, pending_stride):
    """事后按待压缩前缀结算各事件的奖励；固定 recent 不参与成本。"""
    rewards = torch.zeros(len(path.prefix_retained_counts) - 1,
                          device=path.old_log_probs.device, dtype=torch.float32)
    weight, power = rl["memory_cost_weight"], rl["memory_cost_power"]
    prefix_length = document_length - len(path.recent_ids)
    if prefix_length == 0:
        rewards[-1] = -path.qa_nll
        return rewards
    if rl["reward_mode"] == "stepwise":
        retained = torch.tensor(path.prefix_retained_counts, device=rewards.device, dtype=torch.float64)
        if rl["stepwise_cost_basis"] == "full":
            # 账本依次为初始值、固定步长的各次读入、EOF；最后两项的未读长度均为 0。
            read_tokens = torch.arange(retained.numel(), device=rewards.device) * pending_stride
            retained += (prefix_length - read_tokens).clamp_min(0)
        potential = (retained / prefix_length).pow(power)
        rewards = (-weight * potential.diff()).to(rewards.dtype)
        rewards[-1] -= path.qa_nll
    else:
        rewards[-1] = -path.qa_nll - weight * (path.prefix_retained_counts[-1] / prefix_length) ** power
    return rewards


def _paths_backward_batch(
    model, documents, pending_stride, rl, paths, advantages, group_size, bptt_steps, temperature,
    pending_size, recent_size, update_mode,
):
    """每篇回放一条路径；仅联合更新保留 QA→memory→Writer 的梯度。"""
    with model.forward_context():
        replays = execute_replay_batch(
            model, [document.input_ids for document in documents],
            pending_stride, [path.actions for path in paths],
            bptt_steps=bptt_steps, temperature=temperature,
            pending_size=pending_size, recent_size=recent_size,
        )
    joint = update_mode == "joint"
    memories = ([tuple(block.detach().requires_grad_(True) for block in replay.blocks)
                 for replay in replays] if joint else None)
    question_losses = [[None] * len(document.qas) for document in documents]
    if joint:
        # 逐题释放 Reader 图；文档／路径归约保持原有联合更新的权重。
        for offset in range(max(len(document.qas) for document in documents)):
            active = [index for index, document in enumerate(documents) if offset < len(document.qas)]
            indices = [len(documents[index].qas) - offset - 1 for index in active]
            questions = [documents[index].qas[question] for index, question in zip(active, indices, strict=True)]
            with model.forward_context():
                losses = model.answer_nll_batch(
                    [memories[index] for index in active],
                    [(*replays[index].retained_ids, *question.prompt_ids)
                     for index, question in zip(active, questions, strict=True)],
                    [question.target_ids for question in questions],
                )
                normalizers = losses.new_tensor([len(documents[index].qas) * group_size for index in active])
                loss = (losses / normalizers).sum()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite QA training loss")
            if loss.requires_grad:
                loss.backward()
            for index, question, value in zip(active, indices, losses.detach(), strict=True):
                question_losses[index][question] = value
            del losses, loss

    outputs, gradients, values = [], [], []
    with model.forward_context():
        for index, (document, replay, path, advantage) in enumerate(zip(
            documents, replays, paths, advantages, strict=True,
        )):
            task_loss = (torch.stack(question_losses[index]).mean() if joint
                         else path.old_log_probs.new_tensor(path.qa_nll))
            allowed = torch.tensor(replay.decision_mask, device=replay.log_probs.device, dtype=torch.bool)
            if bool(allowed.any()):
                log_probs = replay.log_probs[allowed]
                selected_advantage = advantage[allowed] if advantage.ndim else advantage
                policy_loss, _ = boundary_policy_loss(
                    log_probs, path.old_log_probs[allowed], selected_advantage, rl,
                )
                policy_entropy = replay.entropy[allowed].mean()
                drift = (log_probs - path.old_log_probs[allowed]).abs().max()
            else:
                policy_loss = task_loss.new_zeros(())
                policy_entropy = drift = policy_loss
            loss = (task_loss if joint else 0.) + rl["policy_weight"] * policy_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite training loss: {document.id}")
            if joint:
                outputs.extend(replay.blocks)
                gradients.extend(leaf.grad for leaf in memories[index])
            if policy_loss.requires_grad:
                outputs.append(rl["policy_weight"] * policy_loss / group_size)
                gradients.append(None)
            values.append(torch.stack((task_loss.detach(), policy_loss.detach(),
                                       policy_entropy.detach(), drift.detach())))
    if outputs:
        torch.autograd.backward(outputs, gradients)
    return torch.stack(values)


@torch.no_grad()
def collect_relative_rollouts(model, documents, pending_stride, rl, generator, temperature=1.0,
                             pending_size=None, recent_size=0):
    """同一旧策略完成采样和 greedy 基线，返回可由原生 trainer 调度的轨迹。"""
    pending_size = model.memory_slots if pending_size is None else pending_size
    self_critical = rl["strategy"] == "scst"
    group_size = rl["group_size"]
    score_cache = [{} for _ in documents] if self_critical else None
    with torch.no_grad(), model.forward_context():
        groups = sample_paths_batch(
            model, documents, pending_stride, group_size, generator,
            temperature=temperature, score_cache=score_cache, pending_size=pending_size, recent_size=recent_size,
        )
        if self_critical:
            greedy = sample_paths_batch(
                model, documents, pending_stride, 1, None,
                temperature=temperature, score_cache=score_cache, pending_size=pending_size, recent_size=recent_size,
            )
    batches = []
    for index, (document, paths) in enumerate(zip(documents, groups, strict=True)):
        scored_paths = (*paths, greedy[index][0]) if self_critical else paths
        events = torch.stack([
            path_rewards(path, len(document.input_ids), rl, pending_stride) for path in scored_paths
        ])
        reward = events.sum(dim=1)
        advantage = relative_advantages(events, rl)
        batches.append(RLGroup(
            document, tuple(paths), reward[:group_size], advantage.detach(),
            reward[-1] if self_critical else None,
        ))
    return batches


def backward_relative_batch(model, groups, pending_stride, rl, bptt_steps=None, temperature=1.0,
                            pending_size=None, recent_size=0, update_mode="joint"):
    """回放已固定的 SCST／GRPO 轨迹；不重新采样或改变优势。"""
    pending_size = model.memory_slots if pending_size is None else pending_size
    documents = [group.document for group in groups]
    group_size = rl["group_size"]
    path_values = torch.stack([
        _paths_backward_batch(
            model, documents, pending_stride, rl, [group.paths[index] for group in groups],
            [group.advantages[index] for group in groups], group_size, bptt_steps, temperature,
            pending_size, recent_size, update_mode,
        ) for index in range(group_size)
    ])
    return [summarize_paths(
        group.document, group.paths, group.rewards, group.advantages, path_values[:, index], rl,
        model.memory_slots, pending_stride, update_mode,
        group.baseline,
    ) for index, group in enumerate(groups)]


def post_backward_batch(model, documents, pending_stride, rl, generator, bptt_steps=None, temperature=1.0,
                        pending_size=None, recent_size=0, update_mode="joint"):
    """直接批处理入口，与 trainer 分开的采样／更新调用共用同一实现。"""
    groups = collect_relative_rollouts(
        model, documents, pending_stride, rl, generator, temperature, pending_size, recent_size,
    )
    return backward_relative_batch(
        model, groups, pending_stride, rl, bptt_steps, temperature, pending_size, recent_size, update_mode,
    )


def summarize_paths(document, paths, reward, advantage, values, rl, memory_slots,
                    pending_stride, update_mode, baseline=None):
    """SCST、GRPO 与 PPO 共用路径统计；actor 指标仅统计自主动作。"""
    group_size = len(paths)
    qa_loss, policy_loss, _, drift = values.mean(dim=0)
    counts = [sum(path.decision_mask) for path in paths]
    decisions = sum(counts)
    blocks = sum(len(path.block_lengths) for path in paths)
    if advantage.ndim == 2:
        mask = torch.tensor([path.decision_mask for path in paths], device=advantage.device, dtype=torch.bool)
        selected_advantage = advantage[mask]
    else:
        selected_advantage = advantage
    entropy = sum(float(values[path_index, 2]) * count
                  for path_index, count in enumerate(counts))
    metrics = {
        "qa_nll": (float(qa_loss), 1),
        "policy_loss": (float(policy_loss), 1),
        "reward": (float(reward.mean()), 1),
        "reward_std": (float(reward.std(unbiased=False)), 1),
        "zero_advantage_rate": (float(bool((selected_advantage == 0).all())), 1),
        "memory_blocks": (blocks, group_size),
        "memory_slots": (blocks * memory_slots, group_size),
        "segment_tokens": (sum(sum(path.block_lengths) for path in paths), blocks),
        "commit_rate": (sum(sum(path.actions) for path in paths), decisions),
        "policy_entropy": (entropy, decisions),
        "replay_log_probability_drift": (float(drift) * group_size, group_size),
    }
    length = len(document.input_ids)
    ratios = [(path.prefix_retained_counts[-1] + len(path.recent_ids)) / length
              for path in paths]
    prefix_ratios = [
        path.prefix_retained_counts[-1] / (length - len(path.recent_ids))
        if length > len(path.recent_ids) else 0.0 for path in paths
    ]
    peaks = []
    for path in paths:
        committed = 0
        recent = len(path.recent_ids)
        prefix_length = length - recent
        peak = path.prefix_retained_counts[1]
        for position, action in enumerate(path.actions):
            arriving = min(pending_stride, prefix_length - (position + 1) * pending_stride)
            before_decision = path.prefix_retained_counts[position + 1] - committed * memory_slots
            peak = max(peak, before_decision + arriving)
            committed += action
        peaks.append(peak)
    metrics.update({
        "recent_tokens": (sum(len(path.recent_ids) for path in paths), group_size),
        "pending_tokens": (sum(len(path.pending_ids) for path in paths), group_size),
        "retention_ratio": (sum(ratios), group_size),
        "compression_retention_ratio": (sum(prefix_ratios), group_size),
        "memory_cost": (sum(ratio ** rl["memory_cost_power"] for ratio in prefix_ratios), group_size),
        "pending_peak": (sum(peaks), group_size),
        "advantage_abs_mean": (float(selected_advantage.abs().sum()), selected_advantage.numel()),
    })
    if baseline is not None:
        metrics["baseline_reward"] = (float(baseline), 1)
    loss = (qa_loss if update_mode == "joint" else qa_loss.new_zeros(())) + rl["policy_weight"] * policy_loss
    return loss, metrics
