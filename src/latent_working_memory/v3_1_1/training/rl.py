"""联合更新或仅边界更新；终局／占用差分奖励在完整轨迹之后结算。"""

import torch

from ..rollout import execute_replay_batch, sample_paths_batch


def path_rewards(path, document_length, rl):
    """事后按待压缩前缀结算各事件的奖励；固定 recent 不参与成本。"""
    rewards = path.old_log_probs.new_zeros(len(path.prefix_retained_counts) - 1)
    weight, power = rl["memory_cost_weight"], rl["memory_cost_power"]
    prefix_length = document_length - len(path.recent_ids)
    if prefix_length == 0:
        rewards[-1] = -path.qa_nll
        return rewards
    if rl["reward_mode"] == "retention_delta":
        retained = torch.tensor(path.prefix_retained_counts, device=rewards.device, dtype=torch.float64)
        potential = (retained / prefix_length).pow(power)
        rewards = (-weight * potential.diff()).to(rewards.dtype)
        rewards[-1] -= path.qa_nll
    else:
        rewards[-1] = -path.qa_nll - weight * (path.prefix_retained_counts[-1] / prefix_length) ** power
    return rewards


def _paths_backward_batch(
    model, documents, pending_stride, rl, paths, advantages, num_samples, bptt_steps, temperature,
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
                normalizers = losses.new_tensor([len(documents[index].qas) * num_samples for index in active])
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
                ratios = (log_probs - path.old_log_probs[allowed]).exp()
                clipped = ratios.clamp(1 - rl["clip_epsilon"], 1 + rl["clip_epsilon"])
                selected_advantage = advantage[allowed] if advantage.ndim else advantage
                terms = -torch.minimum(ratios * selected_advantage, clipped * selected_advantage)
                policy_loss = terms.sum() if rl["advantage_mode"] == "reward_to_go" else terms.mean()
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
                outputs.append(rl["policy_weight"] * policy_loss / num_samples)
                gradients.append(None)
            values.append(torch.stack((task_loss.detach(), policy_loss.detach(),
                                       policy_entropy.detach(), drift.detach())))
    if outputs:
        torch.autograd.backward(outputs, gradients)
    return torch.stack(values)


def post_backward_batch(model, documents, pending_stride, rl, generator, bptt_steps=None, temperature=1.0,
                        pending_size=None, recent_size=0, update_mode="joint"):
    """按文档采样并更新，逐位置回报按正文读取位置对齐。"""
    pending_size = model.memory_slots if pending_size is None else pending_size
    self_critical = rl["strategy"] == "scst"
    reward_to_go = rl["advantage_mode"] == "reward_to_go"
    num_samples = rl["num_samples"] if self_critical else rl["group_size"]
    score_cache = [{} for _ in documents] if self_critical else None
    with torch.no_grad(), model.forward_context():
        groups = sample_paths_batch(
            model, documents, pending_stride, num_samples, generator,
            temperature=temperature, score_cache=score_cache, pending_size=pending_size, recent_size=recent_size,
        )
        if self_critical:
            greedy = sample_paths_batch(
                model, documents, pending_stride, 1, None,
                temperature=temperature, score_cache=score_cache, pending_size=pending_size, recent_size=recent_size,
            )
    rewards, advantages, baselines = [], [], []
    for index, (document, paths) in enumerate(zip(documents, groups, strict=True)):
        scored_paths = (*paths, greedy[index][0]) if self_critical else paths
        events = torch.stack([path_rewards(path, len(document.input_ids), rl) for path in scored_paths])
        reward = events.sum(dim=1)
        if reward_to_go:
            # event 0 是首次读入，event -1 是收尾；每个中间事件对应一个固定判断位置。
            returns = events.flip(1).cumsum(dim=1).flip(1)[:, 1:-1]
        else:
            returns = reward
        if self_critical:
            # 最后一行是整篇共用的 greedy 路径，按同一范围比较整轨或逐位置回报。
            advantage = returns[:-1] - returns[-1]
            baselines.append(reward[-1])
        elif reward_to_go:
            baseline = (returns.sum(dim=0, keepdim=True) - returns) / (num_samples - 1)
            advantage = returns - baseline
        else:
            advantage = (reward - reward.mean()) / (
                reward.std(unbiased=False) + rl["advantage_epsilon"]
            )
        rewards.append(reward[:num_samples])
        advantages.append(advantage.detach())
    path_values = torch.stack([
        _paths_backward_batch(
            model, documents, pending_stride, rl, [paths[index] for paths in groups],
            [advantage[index] for advantage in advantages], num_samples, bptt_steps, temperature,
            pending_size, recent_size, update_mode,
        ) for index in range(num_samples)
    ])
    values = path_values.mean(dim=0)
    results = []
    for index, (paths, reward, advantage, value) in enumerate(zip(
        groups, rewards, advantages, values, strict=True,
    )):
        qa_loss, policy_loss, _, drift = value
        counts = [sum(path.decision_mask) for path in paths]
        decisions = sum(counts)
        blocks = sum(len(path.block_lengths) for path in paths)
        if advantage.ndim == 2:
            mask = torch.tensor([path.decision_mask for path in paths], device=advantage.device, dtype=torch.bool)
            selected_advantage = advantage[mask]
        else:
            selected_advantage = advantage
        entropy = sum(float(path_values[path_index, index, 2]) * count
                      for path_index, count in enumerate(counts))
        metrics = {
            "qa_nll": (float(qa_loss), 1),
            "policy_loss": (float(policy_loss), 1),
            "reward": (float(reward.mean()), 1),
            "reward_std": (float(reward.std(unbiased=False)), 1),
            "zero_advantage_rate": (float(bool((selected_advantage == 0).all())), 1),
            "blocks": (blocks, num_samples),
            "memory_slots": (blocks * model.memory_slots, num_samples),
            "segment_tokens": (sum(sum(path.block_lengths) for path in paths), blocks),
            "commit_rate": (sum(sum(path.actions) for path in paths), decisions),
            "policy_entropy": (entropy, decisions),
            "replay_log_probability_drift": (float(drift) * num_samples, num_samples),
        }
        length = len(documents[index].input_ids)
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
                before_decision = path.prefix_retained_counts[position + 1] - committed * model.memory_slots
                peak = max(peak, before_decision + arriving)
                committed += action
            peaks.append(peak)
        metrics.update({
            "recent_tokens": (sum(len(path.recent_ids) for path in paths), num_samples),
            "pending_tokens": (sum(len(path.pending_ids) for path in paths), num_samples),
            "retention_ratio": (sum(ratios), num_samples),
            "prefix_retention_ratio": (sum(prefix_ratios), num_samples),
            "memory_cost": (sum(ratio ** rl["memory_cost_power"] for ratio in prefix_ratios), num_samples),
            "pending_peak": (sum(peaks), num_samples),
            "advantage_abs_mean": (float(selected_advantage.abs().sum()), selected_advantage.numel()),
        })
        if self_critical:
            metrics["baseline_reward"] = (float(baselines[index]), 1)
        loss = (qa_loss if update_mode == "joint" else qa_loss.new_zeros(())) + rl["policy_weight"] * policy_loss
        results.append((loss, metrics))
    return results
