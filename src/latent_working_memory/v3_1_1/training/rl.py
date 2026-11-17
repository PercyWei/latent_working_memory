"""按文档 microbatch 采样与回放，逐题释放 Reader 图并累积联合目标梯度。"""

import torch

from ..rollout import execute_replay_batch, sample_paths_batch


def _paths_backward_batch(
    model, documents, unit_tokens, rl, paths, advantages, num_samples, bptt_steps, temperature,
):
    """各文档并行回放一条路径；所有 memory adjoint 与策略项一次回传。"""
    with model.forward_context():
        replays = execute_replay_batch(
            model, [document.input_ids for document in documents],
            unit_tokens, [path.actions for path in paths],
            bptt_steps=bptt_steps, temperature=temperature,
        )
    memories = [tuple(block.detach().requires_grad_(True) for block in replay[0])
                for replay in replays]
    question_losses = [[None] * len(document.qas) for document in documents]
    # 每轮每篇只处理一道 QA；各篇仍按原来的逆序累积 memory 梯度。
    for offset in range(max(len(document.qas) for document in documents)):
        active = [index for index, document in enumerate(documents) if offset < len(document.qas)]
        indices = [len(documents[index].qas) - offset - 1 for index in active]
        questions = [documents[index].qas[question] for index, question in zip(active, indices, strict=True)]
        with model.forward_context():
            losses = model.answer_nll_batch(
                [memories[index] for index in active],
                [question.prompt_ids for question in questions],
                [question.target_ids for question in questions],
            )
            normalizers = losses.new_tensor([len(documents[index].qas) * num_samples for index in active])
            loss = (losses / normalizers).sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite QA training loss")
        loss.backward()
        for index, question, value in zip(active, indices, losses.detach(), strict=True):
            question_losses[index][question] = value
        del losses, loss

    outputs, gradients, values = [], [], []
    with model.forward_context():
        for document, replay, leaves, questions, path, advantage in zip(
            documents, replays, memories, question_losses, paths, advantages, strict=True,
        ):
            blocks, _, log_probs, entropy, _ = replay
            task_loss = torch.stack(questions).mean()
            if len(log_probs):
                ratios = (log_probs - path.old_log_probs).exp()
                clipped = ratios.clamp(1 - rl["clip_epsilon"], 1 + rl["clip_epsilon"])
                policy_loss = -torch.minimum(ratios * advantage, clipped * advantage).mean()
                policy_entropy = entropy.mean()
                drift = (log_probs - path.old_log_probs).abs().max()
            else:
                policy_loss = task_loss.new_zeros(())
                policy_entropy = drift = policy_loss
            loss = task_loss + rl["policy_weight"] * policy_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite training loss: {document.id}")
            outputs.extend(blocks)
            gradients.extend(leaf.grad for leaf in leaves)
            if policy_loss.requires_grad:
                # 零权重仍连接策略图，保持 boundary 的显式零梯度语义。
                outputs.append(rl["policy_weight"] * policy_loss / num_samples)
                gradients.append(None)
            values.append(torch.stack((
                task_loss.detach(), policy_loss.detach(), policy_entropy.detach(), drift.detach(),
            )))
    torch.autograd.backward(outputs, gradients)
    return torch.stack(values)


def post_backward_batch(model, documents, unit_tokens, rl, generator, bptt_steps=None, temperature=1.0):
    """每篇按 GRPO 或 SCST 构造优势，随机路径的 QA 与策略损失按采样数平均。"""
    self_critical = rl["strategy"] == "scst"
    num_samples = rl["num_samples"] if self_critical else rl["group_size"]
    score_cache = [{} for _ in documents] if self_critical else None
    # 采样和带图回放分开进入 autocast，避免复用无梯度的权重转换缓存。
    with torch.no_grad(), model.forward_context():
        groups = sample_paths_batch(
            model, documents, unit_tokens, num_samples, generator,
            temperature=temperature, score_cache=score_cache,
        )
        if self_critical:
            # 确定性评分不消耗采样随机流，同动作与随机路径共用评分。
            greedy = sample_paths_batch(
                model, documents, unit_tokens, 1, None,
                temperature=temperature, score_cache=score_cache,
            )
    rewards, advantages, baselines = [], [], []
    for index, (document, paths) in enumerate(zip(documents, groups, strict=True)):
        scored_paths = (*paths, greedy[index][0]) if self_critical else paths
        reward = torch.tensor([
            -path.qa_nll - rl["memory_cost_weight"] * (
                model.memory_slots * len(path.block_lengths) / len(document.input_ids)
            ) ** rl["memory_cost_power"]
            for path in scored_paths
        ], device=paths[0].old_log_probs.device, dtype=torch.float32)
        if self_critical:
            baselines.append(reward[-1])
            reward = reward[:-1]
            advantage = reward - baselines[-1]
        else:
            advantage = (reward - reward.mean()) / (
                reward.std(unbiased=False) + rl["advantage_epsilon"]
            )
        rewards.append(reward)
        advantages.append(advantage.detach())
    values = torch.stack([
        _paths_backward_batch(
            model, documents, unit_tokens, rl, [paths[index] for paths in groups],
            [advantage[index] for advantage in advantages], num_samples, bptt_steps, temperature,
        ) for index in range(num_samples)
    ]).mean(dim=0)
    results = []
    for index, (paths, reward, advantage, value) in enumerate(zip(
        groups, rewards, advantages, values, strict=True,
    )):
        qa_loss, policy_loss, entropy, drift = value
        decisions = sum(len(path.actions) for path in paths)
        blocks = sum(len(path.block_lengths) for path in paths)
        metrics = {
            "qa_nll": (float(qa_loss), 1),
            "policy_loss": (float(policy_loss), 1),
            "reward": (float(reward.mean()), 1),
            "reward_std": (float(reward.std(unbiased=False)), 1),
            "zero_advantage_rate": (float(bool((advantage == 0).all())), 1),
            "blocks": (blocks, num_samples),
            "memory_slots": (blocks * model.memory_slots, num_samples),
            "segment_tokens": (len(documents[index].input_ids) * num_samples, blocks),
            "commit_rate": (sum(sum(path.actions) for path in paths), decisions),
            "policy_entropy": (float(entropy) * decisions, decisions),
            "replay_log_probability_drift": (float(drift) * num_samples, num_samples),
        }
        if self_critical:
            metrics["baseline_reward"] = (float(baselines[index]), 1)
        results.append((qa_loss + rl["policy_weight"] * policy_loss, metrics))
    return results
