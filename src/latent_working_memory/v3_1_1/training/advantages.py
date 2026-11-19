"""复用 verl 优势计算；逐事件奖励保留任务专用的两种适配。"""

import numpy as np
import torch
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_remax_outcome_advantage,
    register_adv_est,
)
from verl.utils.torch_functional import masked_mean, masked_var


def effective_gamma(objective):
    return 1.0 if objective["reward_mode"] == "terminal" else objective["gamma"]


def discounted_returns(rewards, gamma):
    """最后一维为真实事件，不压缩强制事件之间的时间间隔。"""
    returns = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for position in range(rewards.shape[-1] - 1, -1, -1):
        running = rewards[..., position] + gamma * running
        returns[..., position] = running
    return returns


@torch.no_grad()
def relative_advantages(events, objective):
    """同一正文的无 padding 奖励；SCST 的最后一行为独立 greedy。"""
    terminal = objective["reward_mode"] == "terminal"
    if objective["strategy"] == "scst":
        if terminal:
            advantages, _ = compute_remax_outcome_advantage(
                token_level_rewards=events[:-1],
                reward_baselines=events[-1].sum().expand(len(events) - 1),
                response_mask=torch.ones_like(events[:-1]),
            )
            return advantages[:, 0]
        baseline = discounted_returns(events[-1], effective_gamma(objective))
        advantages, _ = scst_stepwise_advantages(
            events[:-1], baseline, torch.ones_like(events[:-1]), objective,
        )
        return advantages[:, 1:-1]
    if terminal:
        advantages, _ = compute_grpo_outcome_advantage(
            token_level_rewards=events,
            response_mask=torch.ones_like(events),
            index=np.zeros(len(events), dtype=np.int64),
            epsilon=objective["advantage_epsilon"],
        )
        return advantages[:, 0]
    advantages, _ = grpo_stepwise_advantages(events, torch.ones_like(events), objective)
    return advantages[:, 1:-1]


@register_adv_est("lwm_scst_stepwise")
@torch.no_grad()
def scst_stepwise_advantages(token_level_rewards, reward_baselines, response_mask, config):
    """baseline 已按相同正文位置对齐，为独立 greedy 的折扣回报。"""
    returns = discounted_returns(token_level_rewards * response_mask, effective_gamma(config))
    return (returns - reward_baselines) * response_mask, returns


@register_adv_est("lwm_grpo_stepwise")
@torch.no_grad()
def grpo_stepwise_advantages(token_level_rewards, response_mask, config):
    """调用范围是一篇正文的完整采样组；统计保留所有真实强制事件。"""
    mean = masked_mean(token_level_rewards, response_mask)
    std = masked_var(token_level_rewards, response_mask).sqrt()
    normalized = (token_level_rewards - mean) / (std + config["advantage_epsilon"])
    advantages = discounted_returns(normalized * response_mask, effective_gamma(config))
    return advantages * response_mask, advantages * response_mask


@torch.no_grad()
def generalized_advantages(rewards, old_values, event_mask, gamma, gae_lambda):
    """完整全局 rollout 批次调用原生 GAE，actor 优势白化，价值目标不白化。"""
    if int(event_mask.sum()) == 1:
        # 唯一真实事件是全 recent 正文的 EOF，没有自主动作或可估计的样本方差。
        return torch.zeros_like(rewards), rewards.clone()
    return compute_gae_advantage_return(rewards, old_values, event_mask, gamma, gae_lambda)
