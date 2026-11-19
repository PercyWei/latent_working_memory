"""真实 Qwen3 的后训练批量前向、梯度归约与随机数映射回归。"""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import torch

import test_pretrain_batching as pretrain_tests

from latent_working_memory.v3_1_1 import rollout
from latent_working_memory.v3_1_1.training import rl as rl_objective
from latent_working_memory.v3_1_1.config import (
    BackwardConfig,
    ComputeConfig,
    ExperimentConfig,
    PolicyObjectiveConfig,
    ModelConfig,
    ModelLoraConfig,
    ModelStructureConfig,
    OptimizerConfig,
    PolicyTransitionConfig,
    CompressionConfig,
    RunConfig,
    SchedulerConfig,
    SwanLabConfig,
    TrainingConfig,
    TrainingDataConfig,
    TrainingLoopConfig,
)
from latent_working_memory.v3_1_1.data import Document, QA
from latent_working_memory.v3_1_1.modeling.compressor import MemoryModel
from latent_working_memory.v3_1_1.training.engine import TrainingEngine

from latent_working_memory.v3_1_1.rollout import execute, sample_paths


one_cpu_thread = pretrain_tests.one_cpu_thread
tiny_source = pretrain_tests.tiny_source

def _reference_qa_backward(model, memory_leaves, qa, normalizer, retained_ids):
    """单题只反传到独立 memory 叶子；返回统计时释放本题的 decoder 图。"""
    with model.forward_context():
        loss = model.answer_nll(memory_leaves, retained_ids + qa.prompt_ids, qa.target_ids)
        normalized_loss = loss / normalizer
    if normalized_loss.requires_grad:
        normalized_loss.backward()
    return loss.detach()


def _reference_path_backward(
    model, document, pending_stride, grpo, path, advantage, group_size, bptt_steps=None,
    temperature=1.0,
):
    """反传后只返回无图统计，使本路径的中间张量在下一次前向前释放。"""
    with model.forward_context():
        replay = execute(
            model, document.input_ids, pending_stride, actions=path.actions,
            bptt_steps=bptt_steps, temperature=temperature,
        )
        blocks, log_probs, entropy = replay.blocks, replay.log_probs, replay.entropy
    memory_leaves = tuple(block.detach().requires_grad_(True) for block in blocks)
    question_count = len(document.qas)
    question_losses = [None] * question_count
    # 旧 stack(...).mean().backward() 按后建图的题目先执行，保持同样的累加顺序。
    for index in reversed(range(question_count)):
        question_losses[index] = _reference_qa_backward(
            model, memory_leaves, document.qas[index], question_count * group_size, replay.retained_ids,
        )
    with model.forward_context():
        task_loss = torch.stack(question_losses).mean()
        if len(log_probs):
            ratios = (log_probs - path.old_log_probs).clamp(-20, 20).exp()
            clipped = ratios.clamp(1 - grpo["clip_epsilon"], 1 + grpo["clip_epsilon"])
            terms = -torch.minimum(ratios * advantage, clipped * advantage)
            policy_loss = torch.where(advantage < 0, torch.minimum(terms, -3 * advantage), terms).mean()
            policy_entropy = entropy.mean()
            drift = (log_probs - path.old_log_probs).abs().max()
        else:
            policy_loss = task_loss.new_zeros(())
            policy_entropy = policy_loss
            drift = policy_loss
        loss = task_loss + grpo["policy_weight"] * policy_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite training loss: {document.id}")
        policy_backward = (grpo["policy_weight"] * policy_loss / group_size
                           if policy_loss.requires_grad else None)
    outputs = list(blocks)
    gradients = [leaf.grad for leaf in memory_leaves]
    if policy_loss.requires_grad:
        # 零权重也连接策略图，保留原更新中 boundary 的显式零梯度。
        outputs.append(policy_backward)
        gradients.append(None)
    # 同时注入所有直接 QA adjoint 与策略梯度，让共享 history 只反传一次。
    if outputs:
        torch.autograd.backward(outputs, gradients)
    return torch.stack((
        task_loss.detach(), policy_loss.detach(), policy_entropy.detach(), drift.detach(),
    ))


def reference_post_backward(
    model, document, pending_stride, grpo, generator, bptt_steps=None, sampled_paths=None,
    temperature=1.0,
):
    """采样完整组后逐路径累积梯度；文档归约与参数更新由 engine 执行。"""
    # 采样和带图回放使用独立 autocast 区域，避免复用无梯度的权重转换缓存。
    sample_count = grpo["group_size"]
    with torch.no_grad(), model.forward_context():
        paths = (sampled_paths if sampled_paths is not None
                 else sample_paths(model, document, pending_stride, sample_count, generator, temperature))
        if grpo["strategy"] == "scst":
            greedy = sample_paths(model, document, pending_stride, 1, None, temperature)[0]
            greedy_nll = next((path.qa_nll for path in paths if path.actions == greedy.actions), greedy.qa_nll)
            baseline = -greedy_nll - grpo["memory_cost_weight"] * (
                greedy.prefix_retained_counts[-1] / (len(document.input_ids) - len(greedy.recent_ids))
            ) ** grpo["memory_cost_power"]
    device = paths[0].old_log_probs.device
    rewards = torch.tensor([
        -path.qa_nll - grpo["memory_cost_weight"] * (
            path.prefix_retained_counts[-1] / (len(document.input_ids) - len(path.recent_ids))
        ) ** grpo["memory_cost_power"]
        for path in paths
    ], device=device, dtype=torch.float32)
    std = rewards.std(unbiased=False)
    advantages = ((rewards - rewards.new_tensor(baseline)) if grpo["strategy"] == "scst"
                  else (rewards - rewards.mean()) / (
                      rewards.std(unbiased=True) + grpo["advantage_epsilon"]
                  )).detach()
    group_size = len(paths)
    values = torch.stack([
        _reference_path_backward(
            model, document, pending_stride, grpo, path, advantage, group_size, bptt_steps, temperature,
        )
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
        "zero_advantage_rate": (float(bool((advantages == 0).all())), 1),
        "blocks": (blocks, group_size),
        "memory_slots": (blocks * model.memory_slots, group_size),
        "segment_tokens": (sum(sum(path.block_lengths) for path in paths), blocks),
        "commit_rate": (sum(sum(path.actions) for path in paths), decisions),
        "policy_entropy": (
            float(values[:, 2].sum()) * len(paths[0].actions) if decisions else 0,
            decisions,
        ),
        "replay_log_probability_drift": (float(values[:, 3].sum()), group_size),
    }
    peaks = []
    for path in paths:
        pending = min(pending_stride, len(document.input_ids))
        peak = pending
        for index, commit in enumerate(path.actions):
            arriving = min(pending_stride, len(document.input_ids) - (index + 1) * pending_stride)
            peak = max(peak, pending + arriving)
            pending = arriving if commit else pending + arriving
        peaks.append(peak)
    retention = [(path.prefix_retained_counts[-1] + len(path.recent_ids)) / len(document.input_ids)
                 for path in paths]
    prefix_retention = [path.prefix_retained_counts[-1] / (len(document.input_ids) - len(path.recent_ids))
                        for path in paths]
    metrics.update({
        "recent_tokens": (sum(len(path.recent_ids) for path in paths), group_size),
        "pending_tokens": (sum(len(path.pending_ids) for path in paths), group_size),
        "retention_ratio": (sum(retention), group_size),
        "prefix_retention_ratio": (sum(prefix_retention), group_size),
        "memory_cost": (sum(value ** grpo["memory_cost_power"] for value in prefix_retention), group_size),
        "pending_peak": (sum(peaks), group_size),
        "advantage_abs_mean": (float(advantages.abs().sum()), advantages.numel()),
    })
    if grpo["strategy"] == "scst":
        metrics["baseline_reward"] = (float(rewards.new_tensor(baseline)), 1)
    return qa_loss + grpo["policy_weight"] * policy_loss, metrics


def make_post_models(folder, gradient_checkpointing=False, bptt_steps=2, policy_weight=1.0):
    config = ExperimentConfig(
        stage="post", model_name_or_path=str(folder),
        model=ModelConfig(
            structure=ModelStructureConfig(memory_slots=4, pending_stride=4, pending_size=4,
                                           recent_size=0, boundary_hidden_size=8),
            lora=ModelLoraConfig(lora_rank=2, lora_alpha=4),
        ),
        compute=ComputeConfig(gradient_checkpointing=gradient_checkpointing),
        training=TrainingConfig(
            data=TrainingDataConfig(dataset_dir=str(folder)),
            compression=CompressionConfig(mode="policy"),
            update_mode="joint",
            objective=PolicyObjectiveConfig(
                strategy="grpo", group_size=4, memory_cost_weight=0.01,
                policy_weight=policy_weight, advantage_epsilon=1e-6,
            ),
            loop=TrainingLoopConfig(
                micro_batch_size=4, gradient_accumulation_steps=2, max_steps=6,
            ),
            optimizer=OptimizerConfig(boundary_learning_rate=1e-4, gradient_clip=0.5),
            scheduler=SchedulerConfig(warmup_steps=0),
            backward=BackwardConfig(bptt_steps=bptt_steps),
            run=RunConfig(init_checkpoint=str(folder)),
            swanlab=SwanLabConfig(tracking=False),
        ),
    )
    torch.manual_seed(721)
    reference = MemoryModel(str(folder), asdict(config.model), asdict(config.compute),
                            torch.device("cpu"))
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.025)
        reference.boundary.network[-1].weight.normal_(0, 0.05)
        reference.boundary.network[-1].bias.fill_(0.25)
    actual = MemoryModel(str(folder), asdict(config.model), asdict(config.compute),
                         torch.device("cpu"))
    actual.load_state_dict(reference.state_dict())
    return reference, actual, config


def document(index, length, question_count):
    text = tuple(3 + (index + offset) % 61 for offset in range(length))
    qas = tuple(QA(
        question=f"q{index}-{question}", answer=f"a{question}",
        prompt_ids=tuple(range(3, 5 + question % 3)),
        target_ids=tuple(20 + offset for offset in range(1 + (index + question) % 4)) + (1,),
    ) for question in range(question_count))
    return Document(f"doc-{index}", f"doc-{index}", text, qas)


def gradients(model):
    return {name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def assert_gradients_close(reference, actual):
    assert reference.keys() == actual.keys()
    for name, expected in reference.items():
        result = actual[name]
        assert (expected is None) == (result is None), name
        if expected is not None:
            assert torch.isfinite(result).all(), name
            torch.testing.assert_close(result, expected, rtol=4e-4, atol=4e-6, msg=name)


def fixed_paths(model, rows, pending_stride, group_size):
    result = {}
    with torch.no_grad(), model.forward_context():
        for row in rows:
            count = (len(row.input_ids) - 1) // pending_stride
            paths = []
            for path_index in range(group_size):
                actions = tuple(
                    False if path_index == 0 else True if path_index == 1
                    else (index + path_index) % 2 == 0
                    for index in range(count)
                )
                replay = execute(model, row.input_ids, pending_stride, actions=actions)
                blocks, log_probs, lengths = replay.blocks, replay.log_probs, replay.block_lengths
                # 固定非单位比率也覆盖 PPO clip 的两侧分支。
                old_log_probs = log_probs.detach() + (-0.4, 0.4, 0.0, 0.15)[path_index]
                nll = rollout.qa_nll(model, blocks, row, replay.retained_ids) + path_index * 0.03
                paths.append(rollout.Rollout(
                    actions, old_log_probs, float(nll), lengths,
                    replay.recent_ids, replay.decision_mask, replay.prefix_retained_counts, replay.pending_ids,
                ))
            result[row.id] = tuple(paths)
    return result


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
@pytest.mark.parametrize("bptt_steps", [None, 1, 2])
@pytest.mark.parametrize("temperature", [1.0, 0.35])
def test_ragged_replay_preserves_memories_gradients_and_truncation(
    tiny_source, gradient_checkpointing, bptt_steps, temperature,
):
    reference, actual, _ = make_post_models(tiny_source, gradient_checkpointing, bptt_steps)
    contexts = [document(i, length, 1).input_ids for i, length in enumerate([3, 12, 19, 9])]
    actions = [(), (False, False), (True, False, True, True), (True, True)]
    expected = [execute(reference, ids, 4, actions=path, bptt_steps=bptt_steps, temperature=temperature)
                for ids, path in zip(contexts, actions, strict=True)]
    shapes = []
    handle = actual.writer.get_base_model().model.register_forward_pre_hook(
        lambda module, args, kwargs: shapes.append(kwargs["inputs_embeds"].shape[0]),
        with_kwargs=True,
    )
    results = rollout.execute_replay_batch(actual, contexts, 4, actions, bptt_steps, temperature)
    for left, right in zip(expected, results, strict=True):
        assert left[1] == right[1] and left[4] == right[4]
        for a, b in zip(left[0], right[0], strict=True):
            torch.testing.assert_close(b, a, rtol=3e-4, atol=3e-6)
        for position in [2, 3]:
            torch.testing.assert_close(right[position], left[position], rtol=3e-4, atol=3e-6)
    def loss(rows):
        value = sum(sum(block.square().mean() * (index + 1)
                        for index, block in enumerate(row[0])) for row in rows)
        return value + sum(row[2].sum() + row[3].sum() * 0.07 for row in rows)
    loss(expected).backward()
    loss(results).backward()
    handle.remove()
    assert max(shapes) == len(contexts) - 1
    assert_gradients_close(gradients(reference), gradients(actual))


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
@pytest.mark.parametrize("temperature", [1.0, 0.35])
def test_stochastic_sampling_preserves_document_major_random_draws(tiny_source, gradient_checkpointing, temperature):
    reference, actual, config = make_post_models(tiny_source, gradient_checkpointing)
    with torch.no_grad():
        reference.boundary.network[-1].weight.zero_()
        actual.load_state_dict(reference.state_dict())
    rows = [document(i, length, count)
            for i, (length, count) in enumerate([(3, 1), (12, 3), (19, 2), (9, 10)])]
    generator = torch.Generator().manual_seed(417)
    other = torch.Generator().set_state(generator.get_state())
    pending_stride = config.model.structure.pending_stride
    grpo = asdict(config.training.objective)
    with torch.no_grad(), reference.forward_context():
        expected = [sample_paths(reference, row, pending_stride, grpo["group_size"], generator, temperature) for row in rows]
    shapes = []
    handle = actual.decoder.model.register_forward_pre_hook(
        lambda module, args, kwargs: shapes.append(kwargs["inputs_embeds"].shape[0]),
        with_kwargs=True,
    )
    with torch.no_grad(), actual.forward_context():
        results = rollout.sample_paths_batch(actual, rows, pending_stride, grpo["group_size"], other, temperature)
    handle.remove()
    assert max(shapes) == len(rows)
    assert torch.equal(generator.get_state(), other.get_state())
    for left_group, right_group in zip(expected, results, strict=True):
        assert len(left_group) == len(right_group) == 4
        for left, right in zip(left_group, right_group, strict=True):
            assert left.actions == right.actions and left.block_lengths == right.block_lengths
            torch.testing.assert_close(right.old_log_probs, left.old_log_probs, rtol=3e-4, atol=3e-6)
            assert right.qa_nll == pytest.approx(left.qa_nll, rel=2e-5, abs=3e-6)


def reference_update(engine, rows, paths, step):
    engine.model.train()
    engine.optimizer.zero_grad(set_to_none=True)
    pending_stride = engine.experiment.model.structure.pending_stride
    grpo = asdict(engine.config.objective)
    transition = engine.config.objective.policy_transition
    temperature = transition.temperature(step, engine.config.loop.max_steps) if transition else 1.0
    outputs = [reference_post_backward(
        engine.model, row, pending_stride, grpo, None, engine.config.backward.bptt_steps,
        sampled_paths=paths[row.id], temperature=temperature,
    ) for row in rows]
    for parameter in engine.parameters:
        if parameter.grad is not None:
            parameter.grad.div_(len(rows))
    raw_gradients = gradients(engine.model)
    torch.nn.utils.clip_grad_norm_(engine.parameters, engine.config.optimizer.gradient_clip)
    engine.optimizer.step()
    engine.scheduler.step()
    return outputs, raw_gradients


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
@pytest.mark.parametrize("annealing", [False, True])
def test_engine_microbatch_accumulation_tail_and_state_restore_continuation(
    tiny_source, gradient_checkpointing, monkeypatch, annealing,
):
    reference, actual, config = make_post_models(tiny_source, gradient_checkpointing)
    if annealing:
        config = replace(config, training=replace(config.training, objective=replace(
            config.training.objective, policy_transition=PolicyTransitionConfig(
                method="temperature_annealing", schedule="linear",
                start_temperature=1.0, end_temperature=0.2,
                start_progress=0.0, end_progress=1.0,
            ),
        )))
    expected_engine, engine = TrainingEngine(reference, config), TrainingEngine(actual, config)
    rows = [document(i, length, 10 if i == 3 else 1 + i % 3)
            for i, length in enumerate([3, 12, 19, 9, 14, 5, 10, 17])]
    tail = document(9, 7, 2)
    paths = fixed_paths(
        reference, [*rows, tail], config.model.structure.pending_stride, config.training.objective.group_size,
    )
    calls, writer_shapes, reader_shapes, captured = [], [], [], []
    def sampling(model, batch, pending_stride, group_size, generator, temperature=1.0,
                 score_cache=None, pending_size=None, recent_size=0):
        calls.append([row.id for row in batch])
        return [paths[row.id] for row in batch]
    monkeypatch.setattr(rl_objective, "sample_paths_batch", sampling)
    synchronize = engine._synchronize_gradients
    def capture(samples):
        synchronize(samples)
        captured.append(gradients(engine.model))
    monkeypatch.setattr(engine, "_synchronize_gradients", capture)
    handles = [
        actual.writer.get_base_model().model.register_forward_pre_hook(
            lambda module, args, kwargs: writer_shapes.append(kwargs["inputs_embeds"].shape[0]),
            with_kwargs=True,
        ),
        actual.decoder.model.register_forward_pre_hook(
            lambda module, args, kwargs: reader_shapes.append(kwargs["inputs_embeds"].shape[0]),
            with_kwargs=True,
        ),
    ]
    for step, batch in enumerate([rows, [tail]], 1):
        before_calls = len(calls)
        expected, raw_gradients = reference_update(expected_engine, batch, paths, step)
        metrics = engine.train_batch(batch, torch.Generator().manual_seed(415), step)
        if annealing:
            assert metrics["policy_temperature"] == pytest.approx(1 - (step - 1) * 0.16)
        else:
            assert "policy_temperature" not in metrics
        assert calls[before_calls:] == [[row.id for row in batch[start:start + 4]]
                                        for start in range(0, len(batch), 4)]
        assert metrics["samples"] == len(batch)
        assert metrics["loss"] == pytest.approx(
            sum(float(loss) for loss, _ in expected) / len(batch), rel=3e-5, abs=5e-6,
        )
        assert metrics["qa_nll"] == pytest.approx(
            sum(values["qa_nll"][0] for _, values in expected) / len(batch),
            rel=3e-5, abs=5e-6,
        )
        assert_gradients_close(raw_gradients, captured[-1])
        for (name, left), (_, right) in zip(reference.named_parameters(), actual.named_parameters(), strict=True):
            torch.testing.assert_close(right, left, rtol=4e-4, atol=4e-6, msg=name)
            if not right.requires_grad:
                assert right.grad is None
        assert engine.scheduler.state_dict() == expected_engine.scheduler.state_dict()
    assert max(writer_shapes) == max(reader_shapes) == 4
    assert writer_shapes[-1] == reader_shapes[-1] == 1
    for handle in handles:
        handle.remove()
    # 使用真实 state_dict 继续一步，确保批量图不引入未持久化的训练状态。
    _, restored, _ = make_post_models(tiny_source, gradient_checkpointing)
    restored.load_state_dict(actual.state_dict())
    restored_engine = TrainingEngine(restored, config)
    restored_engine.optimizer.load_state_dict(deepcopy(engine.optimizer.state_dict()))
    restored_engine.scheduler.load_state_dict(engine.scheduler.state_dict())
    expected_engine = TrainingEngine(actual, config)
    expected_engine.optimizer.load_state_dict(deepcopy(engine.optimizer.state_dict()))
    expected_engine.scheduler.load_state_dict(engine.scheduler.state_dict())
    first = expected_engine.train_batch([tail], torch.Generator().manual_seed(123), 3)
    second = restored_engine.train_batch([tail], torch.Generator().manual_seed(123), 3)
    assert first == second
    if annealing:
        assert first["policy_temperature"] == pytest.approx(0.68)
    assert_gradients_close(gradients(actual), gradients(restored))
    for left, right in zip(actual.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(right, left, rtol=0, atol=0)


@pytest.mark.parametrize("no_actions,policy_weight", [(True, 1.0), (False, 0.0)])
def test_no_decisions_and_zero_policy_weight_preserve_gradient_presence(
    tiny_source, no_actions, policy_weight, monkeypatch,
):
    reference, actual, config = make_post_models(tiny_source, policy_weight=policy_weight)
    rows = [document(0, 3 if no_actions else 9, 1), document(1, 2 if no_actions else 12, 3)]
    pending_stride = config.model.structure.pending_stride
    grpo = asdict(config.training.objective)
    paths = fixed_paths(reference, rows, pending_stride, grpo["group_size"])
    monkeypatch.setattr(rl_objective, "sample_paths_batch",
                        lambda model, docs, pending_stride, group_size, generator, temperature=1.0,
                        score_cache=None, pending_size=None, recent_size=0: [paths[row.id] for row in docs])
    expected = [reference_post_backward(reference, row, pending_stride, grpo, None, 2, paths[row.id]) for row in rows]
    results = rl_objective.post_backward_batch(actual, rows, pending_stride, grpo, None, 2)
    for (left_loss, left_metrics), (right_loss, right_metrics) in zip(expected, results, strict=True):
        torch.testing.assert_close(right_loss, left_loss, rtol=3e-5, atol=5e-6)
        assert right_metrics.keys() == left_metrics.keys()
        for name in left_metrics:
            left_numerator, left_denominator = left_metrics[name]
            right_numerator, right_denominator = right_metrics[name]
            assert bool(left_denominator) == bool(right_denominator), name
            if left_denominator:
                assert right_numerator / right_denominator == pytest.approx(
                    left_numerator / left_denominator, rel=3e-5, abs=5e-6,
                ), name
            else:
                assert right_numerator == left_numerator == 0
    assert_gradients_close(gradients(reference), gradients(actual))
    for parameter in actual.boundary.parameters():
        if no_actions:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0


def test_identical_sampled_paths_keep_identical_rewards_despite_batch_score_roundoff(
    tiny_source, monkeypatch,
):
    _, model, config = make_post_models(tiny_source)
    with torch.no_grad():
        model.boundary.network[-1].weight.zero_()
        model.boundary.network[-1].bias.fill_(100.0)
    rows = [document(0, 9, 1), document(1, 14, 2)]
    pending_stride = config.model.structure.pending_stride
    grpo = asdict(config.training.objective)
    scoring = model.answer_nll_batch
    score_calls = []

    def rounded_score(memories, prompts, targets):
        score_calls.append(len(memories))
        # 同一轨迹在不同批量计算形状中可能出现微小的评分舍入差。
        return scoring(memories, prompts, targets) + len(score_calls) * 1e-4

    monkeypatch.setattr(model, "answer_nll_batch", rounded_score)
    with torch.no_grad(), model.forward_context():
        groups = rollout.sample_paths_batch(
            model, rows, pending_stride, grpo["group_size"], torch.Generator().manual_seed(417),
        )
    # 保留各轮真实评分的批量形状，只复用同轨迹的确定性奖励。
    assert score_calls == [2, 1] * grpo["group_size"]
    for row, group in zip(rows, groups, strict=True):
        assert all(path.actions == group[0].actions for path in group)
        assert all(path.qa_nll == group[0].qa_nll for path in group)
        rewards = torch.tensor([
            -path.qa_nll - grpo["memory_cost_weight"] * (
                path.prefix_retained_counts[-1] / (len(row.input_ids) - len(path.recent_ids))
            ) ** grpo["memory_cost_power"]
            for path in group
        ])
        assert rewards.std(unbiased=False) == 0
    scst = asdict(replace(config.training.objective, strategy="scst", group_size=1, advantage_epsilon=None))
    results = rl_objective.post_backward_batch(
        model, rows, pending_stride, scst, torch.Generator().manual_seed(417), temperature=0.5,
    )
    for _, metrics in results:
        assert metrics["reward"] == metrics["baseline_reward"]
        assert metrics["zero_advantage_rate"] == (1.0, 1)
    for parameter in model.boundary.parameters():
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("group_size", [1, 4])
@pytest.mark.parametrize("temperature", [1.0, 0.35])
def test_scst_matches_independent_path_gradients_and_preserves_sampling_rng(
    tiny_source, group_size, temperature,
):
    reference, actual, config = make_post_models(tiny_source)
    with torch.no_grad():
        reference.boundary.network[-1].weight.zero_()
        actual.load_state_dict(reference.state_dict())
    rl = asdict(replace(config.training.objective, strategy="scst", group_size=group_size,
                        advantage_epsilon=None))
    rows = [document(0, 19, 1), document(1, 12, 3)]
    generator = torch.Generator().manual_seed(417)
    other = torch.Generator().set_state(generator.get_state())
    expected = [reference_post_backward(
        reference, row, 4, rl, generator, 2, temperature=temperature,
    ) for row in rows]
    results = rl_objective.post_backward_batch(actual, rows, 4, rl, other, 2, temperature)
    assert torch.equal(generator.get_state(), other.get_state())
    assert any(metrics["zero_advantage_rate"][0] == 0 for _, metrics in results)
    for (left_loss, left_metrics), (right_loss, right_metrics) in zip(expected, results, strict=True):
        torch.testing.assert_close(right_loss, left_loss, rtol=3e-5, atol=5e-6)
        assert right_metrics.keys() == left_metrics.keys()
        for name in left_metrics:
            left_numerator, left_denominator = left_metrics[name]
            right_numerator, right_denominator = right_metrics[name]
            assert right_denominator == left_denominator
            assert right_numerator == pytest.approx(left_numerator, rel=4e-4, abs=1e-5), name
        if group_size == 1:
            assert right_metrics["reward_std"] == (0.0, 1)
    assert_gradients_close(gradients(reference), gradients(actual))
