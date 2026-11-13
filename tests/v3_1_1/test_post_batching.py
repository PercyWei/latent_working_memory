"""真实 Qwen3 的后训练批量前向、梯度归约与随机数映射回归。"""

from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

import test_pretrain_batching as pretrain_tests

from latent_working_memory.v3_1_1 import objective, rollout
from latent_working_memory.v3_1_1.config import (
    BackwardConfig,
    ComputeConfig,
    ExperimentConfig,
    GRPOConfig,
    MethodConfig,
    ModelConfig,
    ModelLoraConfig,
    ModelStructureConfig,
    OptimizerConfig,
    RolloutConfig,
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

from latent_working_memory.v3_1_1.rollout import execute, sample_group


one_cpu_thread = pretrain_tests.one_cpu_thread
tiny_source = pretrain_tests.tiny_source

def _reference_qa_backward(model, memory_leaves, qa, normalizer):
    """单题只反传到独立 memory 叶子；返回统计时释放本题的 decoder 图。"""
    with model.forward_context():
        loss = model.answer_nll(memory_leaves, qa.prompt_ids, qa.target_ids)
        normalized_loss = loss / normalizer
    normalized_loss.backward()
    return loss.detach()


def _reference_path_backward(model, document, method, path, advantage, group_size, bptt_steps=None):
    """反传后只返回无图统计，使本路径的中间张量在下一次前向前释放。"""
    with model.forward_context():
        blocks, _, log_probs, entropy, _ = execute(
            model, document.input_ids, method["rollout"]["unit_tokens"], actions=path.actions,
            bptt_steps=bptt_steps,
        )
    memory_leaves = tuple(block.detach().requires_grad_(True) for block in blocks)
    question_count = len(document.qas)
    question_losses = [None] * question_count
    # 旧 stack(...).mean().backward() 按后建图的题目先执行，保持同样的累加顺序。
    for index in reversed(range(question_count)):
        question_losses[index] = _reference_qa_backward(
            model, memory_leaves, document.qas[index], question_count * group_size,
        )
    with model.forward_context():
        task_loss = torch.stack(question_losses).mean()
        grpo = method["grpo"]
        if len(log_probs):
            ratios = (log_probs - path.old_log_probs).exp()
            clipped = ratios.clamp(1 - grpo["clip_epsilon"], 1 + grpo["clip_epsilon"])
            policy_loss = -torch.minimum(ratios * advantage, clipped * advantage).mean()
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
    torch.autograd.backward(outputs, gradients)
    return torch.stack((
        task_loss.detach(), policy_loss.detach(), policy_entropy.detach(), drift.detach(),
    ))


def reference_post_backward(model, document, method, generator, bptt_steps=None, sampled_paths=None):
    """采样完整组后逐路径累积梯度；文档归约与参数更新由 engine 执行。"""
    # 采样和带图回放使用独立 autocast 区域，避免复用无梯度的权重转换缓存。
    with torch.no_grad(), model.forward_context():
        paths = (sampled_paths if sampled_paths is not None
                 else sample_group(model, document, method["rollout"], generator))
    grpo = method["grpo"]
    device = paths[0].old_log_probs.device
    rewards = torch.tensor([
        -path.qa_nll - grpo["slot_cost"] * model.memory_slots * len(path.block_lengths)
        for path in paths
    ], device=device, dtype=torch.float32)
    std = rewards.std(unbiased=False)
    advantages = ((rewards - rewards.mean()) / (std + grpo["advantage_epsilon"])).detach()
    group_size = len(paths)
    values = torch.stack([
        _reference_path_backward(model, document, method, path, advantage, group_size, bptt_steps)
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
    return qa_loss + grpo["policy_weight"] * policy_loss, metrics


def make_post_models(folder, gradient_checkpointing=False, bptt_steps=2, policy_weight=1.0):
    config = ExperimentConfig(
        stage="post", model_name_or_path=str(folder),
        model=ModelConfig(
            structure=ModelStructureConfig(memory_slots=4, boundary_hidden_size=8),
            lora=ModelLoraConfig(lora_rank=2, lora_alpha=4),
        ),
        compute=ComputeConfig(gradient_checkpointing=gradient_checkpointing),
        method=MethodConfig(
            rollout=RolloutConfig(unit_tokens=4, group_size=4),
            grpo=GRPOConfig(slot_cost=0.01, policy_weight=policy_weight),
        ),
        training=TrainingConfig(
            data=TrainingDataConfig(dataset_dir=str(folder)),
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


def fixed_paths(model, rows, method):
    result = {}
    with torch.no_grad(), model.forward_context():
        for row in rows:
            count = (len(row.input_ids) - 1) // method["rollout"]["unit_tokens"]
            paths = []
            for path_index in range(method["rollout"]["group_size"]):
                actions = tuple(
                    False if path_index == 0 else True if path_index == 1
                    else (index + path_index) % 2 == 0
                    for index in range(count)
                )
                blocks, _, log_probs, _, lengths = execute(
                    model, row.input_ids, method["rollout"]["unit_tokens"], actions=actions,
                )
                # 固定非单位比率也覆盖 PPO clip 的两侧分支。
                old_log_probs = log_probs.detach() + (-0.4, 0.4, 0.0, 0.15)[path_index]
                nll = rollout.qa_nll(model, blocks, row) + path_index * 0.03
                paths.append(rollout.Rollout(actions, old_log_probs, float(nll), lengths))
            result[row.id] = tuple(paths)
    return result


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
@pytest.mark.parametrize("bptt_steps", [None, 1, 2])
def test_ragged_replay_preserves_memories_gradients_and_truncation(
    tiny_source, gradient_checkpointing, bptt_steps,
):
    reference, actual, _ = make_post_models(tiny_source, gradient_checkpointing, bptt_steps)
    contexts = [document(i, length, 1).input_ids for i, length in enumerate([3, 12, 19, 9])]
    actions = [(), (False, False), (True, False, True, True), (True, True)]
    expected = [execute(reference, ids, 4, actions=path, bptt_steps=bptt_steps)
                for ids, path in zip(contexts, actions, strict=True)]
    shapes = []
    handle = actual.writer.get_base_model().model.register_forward_pre_hook(
        lambda module, args, kwargs: shapes.append(kwargs["inputs_embeds"].shape[0]),
        with_kwargs=True,
    )
    results = rollout.execute_replay_batch(actual, contexts, 4, actions, bptt_steps)
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
    assert max(shapes) == len(contexts)
    assert_gradients_close(gradients(reference), gradients(actual))


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_stochastic_sampling_preserves_document_major_random_draws(tiny_source, gradient_checkpointing):
    reference, actual, config = make_post_models(tiny_source, gradient_checkpointing)
    with torch.no_grad():
        reference.boundary.network[-1].weight.zero_()
        actual.load_state_dict(reference.state_dict())
    rows = [document(i, length, count)
            for i, (length, count) in enumerate([(3, 1), (12, 3), (19, 2), (9, 10)])]
    generator = torch.Generator().manual_seed(417)
    other = torch.Generator().set_state(generator.get_state())
    method = asdict(config.method)
    with torch.no_grad(), reference.forward_context():
        expected = [sample_group(reference, row, method["rollout"], generator) for row in rows]
    shapes = []
    handle = actual.decoder.model.register_forward_pre_hook(
        lambda module, args, kwargs: shapes.append(kwargs["inputs_embeds"].shape[0]),
        with_kwargs=True,
    )
    with torch.no_grad(), actual.forward_context():
        results = rollout.sample_group_batch(actual, rows, method["rollout"], other)
    handle.remove()
    assert max(shapes) == len(rows)
    assert torch.equal(generator.get_state(), other.get_state())
    for left_group, right_group in zip(expected, results, strict=True):
        assert len(left_group) == len(right_group) == 4
        for left, right in zip(left_group, right_group, strict=True):
            assert left.actions == right.actions and left.block_lengths == right.block_lengths
            torch.testing.assert_close(right.old_log_probs, left.old_log_probs, rtol=3e-4, atol=3e-6)
            assert right.qa_nll == pytest.approx(left.qa_nll, rel=2e-5, abs=3e-6)


def reference_update(engine, rows, paths):
    engine.model.train()
    engine.optimizer.zero_grad(set_to_none=True)
    method = asdict(engine.experiment.method)
    outputs = [reference_post_backward(
        engine.model, row, method, None, engine.config.backward.bptt_steps,
        sampled_paths=paths[row.id],
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
def test_engine_microbatch_accumulation_tail_and_state_restore_continuation(
    tiny_source, gradient_checkpointing, monkeypatch,
):
    reference, actual, config = make_post_models(tiny_source, gradient_checkpointing)
    expected_engine, engine = TrainingEngine(reference, config), TrainingEngine(actual, config)
    rows = [document(i, length, 10 if i == 3 else 1 + i % 3)
            for i, length in enumerate([3, 12, 19, 9, 14, 5, 10, 17])]
    tail = document(9, 7, 2)
    paths = fixed_paths(reference, [*rows, tail], asdict(config.method))
    calls, writer_shapes, reader_shapes, captured = [], [], [], []
    def sampling(model, batch, rollout_config, generator):
        calls.append([row.id for row in batch])
        return [paths[row.id] for row in batch]
    monkeypatch.setattr(objective, "sample_group_batch", sampling)
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
    for batch in [rows, [tail]]:
        before_calls = len(calls)
        expected, raw_gradients = reference_update(expected_engine, batch, paths)
        metrics = engine.train_batch(batch, torch.Generator().manual_seed(415))
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
    first = expected_engine.train_batch([tail], torch.Generator().manual_seed(123))
    second = restored_engine.train_batch([tail], torch.Generator().manual_seed(123))
    assert first == second
    assert_gradients_close(gradients(actual), gradients(restored))
    for left, right in zip(actual.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(right, left, rtol=0, atol=0)


@pytest.mark.parametrize("no_actions,policy_weight", [(True, 1.0), (False, 0.0)])
def test_no_decisions_and_zero_policy_weight_preserve_gradient_presence(
    tiny_source, no_actions, policy_weight, monkeypatch,
):
    reference, actual, config = make_post_models(tiny_source, policy_weight=policy_weight)
    rows = [document(0, 3 if no_actions else 9, 1), document(1, 2 if no_actions else 12, 3)]
    method = asdict(config.method)
    paths = fixed_paths(reference, rows, method)
    monkeypatch.setattr(objective, "sample_group_batch",
                        lambda model, docs, rollout_config, generator: [paths[row.id] for row in docs])
    expected = [reference_post_backward(reference, row, method, None, 2, paths[row.id]) for row in rows]
    results = objective.post_backward_batch(actual, rows, method, None, 2)
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
    method = asdict(config.method)
    scoring = model.answer_nll_batch
    score_calls = []

    def rounded_score(memories, prompts, targets):
        score_calls.append(len(memories))
        # 同一轨迹在不同批量计算形状中可能出现微小的评分舍入差。
        return scoring(memories, prompts, targets) + len(score_calls) * 1e-4

    monkeypatch.setattr(model, "answer_nll_batch", rounded_score)
    with torch.no_grad(), model.forward_context():
        groups = rollout.sample_group_batch(
            model, rows, method["rollout"], torch.Generator().manual_seed(417),
        )
    # 保留各轮真实评分的批量形状，只复用同轨迹的确定性奖励。
    assert score_calls == [2, 1] * method["rollout"]["group_size"]
    for group in groups:
        assert all(path.actions == group[0].actions for path in group)
        assert all(path.qa_nll == group[0].qa_nll for path in group)
        rewards = torch.tensor([
            -path.qa_nll - method["grpo"]["slot_cost"] * model.memory_slots * len(path.block_lengths)
            for path in group
        ])
        assert rewards.std(unbiased=False) == 0
