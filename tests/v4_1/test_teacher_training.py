"""运行时 teacher 参照不改变 student 目标、元梯度与推理边界。"""

from contextlib import nullcontext
from dataclasses import replace
import weakref

import pytest
import torch
from transformers.cache_utils import DynamicCache

from latent_working_memory.v4_1.data import compression_boundaries
from latent_working_memory.v4_1.modeling.model import OnlineMemoryLM
from latent_working_memory.v4_1.teacher.cache import content_identity
from latent_working_memory.v4_1.teacher.runtime import RuntimeTeacher
from latent_working_memory.v4_1.training.engine import TrainingEngine, evaluate_dev


class FixedReferences:
    """测试对照：预先物化相同采样位置的 q/o/z，模拟原张量缓存路径。"""

    def __init__(self, references):
        self.references = references

    def reference(self, write_index, processed_end, compressed_end, generator=None):
        return self.references[(write_index, processed_end, compressed_end)]


class FixedTeacher:
    """混合正文长度对照使用已物化的逐轮参照，不经过原文缓存身份校验。"""

    def __init__(self, references):
        self.references = references

    def episode(self, episode, generator=None):
        return nullcontext(FixedReferences(self.references))


def _mixed_episodes_and_references(setup):
    teacher = RuntimeTeacher(setup.teacher, setup.config, setup.cache)
    boundaries = compression_boundaries(
        len(setup.episode.context_ids), setup.config.model.pending_size, setup.config.model.recent_size,
    )
    with teacher.episode(setup.episode, torch.Generator().manual_seed(19)) as trajectory:
        references = {boundary: trajectory.reference(*boundary) for boundary in boundaries}
    episodes = (
        setup.episode,
        replace(setup.episode, id="short-document:0", document_id="short-document",
                context_ids=tuple(reversed(setup.episode.context_ids[:12])), target_ids=(10, 1)),
        replace(setup.episode, id="uncompressed-document:0", document_id="uncompressed-document",
                context_ids=setup.episode.context_ids[:7], question_ids=(5, 6), target_ids=(1,)),
    )
    return episodes, references


def test_runtime_teacher_matches_materialized_loss_and_meta_gradients(tiny_setup, monkeypatch):
    setup = tiny_setup
    teacher = RuntimeTeacher(setup.teacher, setup.config, setup.cache)
    boundaries = compression_boundaries(
        len(setup.episode.context_ids), setup.config.model.pending_size, setup.config.model.recent_size
    )
    # 对照逐轮计算：相同查询、目标及采样 RNG，不依赖 student 的执行顺序。
    direct_generator = torch.Generator().manual_seed(19)
    prefix = DynamicCache(config=setup.teacher.config)
    references = {
        boundary: teacher._reference(
            setup.episode, prefix, content_identity(setup.episode.context_ids),
            *boundary, direct_generator,
        )
        for boundary in boundaries
    }
    del prefix
    prepared, prefixes = [], []
    original_reference = teacher._reference

    def track_reference(episode, prefix, identity, *args):
        prepared.append(args[:3])
        prefixes.append(weakref.ref(prefix))
        return original_reference(episode, prefix, identity, *args)

    monkeypatch.setattr(teacher, "_reference", track_reference)
    generator = torch.Generator().manual_seed(19)
    with teacher.episode(setup.episode, generator) as trajectory:
        assert prepared == list(boundaries)
        assert all(prefix() is None for prefix in prefixes)
        torch.testing.assert_close(generator.get_state(), direct_generator.get_state(), rtol=0, atol=0)
        for boundary in boundaries:
            actual = trajectory.reference(*boundary)
            for name, expected in references[boundary].items():
                torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)
            assert boundary not in trajectory.references
        assert not trajectory.references
    assert not trajectory.references
    assert all(not tensor.requires_grad for reference in references.values()
               for tensor in reference.values())
    runtime = OnlineMemoryLM(setup.config.model)
    materialized = OnlineMemoryLM(setup.config.model)
    materialized.load_trainable_state_dict(runtime.trainable_state_dict())
    teacher_weights = {name: tensor.clone() for name, tensor in setup.teacher.state_dict().items()}
    with teacher.episode(setup.episode, torch.Generator().manual_seed(19)) as trajectory:
        runtime_output = runtime(setup.episode, teacher=trajectory)
        assert not trajectory.references
    runtime_output["loss"].backward()
    fixed_output = materialized(setup.episode, teacher=FixedReferences(references))
    fixed_output["loss"].backward()
    for metric in ("qa_loss", "outer_am", "loss"):
        torch.testing.assert_close(runtime_output[metric], fixed_output[metric], rtol=1e-5, atol=1e-6)
    for (name, actual), (_, expected) in zip(runtime.named_parameters(), materialized.named_parameters()):
        if actual.requires_grad:
            assert actual.grad is not None, name
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-6)
    for name, tensor in setup.teacher.state_dict().items():
        torch.testing.assert_close(tensor, teacher_weights[name], rtol=0, atol=0)
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in setup.teacher.parameters())
    assert not any("lora_" in name for name, _ in setup.teacher.named_parameters())


def test_per_gradient_checkpoint_preserves_multistep_meta_gradients(tiny_setup, monkeypatch):
    setup = tiny_setup
    config = replace(setup.config.model, inner_steps=3)
    teacher = RuntimeTeacher(setup.teacher, setup.config, setup.cache)
    boundaries = compression_boundaries(
        len(setup.episode.context_ids), config.pending_size, config.recent_size,
    )
    assert len(boundaries) == 2
    with teacher.episode(setup.episode, torch.Generator().manual_seed(19)) as trajectory:
        references = FixedReferences({
            boundary: trajectory.reference(*boundary)
            for boundary in boundaries
        })
    checked = OnlineMemoryLM(config)
    ordinary = OnlineMemoryLM(replace(config, gradient_checkpointing=False))
    ordinary.load_trainable_state_dict(checked.trainable_state_dict())
    inner_inputs = []
    for memory in checked.memory_layers:
        def track_write(old_memory, teacher_kv, queries, _write=memory.write, **kwargs):
            for tensor in (*teacher_kv[:2], queries):
                assert tensor.requires_grad
                tensor.retain_grad()
                inner_inputs.append(tensor)
            return _write(old_memory, teacher_kv, queries, **kwargs)

        monkeypatch.setattr(memory, "write", track_write)
    actual = checked(setup.episode, teacher=references, lambda_am=0.7)
    expected = ordinary(setup.episode, teacher=references, lambda_am=0.7)
    actual["loss"].backward()
    expected["loss"].backward()
    for metric in ("loss", "qa_loss", "outer_am"):
        torch.testing.assert_close(actual[metric], expected[metric], rtol=1e-5, atol=1e-6)
    assert actual["write_events"] == expected["write_events"] == 2
    expected_parameters = dict(ordinary.named_parameters())
    for name, parameter in checked.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(
                parameter.grad, expected_parameters[name].grad, rtol=2e-5, atol=2e-6,
                msg=lambda message: f"{name}: {message}",
            )
    for suffix in ("initial_slots", "raw_step_size", "decoder.input.weight", "lora_B.default.weight"):
        assert any(name.endswith(suffix) and parameter.grad.abs().sum() > 0
                   for name, parameter in checked.named_parameters()), suffix
    assert inner_inputs
    for tensor in inner_inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_engine_dev_and_teacher_free_inference(tiny_setup):
    setup = tiny_setup
    model = OnlineMemoryLM(setup.config.model)
    teacher = RuntimeTeacher(setup.teacher, setup.config, setup.cache)
    engine = TrainingEngine(model, setup.config.training)
    episodes = (setup.episode, replace(setup.episode, id="document:1", target_ids=(10, 1)))
    metrics = engine.train_batch(episodes, teacher, torch.Generator().manual_seed(7), 20)
    assert metrics["samples"] == 2
    assert metrics["write_events"] == 4
    assert metrics["target_tokens"] == 5
    assert metrics["outer_am"] > 0 and metrics["grad_norm"] > 0
    parameters = model.trainable_state_dict()
    dev = evaluate_dev(model, episodes, teacher, 1.0, torch.Generator().manual_seed(8))
    assert dev["write_events"] == 4 and dev["outer_am"] > 0
    for name, value in model.trainable_state_dict().items():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        inference = model(setup.episode, teacher=None, differentiable=False)
        generated = model.generate(setup.episode.context_ids, setup.episode.question_ids, 3)
    assert not inference["loss"].requires_grad
    assert inference["outer_am"] == 0
    assert generated["write_events"] == 2
    assert all(parameter.grad is None for parameter in model.parameters())


def test_true_batch_preserves_per_sample_losses_and_full_meta_gradients(tiny_setup, monkeypatch):
    episodes, references = _mixed_episodes_and_references(tiny_setup)
    config = replace(tiny_setup.config.model, inner_steps=3)
    batched = OnlineMemoryLM(config)
    serial = OnlineMemoryLM(replace(config, gradient_checkpointing=False))
    serial.load_trainable_state_dict(batched.trainable_state_dict())
    batch_sizes, inner_inputs = [], []
    advance = batched._advance

    def track_advance(token_ids, state, differentiable):
        batch_sizes.append(len(token_ids))
        return advance(token_ids, state, differentiable)

    monkeypatch.setattr(batched, "_advance", track_advance)
    for memory in batched.memory_layers:
        def track_write(old_memory, teacher_kv, queries, _write=memory.write, **kwargs):
            for tensor in (*teacher_kv[:2], queries):
                assert tensor.requires_grad
                tensor.retain_grad()
                inner_inputs.append(tensor)
            return _write(old_memory, teacher_kv, queries, **kwargs)

        monkeypatch.setattr(memory, "write", track_write)
    actual = batched.forward_batch(
        episodes, teachers=[FixedReferences(references) for _ in episodes], lambda_am=0.7,
    )
    expected = [serial(episode, teacher=FixedReferences(references), lambda_am=0.7)
                for episode in episodes]
    assert batch_sizes[0] == 3 and 2 in batch_sizes
    assert [row["write_events"] for row in actual] == [2, 1, 0]
    assert actual[-1]["outer_am"] == 0
    for row, reference in zip(actual, expected):
        for metric in ("loss", "qa_loss", "outer_am"):
            torch.testing.assert_close(row[metric], reference[metric], rtol=2e-5, atol=2e-6)
    torch.stack([row["loss"] for row in actual]).sum().backward()
    torch.stack([row["loss"] for row in expected]).sum().backward()
    expected_parameters = dict(serial.named_parameters())
    for name, parameter in batched.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, expected_parameters[name].grad,
                                       rtol=2e-5, atol=2e-6, msg=lambda message: f"{name}: {message}")
    assert inner_inputs
    for tensor in inner_inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_microbatch_tail_preserves_optimizer_update(tiny_setup, monkeypatch):
    episodes, references = _mixed_episodes_and_references(tiny_setup)
    config = replace(tiny_setup.config.model, inner_steps=3)
    serial = OnlineMemoryLM(replace(config, gradient_checkpointing=False))
    batched = OnlineMemoryLM(config)
    batched.load_trainable_state_dict(serial.trainable_state_dict())
    # 同一全局 batch 的三样本尾批：microbatch=2 时实际执行 2 + 1。
    serial_engine = TrainingEngine(serial, replace(
        tiny_setup.config.training, micro_batch_size=1, gradient_accumulation_steps=4,
    ))
    batched_engine = TrainingEngine(batched, replace(
        tiny_setup.config.training, micro_batch_size=2, gradient_accumulation_steps=2,
    ))
    groups = []
    forward_batch = batched.forward_batch

    def track_batch(episodes, **kwargs):
        groups.append(len(episodes))
        return forward_batch(episodes, **kwargs)

    monkeypatch.setattr(batched, "forward_batch", track_batch)
    teacher = FixedTeacher(references)
    expected = serial_engine.train_batch(episodes, teacher, torch.Generator().manual_seed(7), 20)
    actual = batched_engine.train_batch(episodes, teacher, torch.Generator().manual_seed(7), 20)
    assert groups == [2]
    assert actual["samples"] == 3 and actual["write_events"] == 3 and actual["target_tokens"] == 6
    assert actual.keys() == expected.keys()
    for metric, value in actual.items():
        assert value == pytest.approx(expected[metric], rel=2e-5, abs=2e-6), metric
    expected_parameters = serial.trainable_state_dict()
    for name, value in batched.trainable_state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=2e-5, atol=2e-6,
                                   msg=lambda message: f"{name}: {message}")
