"""运行时 teacher 参照不改变 student 目标、元梯度与推理边界。"""

from contextlib import nullcontext
from dataclasses import replace
import weakref

import pytest
import torch
import torch.utils.checkpoint as checkpoint_module
from transformers.cache_utils import DynamicCache

from latent_working_memory.v4_1.data import compression_boundaries
from latent_working_memory.v4_1.modeling import am as am_module
from latent_working_memory.v4_1.modeling import memory as memory_module
from latent_working_memory.v4_1.modeling import model as model_module
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


@pytest.mark.parametrize("inner_objective", ("am", "output_only_mass_calibrated"))
@pytest.mark.parametrize("backbone_dtype", ("float32", "bfloat16"))
def test_per_gradient_checkpoint_preserves_multistep_meta_gradients(
    tiny_setup, monkeypatch, inner_objective, backbone_dtype,
):
    setup = tiny_setup
    config = replace(setup.config.model, inner_steps=3, inner_objective=inner_objective,
                     backbone_dtype=backbone_dtype)
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
    inner_inputs, teacher_checkpoint_sources, calibration_checkpoint_sources = [], [], []
    memory_checkpoint = memory_module.checkpoint
    model_checkpoint = model_module.checkpoint

    def track_teacher_sources(function, *args, **kwargs):
        if function is memory_module.read_teacher:
            teacher_checkpoint_sources.append(tuple(tensor.dtype for tensor in args))
        return memory_checkpoint(function, *args, **kwargs)

    def track_calibration_sources(function, *args, **kwargs):
        if function is model_module.calibrate_memory_bias:
            calibration_checkpoint_sources.append((args[0].dtype, args[5].dtype))
        return model_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(memory_module, "checkpoint", track_teacher_sources)
    monkeypatch.setattr(model_module, "checkpoint", track_calibration_sources)
    for memory in checked.memory_layers:
        def track_write(old_memory, teacher_kv, pending_kv, queries, _write=memory.write, **kwargs):
            for tensor in (*pending_kv, queries):
                assert tensor.requires_grad
                assert tensor.dtype == torch.float32
                tensor.retain_grad()
                inner_inputs.append(tensor)
            return _write(old_memory, teacher_kv, pending_kv, queries, **kwargs)

        monkeypatch.setattr(memory, "write", track_write)
    # 旧路径在 write 外物化 FP32 teacher/query，作为非 checkpoint 对照。
    for memory in ordinary.memory_layers:
        def materialized_write(old_memory, teacher_kv, pending_kv, queries,
                               _write=memory.write, **kwargs):
            key = torch.cat((teacher_kv[0].float(), pending_kv[0].float()), dim=2)
            value = torch.cat((teacher_kv[1].float(), pending_kv[1].float()), dim=2)
            bias = torch.cat((teacher_kv[2].float(), teacher_kv[2].new_zeros(
                (*teacher_kv[2].shape[:-1], pending_kv[0].shape[2]),
            )), dim=-1)
            empty = key[:, :, :0]
            return _write(old_memory, (key, value, bias), (empty, empty), queries.float(), **kwargs)

        monkeypatch.setattr(memory, "write", materialized_write)
    actual_diagnostics, expected_diagnostics = [], []
    actual = checked(setup.episode, teacher=references, outer_write_obj_weight=0.7,
                     inner_diagnostics=actual_diagnostics)
    expected = ordinary(setup.episode, teacher=references, outer_write_obj_weight=0.7,
                        inner_diagnostics=expected_diagnostics)
    actual["loss"].backward()
    expected["loss"].backward()
    assert actual.keys() == expected.keys()
    assert actual_diagnostics == expected_diagnostics
    for metric in actual:
        torch.testing.assert_close(actual[metric], expected[metric], rtol=1e-5, atol=1e-6)
    assert actual["write_events"] == expected["write_events"] == 2
    dtype = torch.float32
    assert len(teacher_checkpoint_sources) == 2 * len(checked.memory_layers)
    assert all(dtypes[0] == dtypes[4] == dtypes[5] == dtype
               for dtypes in teacher_checkpoint_sources)
    if inner_objective == "output_only_mass_calibrated":
        assert calibration_checkpoint_sources == [(dtype, dtype)] * (2 * len(checked.memory_layers))
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


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_teacher_source_inputs_match_materialized_fp32_outputs_and_meta_gradients(dtype):
    generator = torch.Generator().manual_seed(73)
    specs = (
        ((2, 3, 4, 2), dtype),       # queries
        ((2, 2, 2, 2), torch.float32),  # old memory key
        ((2, 2, 2, 2), torch.float32),  # old memory value
        ((2, 2, 2), torch.float32),     # old memory bias
        ((2, 2, 5, 2), dtype),       # pending key
        ((2, 2, 5, 2), dtype),       # pending value
        ((2, 2, 2, 2), torch.float32),  # updated memory key
        ((2, 2, 2), torch.float32),     # updated memory bias
    )
    actual_inputs = [torch.randn(shape, dtype=kind, generator=generator).requires_grad_()
                     for shape, kind in specs]
    expected_inputs = [tensor.detach().clone().requires_grad_() for tensor in actual_inputs]
    query, old_key, old_value, old_bias, pending_key, pending_value, key, bias = actual_inputs
    query, pending_key, pending_value = query.float(), pending_key.float(), pending_value.float()
    actual = (*am_module.read_teacher(query, old_key, old_value, old_bias, pending_key, pending_value),
              am_module.calibrate_memory_bias(query, key, bias, old_key, old_bias, pending_key, 0.1))
    query, old_key, old_value, old_bias, pending_key, pending_value, key, bias = expected_inputs
    query, pending_key, pending_value = query.float(), pending_key.float(), pending_value.float()
    teacher_key = torch.cat((old_key.float(), pending_key), dim=2)
    teacher_value = torch.cat((old_value.float(), pending_value), dim=2)
    teacher_bias = torch.cat((old_bias.float(), old_bias.new_zeros((2, 2, 5))), dim=-1)
    expected_read = am_module.local_read(query.float(), teacher_key, teacher_value, teacher_bias)
    teacher_mass = am_module._attention_scores(query.float(), teacher_key, teacher_bias).logsumexp(-1)
    memory_mass = am_module._attention_scores(query.float(), key, bias).logsumexp(-1)
    scales = teacher_mass.std(dim=-3, correction=0).clamp_min(0.1).detach()
    weights = scales.double().square().reciprocal()
    gap = (teacher_mass.double() - memory_mass.double()).mean(dim=-3)
    offset = (weights * gap).sum(dim=-1) / weights.sum(dim=-1)
    expected = (*expected_read, bias + offset.to(bias.dtype).unsqueeze(-1))
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(
        sum(value.square().sum() for value in actual), actual_inputs, create_graph=True,
    )
    expected_gradient = torch.autograd.grad(
        sum(value.square().sum() for value in expected), expected_inputs, create_graph=True,
    )
    for value, reference in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    actual_meta_gradient = torch.autograd.grad(sum(value.sum() for value in actual_gradient), actual_inputs)
    expected_meta_gradient = torch.autograd.grad(sum(value.sum() for value in expected_gradient), expected_inputs)
    for value, reference in zip(actual_meta_gradient, expected_meta_gradient):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_checkpoint_retains_bfloat16_sources_without_fp32_cast_storage():
    raw = tuple(torch.randn(5, dtype=torch.bfloat16).requires_grad_() for _ in range(3))
    reference = tuple(value.detach().clone().requires_grad_() for value in raw)
    cast = tuple(value.float() for value in raw)
    cast_references = tuple(weakref.ref(value) for value in cast)

    def function(query, key, value):
        return (query.sin() + key.cos() + value.square()).sum()

    with model_module._save_cast_sources(tuple(zip(cast, raw))):
        actual = checkpoint_module.checkpoint(function, *cast, use_reentrant=False)
    del cast
    assert all(reference() is None for reference in cast_references)
    expected = function(*(value.float() for value in reference))
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for value, expected in zip(raw, reference):
        torch.testing.assert_close(value.grad, expected.grad, rtol=0, atol=0)


@pytest.mark.parametrize("inner_objective", ("am", "output_only_mass_calibrated"))
def test_engine_dev_and_teacher_free_inference(tiny_setup, inner_objective):
    setup = tiny_setup
    model = OnlineMemoryLM(replace(setup.config.model, inner_objective=inner_objective))
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


@pytest.mark.parametrize("inner_objective", ("am", "output_only_mass_calibrated"))
def test_true_batch_preserves_per_sample_losses_and_full_meta_gradients(
    tiny_setup, monkeypatch, inner_objective,
):
    episodes, references = _mixed_episodes_and_references(tiny_setup)
    config = replace(tiny_setup.config.model, inner_steps=3, inner_objective=inner_objective)
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
        def track_write(old_memory, teacher_kv, pending_kv, queries, _write=memory.write, **kwargs):
            for tensor in (*pending_kv, queries):
                assert tensor.requires_grad
                tensor.retain_grad()
                inner_inputs.append(tensor)
            return _write(old_memory, teacher_kv, pending_kv, queries, **kwargs)

        monkeypatch.setattr(memory, "write", track_write)
    actual = batched.forward_batch(
        episodes, teachers=[FixedReferences(references) for _ in episodes], outer_write_obj_weight=0.7,
    )
    expected = [serial(episode, teacher=FixedReferences(references), outer_write_obj_weight=0.7)
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


def test_output_only_diagnostic_vjps_release_teacher_checkpoint_cache(tiny_setup, monkeypatch):
    config = replace(tiny_setup.config.model, inner_steps=3,
                     inner_objective="output_only_mass_calibrated")
    model = OnlineMemoryLM(config)
    _, references = _mixed_episodes_and_references(tiny_setup)
    teacher_frames = []
    original_checkpoint = memory_module.checkpoint
    frame_type = checkpoint_module._CheckpointFrame

    def capture_frame(*args, **kwargs):
        frame = frame_type(*args, **kwargs)
        teacher_frames.append(weakref.ref(frame))
        return frame

    def track_teacher_checkpoint(function, *args, **kwargs):
        if function is not memory_module.read_teacher:
            return original_checkpoint(function, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(checkpoint_module, "_CheckpointFrame", capture_frame)
            return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(memory_module, "checkpoint", track_teacher_checkpoint)
    output = model(tiny_setup.episode, teacher=FixedReferences(references))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(teacher_frames) == output["write_events"] * len(model.memory_layers)
    for name in ("qa_loss", "outer_am"):
        torch.autograd.grad(output[name], parameters, retain_graph=True, allow_unused=True)
        for reference in teacher_frames:
            frame = reference()
            assert frame is not None
            assert all(not cache for cache in frame.recomputed.values())
    output["loss"].backward()
