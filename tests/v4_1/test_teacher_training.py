"""运行时 teacher 参照不改变 student 目标、元梯度与推理边界。"""

from dataclasses import replace

import torch

from latent_working_memory.v4_1.data import compression_boundaries
from latent_working_memory.v4_1.modeling.model import OnlineMemoryLM
from latent_working_memory.v4_1.teacher.runtime import RuntimeTeacher
from latent_working_memory.v4_1.training.engine import TrainingEngine, evaluate_dev


class FixedReferences:
    """测试对照：预先物化相同采样位置的 q/o/z，模拟原张量缓存路径。"""

    def __init__(self, references):
        self.references = references

    def reference(self, write_index, processed_end, compressed_end, generator=None):
        return self.references[(write_index, processed_end, compressed_end)]


def test_runtime_teacher_matches_materialized_loss_and_meta_gradients(tiny_setup):
    setup = tiny_setup
    teacher = RuntimeTeacher(setup.teacher, setup.config, setup.cache)
    boundaries = compression_boundaries(
        len(setup.episode.context_ids), setup.config.model.pending_size, setup.config.model.recent_size
    )
    generator = torch.Generator().manual_seed(19)
    with teacher.episode(setup.episode) as trajectory:
        references = {
            boundary: trajectory.reference(*boundary, generator=generator)
            for boundary in boundaries
        }
        assert trajectory.prefix.get_seq_length() == len(setup.episode.context_ids)
    assert trajectory.prefix is None
    assert all(not tensor.requires_grad for reference in references.values()
               for tensor in reference.values())
    runtime = OnlineMemoryLM(setup.config.model)
    materialized = OnlineMemoryLM(setup.config.model)
    materialized.load_trainable_state_dict(runtime.trainable_state_dict())
    teacher_weights = {name: tensor.clone() for name, tensor in setup.teacher.state_dict().items()}
    with teacher.episode(setup.episode) as trajectory:
        runtime_output = runtime(
            setup.episode, teacher=trajectory, generator=torch.Generator().manual_seed(19),
        )
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


def test_per_loss_checkpoint_preserves_multistep_meta_gradients(tiny_setup, monkeypatch):
    setup = tiny_setup
    config = replace(setup.config.model, inner_steps=3)
    teacher = RuntimeTeacher(setup.teacher, setup.config, setup.cache)
    boundaries = compression_boundaries(
        len(setup.episode.context_ids), config.pending_size, config.recent_size,
    )
    assert len(boundaries) == 2
    with teacher.episode(setup.episode) as trajectory:
        generator = torch.Generator().manual_seed(19)
        references = FixedReferences({
            boundary: trajectory.reference(*boundary, generator=generator)
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
    metrics = engine.train_batch(episodes, teacher, torch.Generator().manual_seed(7))
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
