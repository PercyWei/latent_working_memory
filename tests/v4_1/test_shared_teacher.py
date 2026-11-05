"""共享冻结基座时，teacher 的模块、预测和 student 更新仍保持独立语义。"""

from dataclasses import replace

import pytest
import torch

from latent_working_memory.v4_1.modeling.model import OnlineMemoryLM
from latent_working_memory.v4_1.teacher.replay import load_teacher
from latent_working_memory.v4_1.teacher.runtime import RuntimeTeacher
from latent_working_memory.v4_1.training.engine import TrainingEngine


@pytest.mark.parametrize("dtype", ("float32", "bfloat16"))
def test_shared_teacher_preserves_native_modules_and_predictions(tiny_setup, dtype):
    config = replace(tiny_setup.config.model, backbone_dtype=dtype)
    student = OnlineMemoryLM(config)
    independent = load_teacher(config, torch.device("cpu"))
    teacher = load_teacher(config, torch.device("cpu"), student.backbone)

    assert teacher is not student.backbone and teacher.config is not student.backbone.config
    assert teacher.config._attn_implementation == "sdpa"
    assert student.backbone.config._attn_implementation == "eager"
    assert teacher.model.rotary_emb is not student.backbone.model.rotary_emb
    assert teacher.model.rotary_emb.original_inv_freq.device.type == "cpu"
    assert not any("lora_" in name for name, _ in teacher.named_parameters())
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in teacher.parameters())
    assert all(not tensor.is_meta for tensor in (*teacher.parameters(), *teacher.buffers()))
    frozen_storage = {parameter.untyped_storage().data_ptr()
                      for parameter in student.backbone.parameters() if not parameter.requires_grad}
    teacher_storage = {parameter.untyped_storage().data_ptr() for parameter in teacher.parameters()}
    assert teacher_storage == frozen_storage
    for teacher_layer, student_layer in zip(teacher.model.layers, student.backbone.model.layers):
        assert teacher_layer.self_attn is not student_layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            native = getattr(teacher_layer.self_attn, name)
            adapted = getattr(student_layer.self_attn, name)
            assert not hasattr(native, "lora_A")
            assert native.weight is adapted.base_layer.weight
    with torch.no_grad():
        ids = torch.tensor([tiny_setup.episode.context_ids])
        expected = independent(ids, use_cache=False).logits
        actual = teacher(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_teacher_preserves_distill_meta_gradient_optimizer_update(tiny_setup):
    config = replace(
        tiny_setup.config,
        model=replace(tiny_setup.config.model, inner_objective="output_only_mass_calibrated"),
        training=replace(tiny_setup.config.training, outer_write_obj="distill"),
    )
    shared_student = OnlineMemoryLM(config.model)
    ordinary_student = OnlineMemoryLM(config.model)
    ordinary_student.load_trainable_state_dict(shared_student.trainable_state_dict())
    teacher = load_teacher(config.model, torch.device("cpu"), shared_student.backbone)
    teacher_weights = {name: tensor.clone() for name, tensor in teacher.state_dict().items()}
    shared_runtime = RuntimeTeacher(teacher, config, tiny_setup.cache)
    ordinary_runtime = RuntimeTeacher(tiny_setup.teacher, config, tiny_setup.cache)
    shared_engine = TrainingEngine(shared_student, config.training)
    ordinary_engine = TrainingEngine(ordinary_student, config.training)
    actual = shared_engine.train_batch(
        (tiny_setup.episode,), shared_runtime, torch.Generator().manual_seed(19), 1,
    )
    expected = ordinary_engine.train_batch(
        (tiny_setup.episode,), ordinary_runtime, torch.Generator().manual_seed(19), 1,
    )
    for name in ("loss", "qa_loss", "outer_am", "distill_kl", "grad_norm"):
        assert actual[name] == pytest.approx(expected[name], rel=1e-6, abs=1e-6), name
    expected_parameters = dict(ordinary_student.named_parameters())
    for name, parameter in shared_student.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, expected_parameters[name].grad,
                                       rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(parameter, expected_parameters[name], rtol=2e-5, atol=2e-6)
    for name, tensor in teacher.state_dict().items():
        torch.testing.assert_close(tensor, teacher_weights[name], rtol=0, atol=0)
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in teacher.parameters())
