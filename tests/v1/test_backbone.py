from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from latent_working_memory.v1.backbone import (
    LatentMemoryBackbone,
    QuestionAnswerTokens,
    tokenize_question_answer,
)
from latent_working_memory.v1.data import EncoderCell
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.objectives import teacher_student_kl


class RecordingTokenizer:
    eos_token_id = 2

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        self.inputs.append(text)
        return [3 + index for index, _ in enumerate(text.split(), start=0)]


def _backbone() -> LatentMemoryBackbone:
    torch.manual_seed(19)
    base_model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    )
    return LatentMemoryBackbone(
        base_model=base_model,
        bos_token_id=1,
        eos_token_id=2,
        d_mem=8,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules=("q_proj", "v_proj"),
        lora_dropout=0.0,
    )


def test_question_and_answer_are_tokenized_as_separate_segments() -> None:
    tokenizer = RecordingTokenizer()
    tokens = tokenize_question_answer(tokenizer, "Where now?", "North")
    assert tokenizer.inputs == ["Question: Where now?\nAnswer:", " North"]
    assert tokens.target_ids[-1] == tokenizer.eos_token_id
    assert tokens.answer_token_count == len(tokens.target_ids) - 1


def test_teacher_and_cell_encoder_disable_reader_adapter() -> None:
    backbone = _backbone()
    cells = (
        EncoderCell((7, 8), source_start=0),
        EncoderCell((9, 10), source_start=2),
    )
    qa = QuestionAnswerTokens(prompt_ids=(11, 12), target_ids=(13, 14, 2))

    first_encoding = backbone.frozen_cell_encoding(cells)
    first_teacher = backbone.teacher_output((7, 8, 9, 10), qa)
    with torch.no_grad():
        for name, parameter in backbone.language_model.named_parameters():
            if "lora_" in name:
                parameter.add_(0.75)

    second_encoding = backbone.frozen_cell_encoding(cells)
    second_teacher = backbone.teacher_output((7, 8, 9, 10), qa)
    assert torch.equal(first_encoding.hidden_states, second_encoding.hidden_states)
    assert torch.equal(first_teacher.target_logits, second_teacher.target_logits)


def test_cell_features_are_invariant_to_update_grouping() -> None:
    backbone = _backbone()
    cells = (
        EncoderCell((7, 8), source_start=4),
        EncoderCell((9, 10), source_start=6),
    )
    joint_encoding = backbone.frozen_cell_encoding(cells)
    separate_encodings = tuple(backbone.frozen_cell_encoding((cell,)) for cell in cells)
    assert torch.equal(
        joint_encoding.hidden_states,
        torch.cat(tuple(encoding.hidden_states for encoding in separate_encodings)),
    )
    assert torch.equal(
        backbone.project_cell_encoding(joint_encoding),
        torch.cat(
            tuple(backbone.project_cell_encoding(encoding) for encoding in separate_encodings)
        ),
    )


def test_answer_relative_alignment_and_student_gradient_path() -> None:
    backbone = _backbone()
    writer = JointMemoryWriter(
        d_mem=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        slot_limit=16,
    )
    cells = (EncoderCell((7, 8, 9, 10), source_start=0),)
    qa = QuestionAnswerTokens(prompt_ids=(11, 12, 15), target_ids=(13, 14, 2))

    frozen = backbone.frozen_cell_encoding(cells)
    features = backbone.project_cell_encoding(frozen)
    memory = writer(writer.initialize_state(2), features, grow_by=0).values
    teacher = backbone.teacher_output((7, 8, 9, 10), qa)
    student = backbone.student_output(memory, qa)

    assert teacher.target_logits.shape == student.target_logits.shape == (3, 64)
    assert not teacher.target_logits.requires_grad
    assert student.target_logits.requires_grad
    assert all(
        not parameter.requires_grad
        for name, parameter in backbone.language_model.named_parameters()
        if "lora_" not in name
    )

    loss = student.mean_nll + 0.1 * teacher_student_kl(teacher.target_logits, student.target_logits)
    loss.backward()
    assert backbone.input_projection.weight.grad is not None
    assert torch.count_nonzero(backbone.input_projection.weight.grad).item() > 0
    assert backbone.memory_projection.weight.grad is not None
    assert torch.count_nonzero(backbone.memory_projection.weight.grad).item() > 0
    assert writer.output_projection.weight.grad is not None
    assert torch.count_nonzero(writer.output_projection.weight.grad).item() > 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for name, parameter in backbone.language_model.named_parameters()
        if "lora_" in name
    )
    assert all(
        parameter.grad is None
        for name, parameter in backbone.language_model.named_parameters()
        if "lora_" not in name
    )


def test_trainable_backbone_state_round_trip_and_greedy_generation() -> None:
    source = _backbone()
    state = source.trainable_state_dict()
    target = _backbone()
    target.load_trainable_state_dict(state)

    qa = QuestionAnswerTokens(prompt_ids=(11, 12), target_ids=(13, 2))
    memory = torch.randn(2, 8)
    assert torch.equal(
        source.student_output(memory, qa).target_logits,
        target.student_output(memory, qa).target_logits,
    )
    generated = target.greedy_student(memory, qa.prompt_ids, max_new_tokens=3)
    assert 1 <= len(generated) <= 3
    teacher_generated = target.greedy_teacher((7, 8, 9, 10), qa.prompt_ids, max_new_tokens=3)
    assert 1 <= len(teacher_generated) <= 3
