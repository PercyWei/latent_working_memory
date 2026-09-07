from __future__ import annotations

import pytest
import torch

from latent_working_memory.v1.objectives import (
    build_reader_output,
    symmetric_token_kl,
    teacher_student_kl,
)


def test_reader_output_uses_token_mean_not_mean_of_perplexities() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    targets = torch.tensor([0, 1], dtype=torch.long)
    output = build_reader_output(logits, targets)
    assert output.target_length == 2
    assert output.token_nll.shape == (2,)
    assert output.mean_nll.item() == pytest.approx(output.token_nll.sum().item() / 2)


def test_distillation_losses_compare_answer_relative_logits() -> None:
    teacher = torch.tensor([[3.0, 1.0], [0.0, 2.0]])
    assert teacher_student_kl(teacher, teacher).item() == pytest.approx(0.0, abs=1e-7)
    assert symmetric_token_kl(teacher, teacher).item() == pytest.approx(0.0, abs=1e-7)

    student = torch.tensor([[1.0, 3.0], [2.0, 0.0]])
    assert teacher_student_kl(teacher, student).item() > 0
    assert symmetric_token_kl(teacher, student).item() > 0
