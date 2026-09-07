from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ReaderOutput:
    target_logits: Tensor
    token_nll: Tensor

    def __post_init__(self) -> None:
        if self.target_logits.ndim != 2:
            raise ValueError("target_logits must have shape [target_length, vocab_size]")
        if self.token_nll.ndim != 1:
            raise ValueError("token_nll must have shape [target_length]")
        if self.target_logits.shape[0] != self.token_nll.shape[0]:
            raise ValueError("target_logits and token_nll must have the same target length")
        if self.target_logits.shape[0] <= 0 or self.target_logits.shape[1] <= 0:
            raise ValueError("reader output dimensions must be non-zero")

    @property
    def target_length(self) -> int:
        return self.target_logits.shape[0]

    @property
    def mean_nll(self) -> Tensor:
        return self.token_nll.mean()


def gold_token_nll(target_logits: Tensor, target_ids: Tensor) -> Tensor:
    _validate_target_pair(target_logits, target_ids)
    return functional.cross_entropy(target_logits.float(), target_ids, reduction="none")


def build_reader_output(target_logits: Tensor, target_ids: Tensor) -> ReaderOutput:
    return ReaderOutput(
        target_logits=target_logits, token_nll=gold_token_nll(target_logits, target_ids)
    )


def teacher_student_kl(
    teacher_logits: Tensor,
    student_logits: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    _validate_logit_pair(teacher_logits, student_logits)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_log_probs = functional.log_softmax(teacher_logits.float() / temperature, dim=-1)
    student_log_probs = functional.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(
        dim=-1
    ).mean() * temperature**2


def symmetric_token_kl(first_logits: Tensor, second_logits: Tensor) -> Tensor:
    _validate_logit_pair(first_logits, second_logits)
    first_log_probs = functional.log_softmax(first_logits.float(), dim=-1)
    second_log_probs = functional.log_softmax(second_logits.float(), dim=-1)
    first_probs = first_log_probs.exp()
    second_probs = second_log_probs.exp()
    first_to_second = (first_probs * (first_log_probs - second_log_probs)).sum(dim=-1)
    second_to_first = (second_probs * (second_log_probs - first_log_probs)).sum(dim=-1)
    return 0.5 * (first_to_second + second_to_first).mean()


def _validate_target_pair(target_logits: Tensor, target_ids: Tensor) -> None:
    if target_logits.ndim != 2:
        raise ValueError("target_logits must have shape [target_length, vocab_size]")
    if target_ids.ndim != 1:
        raise ValueError("target_ids must have shape [target_length]")
    if target_logits.shape[0] != target_ids.shape[0] or target_ids.shape[0] <= 0:
        raise ValueError("target logits and IDs must have the same non-zero target length")
    if target_ids.dtype != torch.long:
        raise TypeError("target_ids must use torch.long")


def _validate_logit_pair(first_logits: Tensor, second_logits: Tensor) -> None:
    if first_logits.ndim != 2 or second_logits.ndim != 2:
        raise ValueError("logits must have shape [target_length, vocab_size]")
    if first_logits.shape != second_logits.shape or first_logits.shape[0] <= 0:
        raise ValueError("logit tensors must have the same non-zero shape")
