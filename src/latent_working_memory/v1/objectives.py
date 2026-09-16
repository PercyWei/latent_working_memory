from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ReaderOutput:
    token_nll: Tensor

    def __post_init__(self) -> None:
        if self.token_nll.ndim != 1 or self.token_nll.numel() == 0:
            raise ValueError("token_nll must have non-empty shape [target_length]")

    @property
    def target_length(self) -> int:
        return self.token_nll.shape[0]

    @property
    def mean_nll(self) -> Tensor:
        return self.token_nll.mean()


def gold_token_nll(target_logits: Tensor, target_ids: Tensor) -> Tensor:
    _validate_target_pair(target_logits, target_ids)
    return functional.cross_entropy(target_logits.float(), target_ids, reduction="none")


def _validate_target_pair(target_logits: Tensor, target_ids: Tensor) -> None:
    if target_logits.ndim != 2:
        raise ValueError("target_logits must have shape [target_length, vocab_size]")
    if target_ids.ndim != 1:
        raise ValueError("target_ids must have shape [target_length]")
    if target_logits.shape[0] != target_ids.shape[0] or target_ids.shape[0] <= 0:
        raise ValueError("target logits and IDs must have the same non-zero target length")
    if target_ids.dtype != torch.long:
        raise TypeError("target_ids must use torch.long")
