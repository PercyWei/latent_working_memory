from __future__ import annotations

import pytest
import torch

from latent_working_memory.v1.objectives import (
    ReaderOutput,
    gold_token_nll,
)


def test_reader_output_uses_token_mean_not_mean_of_perplexities() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    targets = torch.tensor([0, 1], dtype=torch.long)
    output = ReaderOutput(gold_token_nll(logits, targets))
    assert output.target_length == 2
    assert output.token_nll.shape == (2,)
    assert output.mean_nll.item() == pytest.approx(output.token_nll.sum().item() / 2)
