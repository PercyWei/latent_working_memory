from __future__ import annotations

import math

import pytest

from cdic_repro.experiments.generation_metrics import score_generation_records


def test_exact_predictions_receive_perfect_generation_scores() -> None:
    metrics = score_generation_records(
        [
            {
                "prediction": "one two three four",
                "reference": "one two three four",
                "loss": math.log(2.0),
                "loss_tokens": 4,
            }
        ]
    )

    assert metrics["ppl"] == pytest.approx(2.0)
    assert metrics["bleu"] == pytest.approx(1.0)
    assert metrics["rouge_1"] == pytest.approx(1.0)
    assert metrics["rouge_2"] == pytest.approx(1.0)
    assert metrics["rouge_l"] == pytest.approx(1.0)


def test_generation_metrics_report_partial_overlap() -> None:
    metrics = score_generation_records(
        [
            {
                "prediction": "the red fox runs quickly",
                "reference": "the red fox sleeps quietly",
                "loss": 1.0,
                "loss_tokens": 5,
            },
            {
                "prediction": "a b c d",
                "reference": "a b c d",
                "loss": 2.0,
                "loss_tokens": 5,
            },
        ]
    )

    assert 0.0 < metrics["bleu"] < 1.0
    assert 0.0 < metrics["rouge_2"] < metrics["rouge_1"] < 1.0
    assert metrics["token_weighted_loss"] == pytest.approx(1.5)
