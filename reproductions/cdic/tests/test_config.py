from __future__ import annotations

import pytest

from cdic_repro.config import RetrievalConfig, SupportOrder


def test_paper_retrieval_defaults() -> None:
    config = RetrievalConfig()

    assert config.threshold == 0.8
    assert config.decay == 0.05
    assert config.support_order is SupportOrder.SCORE_DESC
    assert config.max_retrieved is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 1.1}, "threshold"),
        ({"decay": -0.1}, "decay"),
        ({"max_retrieved": 0}, "max_retrieved"),
    ],
)
def test_invalid_retrieval_config_is_rejected(kwargs: dict[str, float | int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RetrievalConfig(**kwargs)
