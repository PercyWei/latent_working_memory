"""不等长目标与不同压缩次数下的 NLL 和配对对照统计。"""

import pytest

from latent_working_memory.v2.pretrain.evaluation import summarize


def test_lm_weighting_and_matched_control():
    records = [
        {
            "depth": 2,
            "ratio_bin": 4,
            "rounds": [
                {"round": 1, "ae": 2.0, "lm": 1.0, "ae_tokens": 3, "lm_tokens": 2},
                {"round": 2, "ae": 4.0, "lm": 3.0, "ae_tokens": 7, "lm_tokens": 4},
            ],
            "one_shot": {"ae": 3.0, "lm": 0.5},
        },
        {
            "depth": 1,
            "ratio_bin": 3,
            "rounds": [
                {"round": 1, "ae": 5.0, "lm": 2.0, "ae_tokens": 5, "lm_tokens": 6},
            ],
            "one_shot": {"ae": 4.0, "lm": 1.0},
        },
    ]
    metrics = summarize(records)
    assert metrics["all/lm_tokens"] == 12
    assert metrics["all/lm_nll"] == pytest.approx(26 / 12)
    assert metrics["round/1/lm_nll"] == pytest.approx(14 / 8)
    assert not any("ppl" in key for key in metrics)
    assert metrics["trajectory_lm"] == 2.0
    assert metrics["one_shot_lm"] == 0.75
    assert metrics["final_minus_one_shot_lm"] == 1.75
    assert metrics["trajectory_ae"] == 4.0
    assert metrics["one_shot_ae"] == 3.5
    assert metrics["final_minus_one_shot_ae"] == 1.0
    assert "final_ae" not in metrics and "final_lm" not in metrics
