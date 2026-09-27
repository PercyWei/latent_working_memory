"""不等长目标与不同压缩次数下的 NLL 和配对对照统计。"""

import pytest

from latent_working_memory.v2.pretrain.evaluation import summarize


def test_lm_weighting_and_matched_control():
    records = [
        {
            "depth": 2,
            "ratio_bin": 4,
            "multi_compression": [
                {"round": 1, "ae": 2.0, "lm": 1.0, "ae_tokens": 3, "lm_tokens": 2},
                {"round": 2, "ae": 4.0, "lm": 3.0, "ae_tokens": 7, "lm_tokens": 4},
            ],
        },
        {
            "depth": 1,
            "ratio_bin": 3,
            "multi_compression": [
                {"round": 1, "ae": 5.0, "lm": 2.0, "ae_tokens": 5, "lm_tokens": 6},
            ],
        },
    ]
    for row in records:
        row["single_compression"] = [
            dict(r, ae=r["ae"] / 2, lm=r["lm"] / 2) for r in row["multi_compression"]
        ]
    metrics = summarize(records)
    assert metrics["multi_compression/all/lm_tokens"] == 12
    assert metrics["multi_compression/all/lm_nll"] == pytest.approx(26 / 12)
    assert metrics["multi_compression/round/1/lm_nll"] == pytest.approx(14 / 8)
    assert not any("ppl" in key for key in metrics)
    assert metrics["multi_compression/trajectory_lm"] == 2.0
    assert metrics["final_round_single_compression_lm"] == 1.25
    assert metrics["final_round_multi_compression_lm"] == 2.5
    assert metrics["final_round_compression_gap_lm"] == 1.25
    assert metrics["multi_compression/trajectory_ae"] == 4.0
    assert metrics["final_round_single_compression_ae"] == 2.25
    assert metrics["final_round_multi_compression_ae"] == 4.5
    assert metrics["final_round_compression_gap_ae"] == 2.25
    assert metrics["single_compression/trajectory_ae"] == 2.0
    assert metrics["single_compression/all/lm_nll"] == pytest.approx(13 / 12)
    assert "final_ae" not in metrics and "final_lm" not in metrics
    assert all(
        key.startswith(("multi_compression/", "single_compression/", "final_round_"))
        for key in metrics
    )


def test_final_summary_omits_final_round_metrics_and_keeps_both_paths():
    records = [
        {
            "depth": 2,
            "ratio_bin": 4,
            "multi_compression": [
                {"round": 1, "ae": 2.0, "lm": 1.0, "ae_tokens": 3, "lm_tokens": 2},
                {"round": 2, "ae": 4.0, "lm": 3.0, "ae_tokens": 7, "lm_tokens": 4},
            ],
            "single_compression": [
                {"round": 1, "ae": 1.0, "lm": 0.5, "ae_tokens": 3, "lm_tokens": 2},
                {"round": 2, "ae": 2.0, "lm": 1.5, "ae_tokens": 7, "lm_tokens": 4},
            ],
        }
    ]
    metrics = summarize(records, include_final_comparison=False)
    assert metrics["multi_compression/trajectory_ae"] == 3.0
    assert metrics["single_compression/round/2/ae_nll"] == 2.0
    assert metrics["single_compression/trajectory_ae"] == 1.5
    for task in ("ae", "lm"):
        assert f"final_round_single_compression_{task}" not in metrics
        assert f"final_round_multi_compression_{task}" not in metrics
        assert f"final_round_compression_gap_{task}" not in metrics
