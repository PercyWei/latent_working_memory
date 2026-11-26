"""A few semantic edge cases; the real adapter/tokenizer is checked by a CPU run."""

import re
from unittest.mock import patch

import pytest

from latent_working_memory.v3_2.exploration.data_statistics import metrics as metric_module
from latent_working_memory.v3_2.exploration.data_statistics.metrics import compute_metrics
from latent_working_memory.v3_2.exploration.data_statistics.schema import (
    Document,
    MetricConfig,
    Question,
)


def test_adjacent_evidence_is_not_a_cut_and_overlap_is_not_double_counted():
    text = "a b c d e f g h"
    offsets = [match.span() for match in re.finditer(r"\S+", text)]
    document = Document(
        "d",
        "fixture",
        "dev",
        text,
        ((0, 7), (8, 15)),
        (
            Question("adjacent", ((0, 7), (8, 15)), "span"),
            Question("overlap", ((2, 9), (6, 13)), "span"),
            Question("missing", (), "span"),
        ),
        {},
        "paragraph",
    )
    result = compute_metrics(document, offsets, MetricConfig((2,), (1,), (None,), 4, (2,)))
    adjacent, overlap, missing = result["qas"]
    assert adjacent["metrics"]["evidence_tokens"] == 8
    assert adjacent["blocks"][0]["evidence_cut"] is False
    assert adjacent["blocks"][0]["evidence_block_count"] == 2
    assert adjacent["metrics"]["number_evidence_background_difference"] is None
    assert overlap["metrics"]["evidence_tokens"] == 6
    assert overlap["blocks"][0]["evidence_cut"] is True
    assert missing["status"] == "missing_evidence_annotation"
    assert missing["metrics"] == {}
    assert "blocks" not in missing and "recent" not in missing
    assert "evidence_cut_rate" not in result["metrics"]


def test_natural_feasibility_uses_exact_block_lengths():
    text = "abcdefghijkl"
    offsets = [(index, index + 1) for index in range(len(text))]
    questions = (Question("q", ((0, 1),), "span"),)
    legal = Document(
        "d", "fixture", "dev", text, ((0, 3), (3, 4), (4, 8), (8, 12)), questions, {}, "paragraph"
    )
    impossible = Document(
        "d", "fixture", "dev", text, ((0, 3), (3, 8), (8, 12)), questions, {}, "paragraph"
    )
    config = MetricConfig((3,), (4,), (4,), 4, (0,))
    assert compute_metrics(legal, offsets, config)["segmentations"][0]["natural_feasible"] is True
    result = compute_metrics(impossible, offsets, config)
    assert result["segmentations"][0]["uniform_feasible"] is True
    assert result["segmentations"][0]["natural_feasible"] is False
    assert result["qas"][0]["recent"] == [{"recent_tokens": 0, "tail_overlap": False}]
    invalid_uniform = compute_metrics(legal, offsets, MetricConfig((4,), (4,), (4,), 4, (0,)))
    assert invalid_uniform["segmentations"][0]["uniform_feasible"] is False
    # QA geometry remains defined when only min/max constraints make a partition infeasible.
    assert invalid_uniform["qas"][0]["blocks"][0]["evidence_cut"] is False
    too_many_blocks = compute_metrics(legal, offsets, MetricConfig((16,), (1,), (None,), 4, (0,)))
    assert too_many_blocks["qas"][0]["blocks"][0] == {
        "blocks": 16,
        "evidence_cut": None,
        "evidence_block_count": None,
    }


def test_zero_density_and_short_windows_are_missing_where_undefined():
    text = '"Hello there." (Hello there.) Plain words only.'
    offsets = [match.span() for match in re.finditer(r"\S+", text)]
    document = Document(
        "d",
        "fixture",
        "dev",
        text,
        ((0, len(text)),),
        (Question("q", ((1, 6),), "span"),),
        {},
        "paragraph",
    )
    result = compute_metrics(document, offsets, MetricConfig((1,), (1,), (None,), 2, (0,)))
    assert result["metrics"]["entity_density_mean"] == 0
    assert result["metrics"]["entity_density_cv"] is None
    assert result["metrics"]["number_density_cv"] is None
    short = compute_metrics(document, offsets, MetricConfig((1,), (1,), (None,), 128, (0,)))
    assert short["metrics"]["window_count"] == 0
    assert short["metrics"]["entity_density_mean"] is None
    assert short["qas"][0]["metrics"]["entity_evidence_background_difference"] is None


def test_mentions_crossing_windows_and_dates_are_counted_once():
    text = "we met New York in 2026-11-26 and paid 12.5%."
    offsets = [match.span() for match in re.finditer(r"\S+", text)]
    document = Document(
        "d",
        "fixture",
        "dev",
        text,
        ((0, len(text)),),
        (Question("q", ((offsets[2][0], offsets[3][1]),), "span"),),
        {},
        "paragraph",
    )
    result = compute_metrics(document, offsets, MetricConfig((1,), (1,), (None,), 3, (0,)))
    assert result["metrics"]["window_count"] == 3
    assert result["metrics"]["entity_density_mean"] == pytest.approx(100 / 9)
    assert result["metrics"]["number_density_mean"] == pytest.approx(200 / 9)
    qa = result["qas"][0]["metrics"]
    assert qa["evidence_window_count"] == 2
    assert qa["entity_evidence_background_difference"] == pytest.approx(100 / 6)


def test_sweep_matches_singletons_and_reuses_density_without_repeating_recent():
    text = "we met New York in 2026 and paid 12 dollars at home"
    offsets = [match.span() for match in re.finditer(r"\S+", text)]
    document = Document(
        "d",
        "fixture",
        "dev",
        text,
        tuple((offsets[start][0], offsets[start + 3][1]) for start in (0, 4, 8)),
        (Question("q", ((offsets[2][0], offsets[3][1]),), "span"),),
        {},
        "paragraph",
    )
    config = MetricConfig((2, 3, 16), (1, 4), (3, None), 4, (0, 2, 256))
    with patch.object(
        metric_module, "_density_windows", wraps=metric_module._density_windows
    ) as density:
        result = compute_metrics(document, offsets, config)
    assert density.call_count == 1
    qa = result["qas"][0]
    assert [row["blocks"] for row in qa["blocks"]] == [2, 3, 16]
    assert qa["recent"] == [
        {"recent_tokens": 0, "tail_overlap": False},
        {"recent_tokens": 2, "tail_overlap": False},
        {"recent_tokens": 256, "tail_overlap": True},
    ]
    assert len(result["segmentations"]) == len(config.segmentations) == 9
    assert not {"evidence_cut", "evidence_block_count", "tail_overlap"} & qa["metrics"].keys()
    assert (
        not {
            "uniform_feasible",
            "natural_feasible",
            "evidence_cut_rate",
            "tail_overlap_rate",
        }
        & result["metrics"].keys()
    )
    for (blocks, minimum, maximum), segmentation in zip(
        config.segmentations, result["segmentations"], strict=True
    ):
        singleton = compute_metrics(
            document, offsets, MetricConfig((blocks,), (minimum,), (maximum,), 4, (2,))
        )
        assert singleton["metrics"] == result["metrics"]
        assert singleton["segmentations"] == [segmentation]
        assert singleton["qas"][0]["metrics"] == qa["metrics"]
        assert singleton["qas"][0]["blocks"] == [
            next(row for row in qa["blocks"] if row["blocks"] == blocks)
        ]
        assert singleton["qas"][0]["recent"] == [qa["recent"][1]]


def test_config_sweep_validation_and_skipped_combinations():
    default = MetricConfig()
    assert default.blocks == (4, 8, 16)
    assert default.min_block_tokens == (32, 128, 256)
    assert default.max_block_tokens == (512, 1024, 2048, None)
    assert default.recent_tokens == (0, 128, 256, 512)
    assert len(default.segmentations) == 36
    assert default.skipped_segmentations == ()
    config = MetricConfig((2,), (4, 8), (3, 6, None), 128, (0, 4))
    assert config.segmentations == ((2, 4, 6), (2, 4, None), (2, 8, None))
    assert config.skipped_segmentations == ((2, 4, 3), (2, 8, 3), (2, 8, 6))
    for field in ("blocks", "min_block_tokens", "max_block_tokens", "recent_tokens"):
        with pytest.raises(ValueError, match="nonempty"):
            MetricConfig(**{field: ()})
        with pytest.raises(ValueError, match="duplicate"):
            MetricConfig(**{field: (1, 1)})
    for field in ("blocks", "min_block_tokens", "max_block_tokens"):
        with pytest.raises(ValueError, match="positive"):
            MetricConfig(**{field: (0,)})
    with pytest.raises(ValueError, match="nonnegative"):
        MetricConfig(recent_tokens=(-1,))
    with pytest.raises(ValueError, match="no valid segmentation"):
        MetricConfig(min_block_tokens=(4,), max_block_tokens=(3,))
    with pytest.raises(ValueError, match="window_tokens must be positive"):
        MetricConfig(window_tokens=0)
    with pytest.raises(TypeError, match="must be a tuple"):
        MetricConfig(blocks=4)
