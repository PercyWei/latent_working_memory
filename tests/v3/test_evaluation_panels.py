import json
from types import SimpleNamespace

import pytest
import swanlab

from latent_working_memory.v3.evaluate import evaluate
from latent_working_memory.v3.config import DYNAMIC_METHODS
from latent_working_memory.v3.tracking import (
    METHOD_COLORS,
    configure_training_metrics,
    evaluation_media,
)

from .test_evaluate import Task, trajectory


def test_dashboard_registers_training_curves_and_quality_charts_table_and_retention(tmp_path):
    names = set()
    run = SimpleNamespace(define_metric=lambda name, **options: names.add(name))
    for method, stage in (
        ("dynamic", "pretrain"),
        ("autocompressors", "pretrain"),
        ("autocompressors", "lm"),
        ("icae_single", "qa"),
        ("memory_change", "warmup"),
        ("information_loss", "policy"),
    ):
        configure_training_metrics(run, method, stage)
    summary = evaluate(
        Task(), [trajectory()], tmp_path, "test", 8,
        metadata={"global_step": 37, "stage_step": 37},
    )
    names.update(evaluation_media([summary]))
    original = {
        "train/grad_norm",
        "train/slots_final",
        "dev/slots_final",
        "resources/optimizer_step_seconds",
        "resources/peak_memory_allocated_gib",
        "train/ae_lm_loss",
        "dev/ae_lm_loss",
        "train/lm_loss",
        "dev/lm_loss",
        "train/qa_loss",
        "dev/qa_loss",
        "evaluation/nll",
    }
    assert names - original == {
        "train/stage",
        "train/epoch",
        "dev/qa_old_nll",
        "dev/qa_new_nll",
        "evaluation/em",
        "evaluation/f1",
        "evaluation/nll_retention/step_37",
        "evaluation/statistics",
    }


def test_evaluation_separates_quality_charts_and_keeps_retention_counts_local(tmp_path):
    summary = evaluate(
        Task(), [trajectory()], tmp_path, "test", 8,
        metadata={"global_step": 37, "stage_step": 37},
    )
    originals = {
        name: (tmp_path / name).read_bytes() for name in ("summary.json", "details.jsonl")
    }
    media = evaluation_media([summary])
    assert set(media) == {
        "evaluation/nll",
        "evaluation/em",
        "evaluation/f1",
        "evaluation/nll_retention/step_37",
        "evaluation/statistics",
    }
    assert all(
        isinstance(media[f"evaluation/{metric}"], swanlab.echarts.Line)
        for metric in ("nll", "em", "f1")
    )
    assert isinstance(media["evaluation/nll_retention/step_37"], swanlab.echarts.HeatMap)
    assert isinstance(media["evaluation/statistics"], swanlab.Html)
    labels = ["1", "3", "总体"]
    rows = [summary["quality"]["by_introduction"][step] for step in ("1", "3")]
    rows.append(summary["quality"]["all"])
    label = "policy · stage step 37 · global step 37"
    for metric in ("em", "f1"):
        quality = media[f"evaluation/{metric}"].options
        assert quality["xAxis"][0]["data"] == labels
        assert [series["name"] for series in quality["series"]] == [label]
        assert [(axis["min"], axis["max"]) for axis in quality["yAxis"]] == [(0, 1)]
        points = quality["series"][0]["data"]
        assert [point[0] for point in points] == labels
        assert [point[1] for point in points] == pytest.approx([row[metric] for row in rows])
    nll = media["evaluation/nll"].options
    assert nll["xAxis"][0]["data"] == labels
    assert [point[1] for point in nll["series"][0]["data"]] == pytest.approx(
        [row["nll"] for row in rows]
    )
    heatmap = media["evaluation/nll_retention/step_37"].options
    assert heatmap["xAxis"][0]["data"] == ["1", "2", "3"]
    assert heatmap["yAxis"][0]["data"] == ["1", "2", "3"]
    assert heatmap["series"][0]["data"] == [[0, 0, 1], [1, 0, 1], [2, 0, 1], [2, 2, 3]]
    local = json.loads((tmp_path / "summary.json").read_text())
    assert [row["questions"] for row in rows] == [1, 1, 2]
    assert local["aggregation"]["retention"] == "trajectory_mean"
    assert [
        (cell["questions"], cell["trajectories"]) for cell in local["nll_retention"]["cells"]
    ] == [(1, 1)] * 4
    assert local["nll_retention"]["by_age"]["0"] == {
        "questions": 2,
        "trajectories": 1,
        "delta_nll": 0,
    }
    record = json.loads((tmp_path / "details.jsonl").read_text())
    assert [len(qa["nll_by_step"]) for qa in record["questions"]] == [3, 1]
    statistics = media["evaluation/statistics"].content
    assert "最终 slots" in statistics and "平均 slots" in statistics
    assert "<td class='number'>4</td><td class='number'>3.33333</td>" in statistics
    assert all((tmp_path / name).read_bytes() == original for name, original in originals.items())


@pytest.mark.parametrize("trajectory_count", [1, 3, 7])
def test_evaluation_build_cost_is_per_trajectory_and_capacity_is_already_a_mean(
    tmp_path, trajectory_count
):
    summary = evaluate(
        Task(),
        [trajectory(f"doc{index}") for index in range(trajectory_count)],
        tmp_path,
        "test",
        8,
        metadata={"global_step": 37, "stage_step": 37},
    )
    summary["costs"]["build_seconds"] = 2.5 * trajectory_count
    media = evaluation_media([summary])
    statistics = media["evaluation/statistics"]
    assert isinstance(statistics, swanlab.Html)
    assert "构建秒 / 轨迹" in statistics.content
    assert "<td class='number'>2.5</td></tr>" in statistics.content
    assert "<td class='number'>4</td><td class='number'>3.33333</td>" in statistics.content


@pytest.mark.parametrize("method", list(METHOD_COLORS))
def test_final_panels_keep_checkpoint_colors_and_measured_introduction_groups(tmp_path, method):
    summary = evaluate(
        Task(method), [trajectory()], tmp_path, "test", 8,
        metadata={"global_step": 37, "stage_step": 37},
    )
    media = evaluation_media([summary])
    assert set(summary["quality"]) == {"all", "by_introduction"}
    series = media["evaluation/nll"].options["series"][0]
    assert series["name"] == f"{summary['stage']} · stage step 37 · global step 37"
    color = series["itemStyle"]["color"]
    for metric in ("em", "f1"):
        quality = media[f"evaluation/{metric}"].options
        assert quality["series"][0]["itemStyle"]["color"] == color
        assert quality["series"][0]["lineStyle"]["color"] == color
        assert quality["series"][0]["showSymbol"] is True
        assert [(axis["min"], axis["max"]) for axis in quality["yAxis"]] == [(0, 1)]
        assert quality["xAxis"][0]["data"] == ["1", "3", "总体"]
        assert media["evaluation/nll"].options["xAxis"][0]["data"] == quality["xAxis"][0]["data"]
    if method in DYNAMIC_METHODS:
        heatmap = media["evaluation/nll_retention/step_37"].options
        assert heatmap["series"][0]["name"] == "NLL"
        assert heatmap["visualMap"]["inRange"]["color"][-1] == METHOD_COLORS[method]
        assert heatmap["visualMap"]["inRange"]["color"][0] != METHOD_COLORS[method]
    else:
        assert not any(name.startswith("evaluation/nll_retention/") for name in media)
