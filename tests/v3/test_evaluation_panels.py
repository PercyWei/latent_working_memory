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


def test_dashboard_registers_training_curves_and_three_evaluation_charts_and_table(tmp_path):
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
    names.update(evaluation_media(evaluate(Task(), [trajectory()], tmp_path, "test", 8)))
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
        "evaluation/em_f1",
        "evaluation/nll_retention",
        "evaluation/statistics",
    }


def test_evaluation_merges_generation_quality_and_keeps_retention_counts_local(tmp_path):
    summary = evaluate(Task(), [trajectory()], tmp_path, "test", 8)
    originals = {
        name: (tmp_path / name).read_bytes() for name in ("summary.json", "trajectories.jsonl")
    }
    media = evaluation_media(summary)
    assert set(media) == {
        "evaluation/nll",
        "evaluation/em_f1",
        "evaluation/nll_retention",
        "evaluation/statistics",
    }
    assert isinstance(media["evaluation/em_f1"], swanlab.echarts.Bar)
    assert isinstance(media["evaluation/nll_retention"], swanlab.echarts.HeatMap)
    assert isinstance(media["evaluation/statistics"], swanlab.Html)
    labels = ["1", "3", "总体"]
    rows = [summary["quality"]["by_introduction"][step] for step in ("1", "3")]
    rows.append(summary["quality"]["all"])
    quality = media["evaluation/em_f1"].options
    assert quality["xAxis"][0]["data"] == labels
    assert [series["name"] for series in quality["series"]] == ["EM", "F1"]
    assert [series["yAxisIndex"] for series in quality["series"]] == [0, 0]
    assert [(axis["min"], axis["max"]) for axis in quality["yAxis"]] == [(0, 1)]
    for series, metric in zip(quality["series"], ("em", "f1"), strict=True):
        assert series["data"] == pytest.approx([row[metric] for row in rows])
    nll = media["evaluation/nll"].options
    assert nll["xAxis"][0]["data"] == labels
    assert [point["value"] for point in nll["series"][0]["data"]] == pytest.approx(
        [row["nll"] for row in rows]
    )
    heatmap = media["evaluation/nll_retention"].options
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
    record = json.loads((tmp_path / "trajectories.jsonl").read_text())
    assert [len(qa["nll_by_step"]) for qa in record["questions"]] == [3, 1]
    statistics = media["evaluation/statistics"].content
    assert "最终记忆容量" in statistics and "平均记忆容量" in statistics
    assert "<td class='number'>4</td><td>slots</td>" in statistics
    assert "<td class='number'>3.33333</td><td>slots</td>" in statistics
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
    )
    summary["costs"]["build_seconds"] = 2.5 * trajectory_count
    media = evaluation_media(summary)
    statistics = media["evaluation/statistics"]
    assert isinstance(statistics, swanlab.Html)
    assert "<td class='number'>2.5</td><td>秒 / 轨迹</td>" in statistics.content
    assert "<td class='number'>4</td><td>slots</td>" in statistics.content
    assert "<td class='number'>3.33333</td><td>slots</td>" in statistics.content


@pytest.mark.parametrize("method", list(METHOD_COLORS))
def test_final_panels_keep_method_colors_and_measured_introduction_groups(tmp_path, method):
    summary = evaluate(Task(method), [trajectory()], tmp_path, "test", 8)
    media = evaluation_media(summary)
    color = METHOD_COLORS[method]
    assert set(summary["quality"]) == {"all", "by_introduction"}
    series = media["evaluation/nll"].options["series"][0]
    assert series["name"] == method
    assert series["itemStyle"]["color"] == color
    quality = media["evaluation/em_f1"].options
    em, f1 = quality["series"]
    assert em["itemStyle"]["color"].lower() == color.lower()
    assert f1["itemStyle"]["color"] != em["itemStyle"]["color"]
    assert all(
        int(f1["itemStyle"]["color"][index : index + 2], 16) >= int(color[index : index + 2], 16)
        for index in (1, 3, 5)
    )
    assert [(axis["min"], axis["max"]) for axis in quality["yAxis"]] == [(0, 1)]
    assert quality["xAxis"][0]["data"] == ["1", "3", "总体"]
    assert media["evaluation/nll"].options["xAxis"][0]["data"] == quality["xAxis"][0]["data"]
    if method in DYNAMIC_METHODS:
        heatmap = media["evaluation/nll_retention"].options
        assert heatmap["series"][0]["name"] == method
        assert heatmap["visualMap"]["inRange"]["color"][-1] == color
        assert heatmap["visualMap"]["inRange"]["color"][0] == f1["itemStyle"]["color"]
    else:
        assert "evaluation/nll_retention" not in media
