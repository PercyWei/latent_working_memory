import pytest
import swanlab
from types import SimpleNamespace

from latent_working_memory.v3.evaluate import evaluate
from latent_working_memory.v3.tracking import (
    METHOD_COLORS,
    configure_training_metrics,
    evaluation_media,
)

from .test_evaluate import Task, trajectory


def test_dashboard_adds_six_panels_including_epoch_across_all_stages(tmp_path):
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
        "evaluation/em",
        "evaluation/f1",
    }
    assert names - original == {
        "train/stage",
        "train/epoch",
        "dev/qa_old_nll",
        "dev/qa_new_nll",
        "evaluation/capacity",
        "evaluation/build_seconds_per_trajectory",
    }


def test_evaluation_merges_quality_and_capacity_without_table_panels(tmp_path):
    summary = evaluate(Task(), [trajectory()], tmp_path, "test", 8)
    originals = {
        name: (tmp_path / name).read_bytes() for name in ("summary.json", "trajectories.jsonl")
    }
    media = evaluation_media(summary)
    assert set(media) == {
        "evaluation/nll",
        "evaluation/em",
        "evaluation/f1",
        "evaluation/capacity",
        "evaluation/build_seconds_per_trajectory",
    }
    assert all(isinstance(chart, swanlab.echarts.Bar) for chart in media.values())
    for metric in ("nll", "em", "f1"):
        chart = media[f"evaluation/{metric}"]
        assert chart.options["xAxis"][0]["data"] == ["all", "old", "new"]
        assert len(chart.options["series"]) == 1
        assert [point["value"] for point in chart.options["series"][0]["data"]] == pytest.approx(
            [summary["quality"][group][metric] for group in ("all", "old", "new")]
        )
    capacity = media["evaluation/capacity"]
    assert capacity.options["xAxis"][0]["data"] == ["final_slots", "mean_slots"]
    assert [point["value"] for point in capacity.options["series"][0]["data"]] == pytest.approx(
        [4, 10 / 3]
    )
    assert capacity.options["yAxis"][0]["name"] == "slots"
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
    build = media["evaluation/build_seconds_per_trajectory"]
    assert build.options["series"][0]["data"][0]["value"] == pytest.approx(2.5)
    assert build.options["yAxis"][0]["name"] == "seconds / trajectory"
    assert [
        point["value"] for point in media["evaluation/capacity"].options["series"][0]["data"]
    ] == pytest.approx([4, 10 / 3])


@pytest.mark.parametrize("method", list(METHOD_COLORS))
def test_final_panels_keep_method_colors_and_skip_unmeasured_quality_groups(tmp_path, method):
    summary = evaluate(Task(method), [trajectory()], tmp_path, "test", 8)
    summary["quality"]["old"] = {"questions": 0, "nll": None, "em": None, "f1": None}
    media = evaluation_media(summary)
    for name, chart in media.items():
        assert chart.options["series"][0]["name"] == method
        assert chart.options["series"][0]["itemStyle"]["color"] == METHOD_COLORS[method]
        if name in {"evaluation/nll", "evaluation/em", "evaluation/f1"}:
            assert chart.options["xAxis"][0]["data"] == ["all", "new"]
