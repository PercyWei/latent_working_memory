import json

import pytest
import swanlab

from latent_working_memory.v3.tracking import (
    TRAINING_METRICS,
    configure_training_metrics,
    evaluation_media,
    training_metrics,
)
from .test_evaluate import Task, trajectory
from latent_working_memory.v3.evaluate import evaluate


def test_training_panels_are_registered_before_log_and_use_optimizer_steps():
    class Run:
        def __init__(self):
            self.definitions = []

        def define_metric(self, name, **kwargs):
            self.definitions.append((name, kwargs))

    run = Run()
    configure_training_metrics(run)
    assert tuple(name for name, _ in run.definitions) == TRAINING_METRICS
    assert all(options["x_axis"] == "_step" for _, options in run.definitions)
    assert all(options["section_name"] == name.split("/")[0] for name, options in run.definitions)
    assert not any("hidden" in options for _, options in run.definitions)
    configure_training_metrics(None)


def test_training_publishes_only_current_measured_core_values():
    record = {
        "step": 9,
        "train/loss": 2.1,
        "train/grad_norm": 0.5,
        "train/slots_final": 128,
        "train/qa_old_count": 8,
        "train/gate_g": 0.01,
        "train/write_calls": 4,
        "dev/loss": None,
        "resources/peak_memory_allocated_bytes": 3 * 1024**3,
        "resources/optimizer_step_seconds": 4.2,
    }
    assert training_metrics(record) == {
        "train/loss": 2.1,
        "train/grad_norm": 0.5,
        "train/slots_final": 128,
        "resources/peak_memory_allocated_gib": 3,
        "resources/optimizer_step_seconds": 4.2,
    }
    assert "dev/loss" not in training_metrics({"train/loss": 1})
    assert record["train/qa_old_count"] == 8


def test_evaluation_merges_quality_groups_and_keeps_diagnostics_in_tables(tmp_path):
    summary = evaluate(Task(), [trajectory()], tmp_path, "test", 8)
    rows = [json.loads(line) for line in (tmp_path / "trajectories.jsonl").read_text().splitlines()]
    media = evaluation_media(summary, rows)
    assert set(media) == {
        "evaluation/nll",
        "evaluation/em",
        "evaluation/f1",
        "evaluation/summary",
        "evaluation/details",
        "evaluation/examples",
    }
    for metric in ("nll", "em", "f1"):
        chart = media[f"evaluation/{metric}"]
        assert isinstance(chart, swanlab.echarts.Bar)
        assert chart.options["xAxis"][0]["data"] == ["all", "old", "new"]
        assert len(chart.options["series"]) == 1
        assert [point["value"] for point in chart.options["series"][0]["data"]] == pytest.approx(
            [summary["quality"][group][metric] for group in ("all", "old", "new")]
        )
    assert all(
        isinstance(media[f"evaluation/{name}"], swanlab.echarts.Table)
        for name in ("summary", "details", "examples")
    )
    assert "gate_qa_reads" in media["evaluation/details"].html_content
    assert "Hidden gate question" not in media["evaluation/examples"].html_content
    assert "The blue whale." in media["evaluation/examples"].html_content
