from latent_working_memory.v1.tracking import swanlab_run
from latent_working_memory.v1.reporting import (
    build_evaluation_charts,
    comparison_charts,
    build_test_report_charts,
)


def test_report_axes_and_missing_values():
    report = {
        "groups": {
            "all/ae/memory": {"nll": 2.0, "reads": 6},
            "all/ae/no_memory": {"nll": 3.0, "reads": 6},
            "length_up_to/128/ae/memory": {"nll": 2.5},
            "length_up_to/32/ae/memory": {"nll": 1.5},
            "length_up_to/128/ae/no_memory": {"nll": 3.5},
            "boundary_method/pysbd_conservative/ae/memory": {"nll": 2.0},
        },
        "comparisons": {"all/ae": {"gain_vs_no_memory": 1.0}},
    }
    charts = build_test_report_charts(report)
    assert all(not isinstance(value, (float, int)) for value in charts.values())
    assert charts["charts/report/all/ae/nll"].options["xAxis"][0]["data"] == ["memory", "no_memory"]
    chart = charts["charts/report/length_up_to/ae/nll"].options
    assert chart["xAxis"][0]["data"] == ["32", "128"]
    assert chart["series"][1]["data"] == [None, 3.5]
    rows = charts["tables/report/groups/all"].options["rows"]
    assert rows == [["", "ae/memory", 2.0, 6], ["", "ae/no_memory", 3.0, 6]]


def test_comparison_uses_explicit_labels_and_separate_metrics():
    a = {"groups": {"all/ae/memory": {"nll": 1.0, "bleu_4": 4.0}}}
    b = {"groups": {"all/ae/memory": {"nll": 2.0, "bleu_4": 5.0}}}
    charts = comparison_charts([("A", "test-X", a), ("B", "test-X", b)])
    opts = charts["charts/comparison/ae/nll"].options
    assert opts["xAxis"][0]["data"] == ["A", "B"]
    assert opts["series"][0]["data"] == [1.0, 2.0]
    assert charts["charts/comparison/ae/bleu_4"].options["series"][0]["data"] == [4.0, 5.0]


def test_offline_chart_serialization(tmp_path):
    report = {
        "groups": {"all/ae/memory": {"nll": 2.0, "reads": 3}},
        "comparisons": {"all/ae": {"gain_vs_no_memory": 0.2}},
    }
    with swanlab_run(tmp_path, {}, mode="offline", group="report-test", job_type="evaluate") as run:
        run.log(build_test_report_charts(report))
    assert not run.alive


def test_grouped_sources_and_display_precision():
    report = {"groups": {"all/ae/memory": {"nll": 1.23456789}}, "comparisons": {}}
    charts = build_evaluation_charts([("semantic", report), ("random", report)])
    series = charts["charts/report/all/ae/nll"].options["series"]
    assert [s["name"] for s in series] == ["semantic", "random"]
    assert all(s["data"] == [1.2346] for s in series)
    assert charts["tables/report/groups/all"].options["rows"][0][-1] == 1.2346
    assert report["groups"]["all/ae/memory"]["nll"] == 1.23456789


def test_report_panel_names_are_separate():
    report = {"groups": {"all/ae/memory": {"nll": 1.0}}, "comparisons": {}}
    panels = build_evaluation_charts([("semantic", report)])
    assert {key.split("/")[0] for key in panels} == {"charts", "tables"}
    panels = comparison_charts([("model", "test", report)])
    assert {key.split("/")[0] for key in panels} == {"charts", "tables"}
