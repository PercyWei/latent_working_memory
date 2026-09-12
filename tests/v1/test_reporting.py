import json
from latent_working_memory.v1.reporting import build_evaluation_charts, comparison_charts
from latent_working_memory.v1.tracking import swanlab_run


def report():
    return {
        "groups": {
            "all/ae/memory": {"nll": 1.23456789, "ppl": 3.4, "reads": 6},
            "all/ae/wrong_memory": {"nll": 3.0},
            "length_ratio/128/2/ae/memory": {"nll": 2.5},
            "length_ratio/64/4/ae/memory": {"nll": 1.5},
            "length_ratio/64/2/ae/memory": {"nll": 1.2},
        },
        "comparisons": {"all/ae": {"gain_vs_wrong_memory": 1.7}},
    }


def test_joint_axes_source_pairing_and_numeric_precision():
    a = report()
    b = report()
    del b["groups"]["length_ratio/64/4/ae/memory"]
    charts = build_evaluation_charts([("semantic", a), ("random", b)])
    opts = charts["charts/report/length_ratio/ae/nll"].options
    assert opts["xAxis"][0]["data"] == ["64/r2", "64/r4", "128/r2"]
    assert [s["name"] for s in opts["series"]] == ["semantic", "random"]
    assert opts["series"][1]["data"][1]["value"] is None
    opts = charts["charts/report/all/ae/nll"].options
    assert opts["xAxis"][0]["data"] == ["memory", "wrong_memory"]
    assert opts["series"][0]["data"][0]["value"] == 1.2346
    assert (
        opts["series"][0]["data"][0]["itemStyle"]["color"]
        != opts["series"][1]["data"][0]["itemStyle"]["color"]
    )
    assert a["groups"]["all/ae/memory"]["nll"] == 1.23456789
    assert {k.split("/")[0] for k in charts} == {"charts", "tables"}


def test_comparison_labels_and_separate_metrics():
    charts = comparison_charts([("A", "test-X", report()), ("B", "test-X", report())])
    opts = charts["charts/comparison/ae/nll"].options
    assert opts["xAxis"][0]["data"] == ["A", "B"]
    assert [v["value"] for v in opts["series"][0]["data"]] == [1.2346, 1.2346]
    assert {k.split("/")[0] for k in charts} == {"charts", "tables"}


def test_offline_chart_serialization(tmp_path):
    with swanlab_run(tmp_path, {}, mode="offline", group="report-test", job_type="evaluate") as run:
        run.log(build_evaluation_charts([("semantic", report()), ("random", report())]))
    assert not run.alive


def test_serialized_labels_and_metric_coverage():
    metrics = {
        "groups": {
            **{
                f"all/continuation/{condition}": {"nll": 2.0, "ppl": 7.4}
                for condition in [
                    "memory",
                    "wrong_memory",
                    "no_memory",
                    "full_context",
                    "base_full_context",
                ]
            },
            "all/ae/memory": {"nll": 2.0, "ppl": 7.4, "bleu_4": 1.0, "correct_prefix_ratio": 0.02},
            "all/ae/wrong_memory": {"nll": 3.0, "ppl": 20.0},
        },
        "comparisons": {},
    }
    charts = build_evaluation_charts([("semantic", metrics), ("random", metrics)])
    for metric in ["nll", "ppl"]:
        options = json.loads(
            charts[f"charts/report/all/continuation/{metric}"].dump_options_with_quotes()
        )
        assert options["xAxis"][0]["data"] == [
            "memory",
            "wrong_memory",
            "no_memory",
            "full_context",
            "base_full_context",
        ]
        assert options["xAxis"][0]["axisLabel"]["interval"] == 0
        assert options["yAxis"][0]["minInterval"] == 0.001
        assert all(s["label"]["show"] is False for s in options["series"])
        assert options["grid"]["containLabel"] is True
        assert all(len(s["data"]) == 5 for s in options["series"])
    for metric in ["bleu_4", "correct_prefix_ratio"]:
        options = json.loads(charts[f"charts/report/all/ae/{metric}"].dump_options_with_quotes())
        assert options["xAxis"][0]["data"] == ["memory"]
        assert "错误记忆未评估" in options["title"]["subtext"]
