import json
import colorsys
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
    assert opts["xAxis"][0]["data"] == ["64\nr2", "64\nr4", "128\nr2"]
    assert [s["name"] for s in opts["series"]] == [
        "condition=memory / test=semantic",
        "condition=memory / test=random",
    ]
    assert opts["series"][1]["data"][1] is None
    opts = charts["charts/report/all/ae/nll"].options
    assert opts["xAxis"][0]["data"] == ["memory", "wrong\nmemory"]
    assert opts["series"][0]["data"][0] == 1.2346
    assert opts["series"][0]["itemStyle"]["color"] != opts["series"][1]["itemStyle"]["color"]
    assert a["groups"]["all/ae/memory"]["nll"] == 1.23456789
    assert {k.split("/")[0] for k in charts} == {"charts", "tables"}


def test_comparison_labels_and_separate_metrics():
    trains, tests = ["semantic", "random", "mixed"], ["semantic", "random"]
    charts = comparison_charts([(train, test, report()) for train in trains for test in tests])
    opts = charts["charts/comparison/ae/nll"].options
    assert opts["xAxis"][0]["data"] == trains
    assert len(opts["series"]) == 6
    hues = []
    for i, train in enumerate(trains):
        pair = opts["series"][2 * i : 2 * i + 2]
        color_components = []
        for series, test in zip(pair, tests, strict=True):
            assert series["name"] == f"train={train} / test={test}"
            assert series["stack"] == test
            assert series["data"] == [1.2346 if j == i else None for j in range(3)]
            color = series["itemStyle"]["color"].lstrip("#")
            color_components.append(
                colorsys.rgb_to_hls(*(int(color[j : j + 2], 16) / 255 for j in (0, 2, 4)))
            )
        assert abs(color_components[0][0] - color_components[1][0]) < 0.01
        assert color_components[0][1] < color_components[1][1]
        hues.append(round(color_components[0][0], 1))
    assert len(set(hues)) == 3
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
            **{
                f"all/ae/{c}": {
                    "nll": 3.0,
                    "ppl": 20.0,
                    "bleu_4": 0.5,
                    "correct_prefix_ratio": 0.01,
                }
                for c in ["wrong_memory", "full_context", "base_full_context"]
            },
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
            "wrong\nmemory",
            "no\nmemory",
            "full\ncontext",
            "base\nfull\ncontext",
        ]
        assert options["xAxis"][0]["axisLabel"]["interval"] == 0
        assert options["yAxis"][0]["minInterval"] == 0.001
        assert all(s["label"]["show"] is False for s in options["series"])
        assert options["grid"]["containLabel"] is False
        assert options["grid"]["height"] == "55%"
        assert options["grid"]["top"] == "20%"
        assert all(len(s["data"]) == 5 for s in options["series"])
        assert len(options["series"]) == 10
        assert {s["stack"] for s in options["series"]} == {"semantic", "random"}
        for i, condition in enumerate(
            ["memory", "wrong_memory", "no_memory", "full_context", "base_full_context"]
        ):
            for j, source in enumerate(["semantic", "random"]):
                series = options["series"][2 * i + j]
                assert series["name"] == f"condition={condition} / test={source}"
                assert sum(v is not None for v in series["data"]) == 1
                assert series["data"][i] is not None
    for metric in ["bleu_4", "correct_prefix_ratio"]:
        options = json.loads(charts[f"charts/report/all/ae/{metric}"].dump_options_with_quotes())
        assert options["xAxis"][0]["data"] == [
            "memory",
            "wrong\nmemory",
            "full\ncontext",
            "base\nfull\ncontext",
        ]
