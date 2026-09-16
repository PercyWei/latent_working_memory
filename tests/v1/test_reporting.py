from latent_working_memory.v1.pretrain.reporting import (
    evaluation_overview, paired_reconstructions, development_panel_style,
    configure_development_panels, development_panels, development_scalars,
)
import json
import colorsys
from types import SimpleNamespace
from latent_working_memory.v1.pretrain.reporting import build_evaluation_charts, comparison_charts
from latent_working_memory.v1.pretrain.tracking import pretraining_run


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
    with pretraining_run(tmp_path, {}, mode="offline", group="report-test", job_type="evaluate") as run:
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


def test_compact_reports_keep_details_and_pair_all_controls(tmp_path):
    data = report()
    data["prefix_diagnostics"] = {"memory/prefix-8": {"bleu_4": 20}}
    charts = evaluation_overview([("semantic", data)], "evaluation/test/overview")
    assert set(charts) == {"evaluation/test/overview/ae/nll", "evaluation/test/overview/summary",
                           "evaluation/test/overview/details"}
    # PPL, stratified metrics and prefix diagnostics remain in the tables.
    serialized = str(charts["evaluation/test/overview/details"].html_content)
    assert "prefix-8" in serialized and "length_ratio" in serialized
    path = tmp_path / "reads.jsonl"
    rows = [{"episode_id": "one", "capacity": 32, "condition": c, "reference": "original",
             "prediction": c, "correct_prefix_ratio": 0.0} for c in ("memory", "wrong_memory")]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    media = paired_reconstructions([("semantic", path), ("random", path)], "dev/overview")
    assert len(media["dev/overview/examples"]) == 2


def test_sparse_generation_points_and_eight_curve_limit():
    reports = {'semantic': {'groups': {'all/ae/memory': {'nll': 2.0}}},
               'random': {'groups': {'all/ae/memory': {'nll': 3.0, 'correct_prefix_ratio': 0.0}}}}
    points = development_scalars(reports)
    assert points == {'dev/overview/ae/nll/semantic/memory': 2.0,
                      'dev/overview/ae/nll/random/memory': 3.0,
                      'dev/overview/ae/correct_prefix_ratio/random/memory': 0.0}
    panels = development_panels(list(reports))
    assert len(panels) == 6
    assert len(panels['dev/overview/ae/correct_prefix_ratio']['config']['yAxis']) == 8
    assert all(len(p['config']['yAxis']) <= 8 for p in panels.values())
    keys = [y['key'] for p in panels.values() for y in p['config']['yAxis']]
    assert len(keys) == len(set(keys)) == 42
    assert set(points) <= set(keys)
    assert all(p['config']['xAxis']['key'] == 'step' for p in panels.values())


def test_native_panels_created_once_before_scalar_upload(monkeypatch):
    posts = []
    section = {'index': 'dev-section', 'name': 'dev', 'chartIndex': []}
    charts = {}

    class Api:
        def run(self, path):
            return SimpleNamespace(run_id='cloud-id')

        def _get(self, path, params=None):
            data = ([section] if posts else []) if path.endswith('/sections') else charts[path.split('/')[-2]]
            return SimpleNamespace(ok=True, data=data)

        def _post(self, path, data):
            assert path.endswith('/columns')
            posts.append(data)
            for column in data:
                index = column['chartIndex']
                if index not in charts:
                    charts[index] = {'title': column['chartName'], 'index': index, 'type': 'LINE',
                                     'config': {'yAxis': []}}
                    section['chartIndex'].append(index)
                charts[index]['config']['yAxis'].append(column['key'])
            return SimpleNamespace(ok=True, data=[])

        def _put(self, path, data):
            index = path.split('/')[-3]
            charts[index].update(data)
            return SimpleNamespace(ok=True, data=None)

    monkeypatch.setattr('latent_working_memory.v1.reporting.swanlab.Api', Api)
    run = SimpleNamespace(id='slug', url='https://swanlab.cn/@user/project/runs/slug')
    configure_development_panels(run, ['semantic', 'random'], 'online')
    assert len(posts) == 1 and len(posts[0]) == 42
    assert len(charts) == 6
    assert all('hidden' not in column and column['sectionName'] == 'dev' for column in posts[0])
    configure_development_panels(run, ['semantic', 'random'], 'online')
    assert len(posts) == 1


def test_condition_colors_and_adjacent_source_legends():
    sources = ['semantic', 'random']
    panels = development_panels(sources)
    panel = panels['dev/overview/ae/nll']
    style = development_panel_style(panel, sources, 'run')
    labels = [value['name'] for value in style.values()]
    assert labels == [f'{source}/{condition}' for condition in
                      ['memory', 'wrong_memory', 'full_context', 'base_full_context']
                      for source in sources]
    hues = []
    for i in range(0, len(labels), 2):
        pair = list(style.values())[i:i + 2]
        hls = [colorsys.rgb_to_hls(*(int(v['colors'][0][j:j + 2], 16) / 255
                                    for j in (1, 3, 5))) for v in pair]
        assert abs(hls[0][0] - hls[1][0]) < 0.005
        assert hls[1][1] - hls[0][1] > 0.2
        hues.append(round(hls[0][0], 1))
    assert len(set(hues)) == 4
    # A source keeps its shade even when LM panels split the sources.
    lm = panels['dev/overview/continuation/nll/random']
    lm_style = development_panel_style(lm, sources, 'run')
    assert next(iter(lm_style.values()))['colors'] == list(style.values())[1]['colors']
