"""从完整 checkpoint 集合重建评估图，累计 step 决定曲线顺序。"""

from colorsys import hsv_to_rgb
from hashlib import sha256
from pathlib import Path

from pyecharts import charts

from latent_working_memory.benchmarks.data import RULER_TASKS


RULER_TASK_LABELS = {
    "niah_single_1": "single_1",
    "niah_single_2": "single_2",
    "niah_single_3": "single_3",
    "niah_multikey_1": "multik_1",
    "niah_multikey_2": "multik_2",
    "niah_multikey_3": "multik_3",
    "niah_multivalue": "multiv",
    "niah_multiquery": "multiq",
    "vt": "vt",
    "cwe": "cwe",
    "fwe": "fwe",
    "qa_1": "qa_1",
    "qa_2": "qa_2",
}


def evaluation_charts(summaries, baselines=None, ruler_length=None):
    """summaries 按真实累计 step 排序；RULER 各 checkpoint 可覆盖不同长度。"""
    baselines = {} if baselines is None else baselines
    if baselines and "benchmark" not in summaries[0]:
        raise ValueError("baseline results apply only to benchmark reports")
    if summaries[0].get("benchmark") == "ruler":
        if len(summaries) == 1:
            return {"evaluation/ruler_lengths": _ruler_lengths_chart(summaries[0], baselines)}
        return {"evaluation/ruler": _ruler_progress_chart(summaries, baselines, ruler_length)}
    if len(summaries) == 1:
        summary = summaries[0]
        return (
            _benchmark_charts(summary, baselines) if "benchmark" in summary else _test_charts(summary)
        )
    return _checkpoint_charts(summaries, baselines)


def _test_charts(summary):
    metrics = summary["metrics"]
    quality = charts.Bar().add_xaxis(["NLL", "EM", "F1"])
    for index, (metric, label, color) in enumerate((
        ("nll", "NLL", "#31688e"),
        ("em", "EM", "#74a9cf"),
        ("f1", "F1", "#2b8cbe"),
    )):
        quality.add_yaxis(
            label,
            [metrics[metric] if position == index else None for position in range(3)],
            yaxis_index=0 if metric == "nll" else 1,
            gap="-100%",
            label_opts={"show": False},
            itemstyle_opts={"color": color},
        )
    quality.set_global_opts(
        tooltip_opts={"trigger": "axis"},
        yaxis_opts={"name": "NLL", "type": "value", "min": 0, "position": "left"},
    )
    quality.extend_axis(yaxis={
        "name": "EM / F1", "type": "value", "min": 0, "max": 1, "position": "right",
        "splitLine": {"show": False},
    })

    write_loss = charts.Bar().add_xaxis(["Before write", "After write"])
    write_loss.add_yaxis(
        Path(summary["training_run"]).name,
        [metrics["write_loss_before"], metrics["write_loss"]],
        label_opts={"show": False},
        itemstyle_opts={"color": "#31688e"},
    )
    write_loss.set_global_opts(
        tooltip_opts={"trigger": "axis"},
        yaxis_opts={"name": "Attention Matching loss", "min": 0},
    )
    return {"evaluation/quality": quality, "evaluation/write_loss": write_loss}


def _group_color(index, count):
    fraction = 0.6 * (1 - index / max(1, count - 1))
    return "#" + "".join(
        f"{round(channel + (255 - channel) * fraction):02x}" for channel in (49, 104, 142)
    )


def _baseline_color(label, index=None, count=0):
    """标签固定色相，内部各组只改变深浅，并避开主方法的蓝色色系。"""
    hue = int.from_bytes(sha256(label.encode("utf-8")).digest()[:8], "big") / 2**64 * 0.78
    if hue >= 0.46:
        hue += 0.22
    channels = [255 * value for value in hsv_to_rgb(hue, 0.65, 0.75)]
    if index is None:
        channels = [value * 0.75 for value in channels]
    else:
        fraction = 0.6 * (1 - index / max(1, count - 1))
        channels = [value + (255 - value) * fraction for value in channels]
    return "#" + "".join(f"{round(value):02x}" for value in channels)


def _benchmark_charts(summary, baselines):
    metrics = summary["metrics"]
    method = Path(summary["training_run"]).name
    groups = sorted(metrics["groups"], key=lambda row: row["category"])
    chart = charts.Bar().add_xaxis([*[row["category_name"] for row in groups], "总体"])
    for baseline_label, baseline in baselines.items():
        reference_groups = {
            row["category"]: row["score"] for row in baseline["metrics"]["groups"]
        }
        chart.add_yaxis(
            baseline_label,
            [100 * reference_groups[row["category"]] for row in groups]
            + [100 * baseline["metrics"]["score"]],
            label_opts={"show": False},
            itemstyle_opts={"color": _baseline_color(baseline_label)},
        )
    chart.add_yaxis(
        method,
        [100 * row["score"] for row in groups] + [100 * metrics["score"]],
        label_opts={"show": False},
        itemstyle_opts={"color": "#31688e"},
    )
    chart.set_global_opts(
        tooltip_opts={"trigger": "axis"},
        yaxis_opts={"name": f"{metrics['metric']} (%)", "min": 0, "max": 100},
    )
    return {f"evaluation/{summary['benchmark']}": chart}


def _ruler_lengths_chart(summary, baselines):
    lengths = sorted(row["length"] for row in summary["metrics"]["by_length"])
    chart = charts.Line().add_xaxis([f"{length / 1024:g}K" for length in lengths])
    methods = [
        (label, baseline, _baseline_color(label)) for label, baseline in baselines.items()
    ]
    methods.append((Path(summary["training_run"]).name, summary, "#31688e"))
    for label, result, color in methods:
        values = {row["length"]: 100 * row["score"] for row in result["metrics"]["by_length"]}
        chart.add_yaxis(
            label, [values.get(length) for length in lengths],
            color=color, is_symbol_show=True, symbol="circle", symbol_size=6,
            label_opts={"show": False},
            linestyle_opts={"color": color, "width": 2},
            itemstyle_opts={"color": color},
        )
    chart.set_global_opts(
        tooltip_opts={"trigger": "axis"},
        legend_opts={"type": "scroll"},
        xaxis_opts={"name": "Context length", "type": "category", "axisLabel": {"interval": 0}},
        yaxis_opts={"name": "Mean accuracy (%)", "min": 0, "max": 100},
    )
    return chart


def ruler_task_chart(summary, length, baselines=None):
    """只绘制一个长度的任务分数；短标签用于横轴，完整任务名用于悬停提示。"""
    baselines = {} if baselines is None else baselines
    chart = charts.Bar().add_xaxis([RULER_TASK_LABELS[task] for task in RULER_TASKS])
    methods = [
        (label, baseline, _baseline_color(label)) for label, baseline in baselines.items()
    ]
    methods.append((Path(summary["training_run"]).name, summary, "#31688e"))
    for label, result, color in methods:
        values = {
            (row["length"], row["task"]): row["score"] for row in result["metrics"]["groups"]
        }
        chart.add_yaxis(
            label,
            [{"name": task, "value": 100 * values[(length, task)]
              if (length, task) in values else None} for task in RULER_TASKS],
            label_opts={"show": False},
            itemstyle_opts={"color": color},
        )
    chart.set_global_opts(
        tooltip_opts={"trigger": "item", "formatter": "{b}<br/>{a}: {c}%"},
        legend_opts={"type": "scroll"},
        xaxis_opts={"type": "category", "axisLabel": {"interval": 0, "hideOverlap": False}},
        yaxis_opts={"name": "Accuracy (%)", "min": 0, "max": 100},
    )
    return chart


def ruler_overall(summary, lengths):
    """按指定长度范围计算 Mean；缺少任一长度时不提供 Overall。"""
    scores = {row["length"]: row["score"] for row in summary["metrics"]["by_length"]}
    return sum(scores[length] for length in lengths) / len(lengths) if set(lengths) <= scores.keys() else None


def _ruler_progress_chart(summaries, baselines, length):
    steps = [summary["checkpoint"]["step"] for summary in summaries]
    method = Path(summaries[0]["training_run"]).name
    values = []
    for summary in summaries:
        by_length = {row["length"]: row["score"] for row in summary["metrics"]["by_length"]}
        value = summary["metrics"]["score"] if length is None else by_length.get(length)
        values.append(100 * value if value is not None else None)
    references = []
    for label, baseline in baselines.items():
        by_length = {row["length"]: row["score"] for row in baseline["metrics"]["by_length"]}
        reference_values = [ruler_overall(baseline, summary["protocol"]["lengths"])
                            if length is None else by_length.get(length) for summary in summaries]
        references.append((
            label, [100 * value if value is not None else None for value in reference_values],
            _baseline_color(label),
        ))
    return _line_chart(steps, [(method, values, "#31688e")], "Mean accuracy (%)", 100, references)


def _line_chart(steps, series, unit, maximum=None, references=()):
    chart = charts.Line().add_xaxis(steps)
    for label, measurements, color in references:
        chart.add_yaxis(
            label,
            measurements,
            color=color,
            is_symbol_show=False,
            label_opts={"show": False},
            linestyle_opts={"color": color, "width": 2, "type": "dashed"},
            itemstyle_opts={"color": color},
        )
    for label, measurements, color in series:
        chart.add_yaxis(
            label,
            measurements,
            color=color,
            is_symbol_show=True,
            symbol="circle",
            symbol_size=6,
            label_opts={"show": False},
            linestyle_opts={"color": color, "width": 2},
            itemstyle_opts={"color": color},
        )
    chart.set_global_opts(
        tooltip_opts={"trigger": "axis"},
        legend_opts={"type": "scroll"},
        xaxis_opts={"name": "累计 step", "type": "value", "minInterval": 1},
        yaxis_opts={"name": unit, "min": 0, **({"max": maximum} if maximum is not None else {})},
    )
    return chart


def _checkpoint_charts(summaries, baselines):
    steps = [summary["checkpoint"]["step"] for summary in summaries]
    if "benchmark" in summaries[0]:
        return _benchmark_checkpoint_charts(summaries, steps, baselines)
    panels = (
        ("nll", (("nll", "NLL", "#31688e"),), "NLL", None),
        (
            "em_f1",
            (("em", "EM", "#74a9cf"), ("f1", "F1", "#2b8cbe")),
            "score",
            1,
        ),
        (
            "write_loss",
            (
                ("write_loss_before", "Before write", "#74a9cf"),
                ("write_loss", "After write", "#2b8cbe"),
            ),
            "Attention Matching loss",
            None,
        ),
    )
    return {
        f"evaluation/{name}": _line_chart(
            steps,
            [
                (label, [summary["metrics"][metric] for summary in summaries], color)
                for metric, label, color in series
            ],
            unit,
            maximum,
        )
        for name, series, unit, maximum in panels
    }


def _benchmark_checkpoint_charts(summaries, steps, baselines):
    first = summaries[0]
    benchmark = first["benchmark"]
    group_field, identity_field = "groups", "category"
    groups = {
        row[identity_field]: row
        for summary in summaries
        for row in summary["metrics"][group_field]
    }
    groups = [groups[identity] for identity in sorted(groups)]
    grouped = [
        {row[identity_field]: row["score"] for row in summary["metrics"][group_field]}
        for summary in summaries
    ]
    series = [("总体", [100 * summary["metrics"]["score"] for summary in summaries], "#1d3f55")]
    for index, row in enumerate(groups):
        label = row["category_name"]
        series.append(
            (
                label,
                [
                    100 * scores[row[identity_field]] if row[identity_field] in scores else None
                    for scores in grouped
                ],
                _group_color(index, len(groups)),
            )
        )
    references = []
    method = Path(first["training_run"]).name
    series = [(f"{method} / {label}", values, color) for label, values, color in series]
    for baseline_label, baseline in baselines.items():
        reference_groups = {
            row[identity_field]: row["score"] for row in baseline["metrics"][group_field]
        }
        references.append((
            f"{baseline_label} / 总体",
            [100 * baseline["metrics"]["score"]] * len(steps),
            _baseline_color(baseline_label),
        ))
        for index, row in enumerate(groups):
            label = row["category_name"]
            references.append((
                f"{baseline_label} / {label}",
                [
                    100 * reference_groups[row[identity_field]]
                    if row[identity_field] in reference_groups
                    else None
                    for scores in grouped
                ],
                _baseline_color(baseline_label, index, len(groups)),
            ))
    return {
        f"evaluation/{benchmark}": _line_chart(
            steps, series, f"{first['metrics']['metric']} (%)", 100, references
        )
    }
