"""从完整 checkpoint 集合重建评估图，累计 step 决定曲线顺序。"""

from colorsys import hsv_to_rgb
from hashlib import sha256

from pyecharts import charts

from latent_working_memory.benchmarks.data import RULER_TASKS


def evaluation_charts(summaries, baselines=None):
    """summaries 按真实累计 step 排序；RULER 各 checkpoint 可覆盖不同长度。"""
    baselines = {} if baselines is None else baselines
    if baselines and "benchmark" not in summaries[0]:
        raise ValueError("baseline results apply only to benchmark reports")
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

    write_loss = charts.Bar().add_xaxis(["写入前", "写入后"])
    write_loss.add_yaxis(
        "attention-matching",
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
    method = summary.get("method", "attention-matching")
    chart = charts.Bar()
    if summary["benchmark"] == "ruler":
        chart.add_xaxis([*RULER_TASKS, "平均"])
        grouped = {(row["length"], row["task"]): row["score"] for row in metrics["groups"]}
        lengths = sorted(metrics["by_length"], key=lambda row: row["length"])
        for index, row in enumerate(lengths):
            values = [grouped[(row["length"], task)] for task in RULER_TASKS]
            label = f"{row['length'] / 1024:g}K"
            chart.add_yaxis(
                f"{method} / {label}" if baselines else label,
                [100 * value for value in (*values, row["score"])],
                label_opts={"show": False},
                itemstyle_opts={"color": _group_color(index, len(lengths))},
            )
        for baseline_label, baseline in baselines.items():
            reference_groups = {
                (row["length"], row["task"]): row["score"]
                for row in baseline["metrics"]["groups"]
            }
            reference_lengths = {
                row["length"]: row["score"] for row in baseline["metrics"]["by_length"]
            }
            for index, row in enumerate(lengths):
                label = f"{row['length'] / 1024:g}K"
                reference = [reference_groups.get((row["length"], task)) for task in RULER_TASKS]
                chart.add_yaxis(
                    f"{baseline_label} / {label}",
                    [
                        100 * value if value is not None else None
                        for value in (*reference, reference_lengths.get(row["length"]))
                    ],
                    label_opts={"show": False},
                    itemstyle_opts={"color": _baseline_color(baseline_label, index, len(lengths))},
                )
    else:
        groups = sorted(metrics["groups"], key=lambda row: row["category"])
        chart.add_xaxis([*[row["category_name"] for row in groups], "总体"])
        chart.add_yaxis(
            method,
            [100 * row["score"] for row in groups] + [100 * metrics["score"]],
            label_opts={"show": False},
            itemstyle_opts={"color": "#31688e"},
        )
        for baseline_label, baseline in baselines.items():
            reference_groups = {
                row["category"]: row["score"] for row in baseline["metrics"]["groups"]
            }
            chart.add_yaxis(
                baseline_label,
                [
                    {
                        "value": 100 * reference_groups[row["category"]],
                        "itemStyle": {
                            "color": _baseline_color(baseline_label, index, len(groups))
                        },
                    }
                    for index, row in enumerate(groups)
                ]
                + [{
                    "value": 100 * baseline["metrics"]["score"],
                    "itemStyle": {"color": _baseline_color(baseline_label)},
                }],
                label_opts={"show": False},
                itemstyle_opts={"color": _baseline_color(baseline_label)},
            )
    chart.set_global_opts(
        tooltip_opts={"trigger": "axis"},
        yaxis_opts={"name": f"{metrics['metric']} (%)", "min": 0, "max": 100},
    )
    return {f"evaluation/{summary['benchmark']}": chart}


def _line_chart(steps, series, unit, maximum=None, references=()):
    chart = charts.Line().add_xaxis(steps)
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
            (("write_loss_before", "写入前", "#74a9cf"), ("write_loss", "写入后", "#2b8cbe")),
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
    group_field, identity_field = (
        ("by_length", "length") if benchmark == "ruler" else ("groups", "category")
    )
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
        label = f"{row['length'] / 1024:g}K" if benchmark == "ruler" else row["category_name"]
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
    if baselines:
        method = first.get("method", "attention-matching")
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
            label = f"{row['length'] / 1024:g}K" if benchmark == "ruler" else row["category_name"]
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
