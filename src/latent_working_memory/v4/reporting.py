"""从完整 checkpoint 集合重建评估图，累计 step 决定曲线顺序。"""

import swanlab

from latent_working_memory.v4.benchmarks.data import RULER_TASKS


def evaluation_media(summaries):
    """summaries 为按真实累计 step 排序、协议一致的非空评估结果。"""
    if len(summaries) == 1:
        summary = summaries[0]
        return _benchmark_media(summary) if "benchmark" in summary else _test_media(summary)
    return _checkpoint_media(summaries)


def _test_media(summary):
    metrics = summary["metrics"]
    quality = swanlab.echarts.Bar().add_xaxis(["NLL", "EM", "F1"])
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

    write_loss = swanlab.echarts.Bar().add_xaxis(["写入前", "写入后"])
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


def _benchmark_media(summary):
    metrics = summary["metrics"]
    chart = swanlab.echarts.Bar()
    if summary["benchmark"] == "ruler":
        chart.add_xaxis([*RULER_TASKS, "平均"])
        grouped = {(row["length"], row["task"]): row["score"] for row in metrics["groups"]}
        lengths = sorted(metrics["by_length"], key=lambda row: row["length"])
        for index, row in enumerate(lengths):
            values = [grouped[(row["length"], task)] for task in RULER_TASKS]
            chart.add_yaxis(
                f"{row['length'] / 1024:g}K",
                [100 * value for value in (*values, row["score"])],
                label_opts={"show": False},
                itemstyle_opts={"color": _group_color(index, len(lengths))},
            )
    else:
        groups = sorted(metrics["groups"], key=lambda row: row["category"])
        chart.add_xaxis([*[row["category_name"] for row in groups], "总体"])
        chart.add_yaxis(
            "attention-matching",
            [100 * row["score"] for row in groups] + [100 * metrics["score"]],
            label_opts={"show": False},
            itemstyle_opts={"color": "#31688e"},
        )
    chart.set_global_opts(
        tooltip_opts={"trigger": "axis"},
        yaxis_opts={"name": f"{metrics['metric']} (%)", "min": 0, "max": 100},
    )
    return {f"evaluation/{summary['benchmark']}": chart}


def _line_chart(steps, series, unit, maximum=None):
    chart = swanlab.echarts.Line().add_xaxis(steps)
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


def _checkpoint_media(summaries):
    steps = [summary["checkpoint"]["step"] for summary in summaries]
    if "benchmark" in summaries[0]:
        return _benchmark_checkpoint_media(summaries, steps)
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


def _benchmark_checkpoint_media(summaries, steps):
    first = summaries[0]
    benchmark = first["benchmark"]
    group_field, identity_field = (
        ("by_length", "length") if benchmark == "ruler" else ("groups", "category")
    )
    groups = sorted(first["metrics"][group_field], key=lambda row: row[identity_field])
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
                [100 * scores[row[identity_field]] for scores in grouped],
                _group_color(index, len(groups)),
            )
        )
    return {
        f"evaluation/{benchmark}": _line_chart(
            steps, series, f"{first['metrics']['metric']} (%)", 100
        )
    }
