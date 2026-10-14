"""官方 benchmark 分数的汇总图；完整分组统计保存在 summary.json。"""

import swanlab

from latent_working_memory.v4.benchmarks.data import RULER_TASKS


def benchmark_media(summary):
    metrics = summary["metrics"]
    chart = swanlab.echarts.Bar()
    if summary["benchmark"] == "ruler":
        chart.add_xaxis([*RULER_TASKS, "平均"])
        grouped = {(row["length"], row["task"]): row["score"] for row in metrics["groups"]}
        lengths = metrics["by_length"]
        for index, row in enumerate(lengths):
            fraction = 0.6 * (1 - index / max(1, len(lengths) - 1))
            color = "#" + "".join(
                f"{round(channel + (255 - channel) * fraction):02x}" for channel in (49, 104, 142)
            )
            values = [grouped[(row["length"], task)] for task in RULER_TASKS]
            chart.add_yaxis(
                f"{row['length'] / 1024:g}K",
                [100 * value for value in (*values, row["score"])],
                label_opts={"show": False},
                itemstyle_opts={"color": color},
            )
    else:
        groups = metrics["groups"]
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
