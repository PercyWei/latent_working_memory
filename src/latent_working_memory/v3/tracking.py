"""v3 的增量训练曲线与最终评估汇总；运行身份复用 v1 tracking。"""

import swanlab

from latent_working_memory.v1.reporting import _shade, _table


TRAINING_METRICS = (
    "train/loss",
    "train/grad_norm",
    "train/slots_final",
    "dev/loss",
    "dev/slots_final",
    "resources/optimizer_step_seconds",
    "resources/peak_memory_allocated_gib",
)
METHOD_COLORS = {
    "icae_single": "#2459A6",
    "icae_multi": "#28764A",
    "autocompressors": "#7951A0",
    "memory_change": "#B45B18",
    "information_loss": "#A53651",
}
QUALITY_GROUPS = ("all", "old", "new")


def configure_training_metrics(tracking):
    """在首个 log 前定义面板，横轴使用 log(step=真实 optimizer step)。"""
    if tracking is None:
        return
    for name in TRAINING_METRICS:
        tracking.define_metric(name, x_axis="_step", section_name=name.split("/", 1)[0])


def training_metrics(record):
    """只上传核心增量指标；计数、动作与门控诊断完整保留在 metrics.jsonl。"""
    values = {name: record[name] for name in TRAINING_METRICS if record.get(name) is not None}
    if "resources/peak_memory_allocated_bytes" in record:
        values["resources/peak_memory_allocated_gib"] = (
            record["resources/peak_memory_allocated_bytes"] / 1024**3
        )
    return values


def evaluation_media(summary, rows):
    """每项核心质量指标合并新旧问题；辅助统计与少量生成样例集中入表。"""
    method = summary["method"]
    color = METHOD_COLORS[method]
    groups = [name for name in QUALITY_GROUPS if summary["quality"][name]["questions"]]
    values = {}
    for metric in ("nll", "em", "f1"):
        chart = swanlab.echarts.Bar().add_xaxis(groups)
        chart.add_yaxis(
            method,
            [
                {
                    "value": summary["quality"][group][metric],
                    "itemStyle": {"color": _shade(color, index, len(groups))},
                }
                for index, group in enumerate(groups)
            ],
            label_opts={"show": False},
            itemstyle_opts={"color": color},
        )
        chart.set_global_opts(
            tooltip_opts={"trigger": "axis"},
            legend_opts={"type": "scroll", "top": 0},
            yaxis_opts={"name": metric, **({"min": 0, "max": 1} if metric != "nll" else {})},
        )
        values[f"evaluation/{metric}"] = chart
    values["evaluation/summary"] = _table(
        [
            {
                "method": method,
                "split": summary["split"],
                "offline_oracle": summary["offline_oracle"],
                "trajectories": summary["trajectories"],
                "group": group,
                **summary["quality"][group],
                **summary["capacity"],
            }
            for group in groups
        ]
    )
    details = [
        {"category": "age", "name": age, **metrics}
        for age, metrics in summary["quality"]["by_age"].items()
    ]
    details.extend(
        {"category": "cost", "name": name, "value": value}
        for name, value in summary["costs"].items()
    )
    details.extend(
        {"category": "selection", "name": name, "value": value}
        for name, value in summary["metadata"].get("selection", {}).items()
    )
    values["evaluation/details"] = _table(details)
    examples = [
        {
            "trajectory_id": row["trajectory_id"],
            "qa_id": question["qa_id"],
            "age": question["age"],
            "question": question["question"],
            "answer": question["answer"],
            "prediction": question["prediction"],
            **{name: question[name] for name in ("nll", "em", "f1")},
        }
        for row in rows
        for question in row["questions"]
    ][:8]
    if examples:
        values["evaluation/examples"] = _table(examples)
    return values
