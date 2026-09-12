from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import swanlab


CHART_METRICS = (
    "nll",
    "ppl",
    "token_accuracy",
    "sequence_match",
    "normalized_token_edit_distance",
    "correct_prefix_ratio",
    "bleu_4",
)


def reconstruction_media(records_path: Path, prefix: str) -> dict[str, Any]:
    examples = []
    for line in records_path.read_text().splitlines():
        record = json.loads(line)
        if "prediction" in record:
            examples.append(
                swanlab.Text(
                    f"Reference:\n{record['reference']}\n\nPrediction:\n{record['prediction']}",
                    caption=f"{record['input_tokens']} tokens, K={record['capacity']}, "
                    f"exact={record['sequence_match']}, prefix={record['correct_prefix_ratio']:.3f}",
                )
            )
    return {
        f"{prefix}/reconstruction" + ("" if start == 0 else f"/page_{start // 100 + 1}"): examples[
            start : start + 100
        ]
        for start in range(0, len(examples), 100)
    }


def _table(rows: list[dict[str, Any]]) -> Any:
    headers = list(dict.fromkeys(key for row in rows for key in row))
    return swanlab.echarts.Table().add(
        headers,
        [
            [round(row[k], 4) if isinstance(row.get(k), float) else row.get(k) for k in headers]
            for row in rows
        ],
    )


def _bar(labels: list[str], series: dict[str, list[Any]]) -> Any:
    chart = swanlab.echarts.Bar().add_xaxis(labels)
    for label, values in series.items():
        chart.add_yaxis(
            label,
            [round(v, 4) if isinstance(v, float) else v for v in values],
            label_opts={"is_show": False},
        )
    chart.set_global_opts(
        tooltip_opts={"trigger": "axis"},
        legend_opts={"type": "scroll"},
        xaxis_opts={"axislabel_opts": {"rotate": 20}},
        yaxis_opts={"min_interval": 0.001},
    )
    return chart


def build_evaluation_charts(
    reports: list[tuple[str, dict[str, Any]]], prefix: str = "report"
) -> dict[str, Any]:
    """Compare evaluation sources and controls for one checkpoint."""
    values = {}
    for section in ("groups", "comparisons"):
        rows = defaultdict(list)
        for source, metrics in reports:
            for key, summary in metrics[section].items():
                category, label = key.split("/", 1)
                rows[category].append({"evaluation_source": source, "group": label, **summary})
        for category, entries in rows.items():
            values[f"{prefix}/tables/{section}/{category}"] = _table(entries)
    panels = defaultdict(dict)
    for source, metrics in reports:
        for key, summary in metrics["groups"].items():
            parts = key.split("/")
            category = parts[0]
            if category == "all":
                _, task, condition = parts
                bucket = condition
                series = source or task
            else:
                category, bucket, task, condition = parts
                series = f"{source}/{condition}" if source else condition
            for metric in CHART_METRICS:
                if metric in summary:
                    panels[(category, task, metric)].setdefault(series, {})[bucket] = summary[
                        metric
                    ]
    for (category, task, metric), series in panels.items():
        labels = list(dict.fromkeys(label for points in series.values() for label in points))
        if category in {"length_up_to", "ratio_up_to", "capacity"}:
            labels.sort(key=float)
        values[f"{prefix}/{category}/{task}/{metric}"] = _bar(
            labels,
            {name: [points.get(label) for label in labels] for name, points in series.items()},
        )
    return values


def build_test_report_charts(metrics: dict[str, Any], prefix: str = "report") -> dict[str, Any]:
    return build_evaluation_charts([("", metrics)], prefix)


def log_test_report(run, metrics, records_path: Path, prefix: str = "report") -> None:
    if run is not None:
        run.log(
            build_test_report_charts(metrics, prefix) | reconstruction_media(records_path, prefix)
        )


def comparison_charts(reports: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    panels = defaultdict(dict)
    training_sources = list(dict.fromkeys(train for train, _, _ in reports))
    for train, test, report in reports:
        for key, summary in report["groups"].items():
            if not key.startswith("all/"):
                continue
            _, task, condition = key.split("/")
            rows.append(
                {
                    "training_source": train,
                    "evaluation_source": test,
                    "task": task,
                    "condition": condition,
                    **summary,
                }
            )
            if condition == "memory":
                for metric in CHART_METRICS:
                    if metric in summary:
                        panels[(task, metric)].setdefault(test, {})[train] = summary[metric]
    values = {"comparison/summary": _table(rows)}
    for (task, metric), series in panels.items():
        values[f"comparison/{task}/{metric}"] = _bar(
            training_sources,
            {
                test: [points.get(train) for train in training_sources]
                for test, points in series.items()
            },
        )
    return values
