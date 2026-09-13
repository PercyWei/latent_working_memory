from __future__ import annotations

import json
import colorsys
from collections import defaultdict
from pathlib import Path
from typing import Any

import swanlab


CHART_METRICS = (
    "nll",
    "ppl",
    "correct_prefix_ratio",
    "bleu_4",
    "exact_match",
)


def reconstruction_media(records_path: Path, prefix: str) -> dict[str, Any]:
    examples = []
    for line in records_path.read_text().splitlines():
        record = json.loads(line)
        if "prediction" in record:
            examples.append(
                swanlab.Text(
                    f"Reference:\n{record['reference']}\n\nPrediction:\n{record['prediction']}",
                    caption=f"{record['condition']}, {record['input_tokens']} tokens, K={record['capacity']}, "
                    f"prefix={record['correct_prefix_ratio']:.4f}",
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


CONDITION_COLORS = {
    "memory": "#2459A6",
    "wrong_memory": "#B45B18",
    "no_memory": "#626B73",
    "full_context": "#28764A",
    "base_full_context": "#7951A0",
}


def _shade(color: str, source_index: int, source_count: int) -> str:
    color = color.lstrip("#")
    rgb = [int(color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    h, light, saturation = colorsys.rgb_to_hls(*rgb)
    light += 0.25 * source_index / max(source_count - 1, 1)
    return "#" + "".join(f"{round(c * 255):02x}" for c in colorsys.hls_to_rgb(h, light, saturation))


def _bar(
    labels: list[str],
    series: dict[str, list[Any]],
    hue_groups: list[str],
    hue_colors: dict[str, str],
    hue_dimension: str,
) -> Any:
    chart = swanlab.echarts.Bar().add_xaxis(
        [label.replace("_", "\n").replace("/r", "\nr") for label in labels]
    )
    for group in dict.fromkeys(hue_groups):
        for source_index, (source, values) in enumerate(series.items()):
            points = [round(value, 4) if isinstance(value, float) else value for value in values]
            points = [
                value if hue == group else None
                for value, hue in zip(points, hue_groups, strict=True)
            ]
            chart.add_yaxis(
                f"{hue_dimension}={group} / test={source}",
                points,
                stack=source,
                label_opts={"show": False},
                itemstyle_opts={"color": _shade(hue_colors[group], source_index, len(series))},
            )
    chart.set_global_opts(
        tooltip_opts={"trigger": "axis"},
        legend_opts={"type": "scroll", "top": 0},
        xaxis_opts={"axisLabel": {"interval": 0, "rotate": 0, "fontSize": 10, "lineHeight": 11}},
        yaxis_opts={"minInterval": 0.001, "splitNumber": 3},
    )
    chart.options["grid"] = {
        "left": "12%",
        "right": "4%",
        "top": "20%",
        "height": "55%",
        "containLabel": False,
    }
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
            values[f"tables/{prefix}/{section}/{category}"] = _table(entries)
    panels = defaultdict(dict)
    for source, metrics in reports:
        for key, summary in metrics["groups"].items():
            parts = key.split("/")
            category = parts[0]
            if category == "all":
                _, task, condition = parts
                bucket = condition
            else:
                category, length, ratio, task, condition = parts
                if condition != "memory":
                    continue
                bucket = f"{length}/r{ratio}"
            for metric in CHART_METRICS:
                if metric in summary:
                    panels[(category, task, metric)].setdefault(source, {})[bucket] = summary[
                        metric
                    ]
    for (category, task, metric), series in panels.items():
        labels = list(dict.fromkeys(label for points in series.values() for label in points))
        if category == "all":
            labels.sort(key=list(CONDITION_COLORS).index)
        else:
            labels.sort(key=lambda label: tuple(map(float, label.split("/r"))))
        values[f"charts/{prefix}/{category}/{task}/{metric}"] = _bar(
            labels,
            {name: [points.get(label) for label in labels] for name, points in series.items()},
            labels if category == "all" else ["memory"] * len(labels),
            CONDITION_COLORS,
            "condition",
        )
    return values


def build_test_report_charts(metrics: dict[str, Any], prefix: str = "report") -> dict[str, Any]:
    return build_evaluation_charts([("", metrics)], prefix)


def log_test_report(run, metrics, records_path: Path, prefix: str = "report") -> None:
    if run is not None:
        run.log(
            build_test_report_charts(metrics, prefix)
            | reconstruction_media(records_path, f"examples/{prefix}")
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
    palette = ("#2459A6", "#B45B18", "#28764A", "#7951A0", "#626B73")
    if len(training_sources) > len(palette):
        raise ValueError("comparison needs an additional distinct training-source color")
    colors = dict(zip(training_sources, palette))
    values = {"tables/comparison/summary": _table(rows)}
    for (task, metric), series in panels.items():
        values[f"charts/comparison/{task}/{metric}"] = _bar(
            training_sources,
            {
                test: [points.get(train) for train in training_sources]
                for test, points in series.items()
            },
            training_sources,
            colors,
            "train",
        )
    return values


OVERVIEW_METRICS = (("ae", "nll"), ("continuation", "nll"), ("ae", "bleu_4"),
                    ("ae", "correct_prefix_ratio"), ("ae", "exact_match"))


def evaluation_overview(reports, prefix):
    """Five comparable panels; redundant and stratified metrics remain in two tables."""
    values = {}
    for task, metric in OVERVIEW_METRICS:
        conditions = [c for c in CONDITION_COLORS if any(
            metric in report["groups"].get(f"all/{task}/{c}", {}) for _, report in reports
        )]
        if conditions:
            values[f"{prefix}/{task}/{metric}"] = _bar(
                conditions,
                {source: [report["groups"].get(f"all/{task}/{c}", {}).get(metric)
                          for c in conditions] for source, report in reports},
                conditions, CONDITION_COLORS, "condition",
            )
    values.update(evaluation_tables(reports, prefix))
    return values


def evaluation_tables(reports, prefix):
    summary, details = [], []
    for source, report in reports:
        for section in ("groups", "comparisons", "prefix_diagnostics"):
            for group, metrics in report.get(section, {}).items():
                row = {"source": source, "section": section, "group": group, **metrics}
                (summary if section == "groups" and group.startswith("all/") else details).append(row)
    return {f"{prefix}/{name}": _table(rows)
            for name, rows in (("summary", summary), ("details", details)) if rows}


def paired_reconstructions(paths, prefix):
    """Keep all generated examples, with controls of the same input in one text card."""
    samples = {}
    for source, path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "prediction" not in row:
                continue
            key = source, row["episode_id"], row["capacity"]
            samples.setdefault(key, []).append(row)
    texts = []
    for (source, episode, capacity), rows in samples.items():
        texts.append(swanlab.Text(
            "Reference:\n" + rows[0]["reference"] + "\n\n" + "\n\n".join(
                f"{row['condition']} (prefix={row['correct_prefix_ratio']:.4f}):\n{row['prediction']}"
                for row in rows
            ), caption=f"{source}, {episode}, K={capacity}",
        ))
    return {f"{prefix}/examples" + (f"/page_{start // 100 + 1}" if start else ""):
            texts[start:start + 100] for start in range(0, len(texts), 100)}


def development_overview(paths, step):
    """Reconstruct cumulative dev curves from saved reports, including sparse generation steps."""
    histories = {
        source: [(int(path.stem.removeprefix("dev-step-")), json.loads(path.read_text()))
                 for path in sorted(records.parent.glob("dev-step-*.json"))
                 if int(path.stem.removeprefix("dev-step-")) <= step]
        for source, records in paths
    }
    values = {}
    for task, metric in OVERVIEW_METRICS:
        observed = sorted({n for history in histories.values() for n, report in history
                           if any(metric in report["groups"].get(f"all/{task}/{c}", {})
                                  for c in CONDITION_COLORS)})
        if not observed:
            continue
        chart = swanlab.echarts.Line().add_xaxis(observed)
        for source_index, (source, history) in enumerate(histories.items()):
            lookup = dict(history)
            for condition, color in CONDITION_COLORS.items():
                points = [lookup.get(n, {}).get("groups", {}).get(
                    f"all/{task}/{condition}", {}).get(metric) for n in observed]
                if any(value is not None for value in points):
                    chart.add_yaxis(
                        f"{source}/{condition}", points, is_smooth=False, is_connect_nones=False,
                        label_opts={"show": False},
                        itemstyle_opts={"color": _shade(color, source_index, len(histories))},
                    )
        chart.set_global_opts(
            xaxis_opts={"type": "value", "name": "optimizer step"},
            yaxis_opts={"name": metric}, tooltip_opts={"trigger": "axis"},
            legend_opts={"type": "scroll"},
        )
        values[f"dev/overview/{task}/{metric}"] = chart
    latest = [(source, history[-1][1]) for source, history in histories.items() if history]
    values.update(evaluation_tables(latest, "dev/overview"))
    values.update(paired_reconstructions(paths, "dev/overview"))
    return values
