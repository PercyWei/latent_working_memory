from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import swanlab

from latent_working_memory.v1.reporting import CONDITION_COLORS, _table, _bar, _shade, configure_line_panels


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


OVERVIEW_METRICS = (
    ("ae", "nll"),
    ("continuation", "nll"),
    ("ae", "bleu_4"),
    ("ae", "correct_prefix_ratio"),
    ("ae", "exact_match"),
)


def evaluation_overview(reports, prefix):
    """Five comparable panels; redundant and stratified metrics remain in two tables."""
    values = {}
    for task, metric in OVERVIEW_METRICS:
        conditions = [
            c
            for c in CONDITION_COLORS
            if any(metric in report["groups"].get(f"all/{task}/{c}", {}) for _, report in reports)
        ]
        if conditions:
            values[f"{prefix}/{task}/{metric}"] = _bar(
                conditions,
                {
                    source: [
                        report["groups"].get(f"all/{task}/{c}", {}).get(metric) for c in conditions
                    ]
                    for source, report in reports
                },
                conditions,
                CONDITION_COLORS,
                "condition",
            )
    values.update(evaluation_tables(reports, prefix))
    return values


def evaluation_tables(reports, prefix):
    summary, details = [], []
    for source, report in reports:
        for section in ("groups", "comparisons", "prefix_diagnostics"):
            for group, metrics in report.get(section, {}).items():
                row = {"source": source, "section": section, "group": group, **metrics}
                (summary if section == "groups" and group.startswith("all/") else details).append(
                    row
                )
    return {
        f"{prefix}/{name}": _table(rows)
        for name, rows in (("summary", summary), ("details", details))
        if rows
    }


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
        texts.append(
            swanlab.Text(
                "Reference:\n"
                + rows[0]["reference"]
                + "\n\n"
                + "\n\n".join(
                    f"{row['condition']} (prefix={row['correct_prefix_ratio']:.4f}):\n{row['prediction']}"
                    for row in rows
                ),
                caption=f"{source}, {episode}, K={capacity}",
            )
        )
    return {
        f"{prefix}/examples" + (f"/page_{start // 100 + 1}" if start else ""): texts[
            start : start + 100
        ]
        for start in range(0, len(texts), 100)
    }


def development_scalars(reports):
    return {
        f"dev/overview/{task}/{metric}/{source}/{condition}": group[metric]
        for source, report in reports.items()
        for task, metric in OVERVIEW_METRICS
        for condition in CONDITION_COLORS
        if metric in (group := report["groups"].get(f"all/{task}/{condition}", {}))
    }


def development_panels(sources):
    panels = {}
    for task, metric in OVERVIEW_METRICS:
        conditions = [c for c in CONDITION_COLORS if task != "ae" or c != "no_memory"]
        # The cloud native LINE API permits at most eight Y metrics per panel.
        groups = (
            [("", sources)]
            if len(sources) * len(conditions) <= 8
            else [(f"/{source}", [source]) for source in sources]
        )
        for suffix, selected in groups:
            title = f"dev/overview/{task}/{metric}{suffix}"
            panels[title] = {
                "title": title,
                "config": {
                    "xAxis": {"key": "step", "name": "step", "type": "FLOAT", "class": "SYSTEM"},
                    "yAxis": [
                        {"key": key, "name": key, "type": "FLOAT", "class": "CUSTOM"}
                        for condition in conditions
                        for source in selected
                        for key in [f"dev/overview/{task}/{metric}/{source}/{condition}"]
                    ],
                    "xName": "optimizer step",
                    "yName": metric,
                },
            }
    return panels


def development_panel_style(panel, sources, run_id):
    custom = {}
    for axis in panel["config"]["yAxis"]:
        source, condition = axis["key"].split("/")[-2:]
        color = _shade(CONDITION_COLORS[condition], sources.index(source), len(sources))
        custom[f"{run_id}-{axis['key']}"] = {
            "name": f"{source}/{condition}",
            "colors": [color, color],
        }
    return custom


def configure_development_panels(run, sources, mode):
    configure_line_panels(
        run,
        development_panels(list(sources)),
        mode,
        lambda panel, run_id: development_panel_style(panel, list(sources), run_id),
    )
