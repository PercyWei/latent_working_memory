"""Publish QA charts, examples and comparisons from saved predictions."""

import argparse
import json
from pathlib import Path

import swanlab

from latent_working_memory.v1.dynamic_evaluation import aggregate_qa
from latent_working_memory.v1.tracking import swanlab_run


CONDITIONS = ("memory", "no_memory", "wrong_memory", "gold_paragraph", "gold_paragraph_base")


def table(rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    return swanlab.echarts.Table().add(
        columns,
        [
            [round(row[k], 4) if isinstance(row.get(k), float) else row.get(k) for k in columns]
            for row in rows
        ],
    )


def bar(labels, series):
    chart = swanlab.echarts.Bar().add_xaxis(labels)
    for name, values in series.items():
        chart.add_yaxis(name, [round(v, 4) if isinstance(v, float) else v for v in values])
    chart.set_global_opts(tooltip_opts={"trigger": "axis"}, legend_opts={"type": "scroll"})
    return chart


def qa_media(metrics, rows, prefix="test"):
    media = {}
    for metric in ("nll", "em", "f1", "hit_limit_rate"):
        series = {
            kind: [metrics.get(f"overall/{c}/{kind}", {}).get(metric) for c in CONDITIONS]
            for kind in ("all", "arrival", "delayed")
        }
        if any(v is not None for values in series.values() for v in values):
            media[f"evaluation/{prefix}/{metric}"] = bar(list(CONDITIONS), series)
    for axis in ("delay-tokens-le", "delay-updates-le"):
        labels = sorted(
            {key.split("/")[0] for key in metrics if key.startswith(axis)},
            key=lambda s: int(s.removeprefix(axis)),
        )
        for metric in ("nll", "f1"):
            if labels:
                media[f"evaluation/{prefix}/{axis}/{metric}"] = bar(
                    labels,
                    {
                        c: [metrics.get(f"{label}/{c}/all", {}).get(metric) for label in labels]
                        for c in CONDITIONS
                    },
                )
    media[f"tables/{prefix}/capacity-and-ratio"] = table(
        [
            {"group": key, **value}
            for key, value in metrics.items()
            if key.startswith("k") and key.endswith("/memory/all")
        ]
    )
    media[f"tables/{prefix}/paired"] = table(
        [
            {"comparison": key, **value}
            for key, value in metrics.items()
            if key.startswith("paired/")
        ]
    )
    examples = {}
    for row in rows:
        if "prediction" in row:
            key = row["capacity"], row["episode_id"], row["read_id"], row["prefix_end"]
            if key not in examples and len(examples) >= 10:
                continue
            examples.setdefault(key, []).append(row)
    media[f"examples/{prefix}/qa"] = [
        swanlab.Text(
            sample[0]["question"]
            + "\n\nReferences: "
            + json.dumps(sample[0]["references"], ensure_ascii=False)
            + "\n\n"
            + "\n".join(f"{r['condition']}: {r['prediction']}" for r in sample),
            caption=f"K={key[0]}, {sample[0]['kind']}, delay={sample[0]['delay_tokens']} tokens",
        )
        for key, sample in examples.items()
    ]
    return media


def training_curves(history_dir, through_step):
    history = sorted(
        (int(path.stem.removeprefix("dev-step-")), json.loads(path.read_text()))
        for path in history_dir.glob("dev-step-*.json")
        if int(path.stem.removeprefix("dev-step-")) <= through_step
    )
    capacities = sorted(
        {
            int(key.split("/")[0][1:])
            for _, report in history
            for key in report
            if key.startswith("k") and key.count("/") == 2 and key.endswith("/memory/all")
        }
    )
    groups = {
        "": {c: f"overall/{c}" for c in CONDITIONS},
        "by-capacity/": {f"K={k}": f"k{k}/memory" for k in capacities},
    }
    charts = {}
    for group, series in groups.items():
        for metric in ("nll", "em", "f1", "hit_limit_rate"):
            views = {
                kind: (metric, {label: f"{key}/{kind}" for label, key in series.items()})
                for kind in ("all", "arrival", "delayed")
            }
            if group == "" and metric != "hit_limit_rate":
                views["paired"] = (
                    f"{metric}_difference",
                    {c: f"paired/memory-minus-{c}" for c in CONDITIONS if c != "memory"},
                )
            options, labels = [], []
            for view, (value_key, view_series) in views.items():
                observed = [
                    (step, report)
                    for step, report in history
                    if any(value_key in report.get(key, {}) for key in view_series.values())
                ]
                if not observed:
                    continue
                chart = swanlab.echarts.Line().add_xaxis([step for step, _ in observed])
                for label, key in view_series.items():
                    chart.add_yaxis(
                        label,
                        [report.get(key, {}).get(value_key) for _, report in observed],
                        is_smooth=False,
                        is_connect_nones=False,
                        label_opts={"show": False},
                    )
                chart.set_global_opts(
                    xaxis_opts={"type": "value", "name": "optimizer step", "min": 0},
                    yaxis_opts={"name": value_key},
                    tooltip_opts={"trigger": "axis"},
                    legend_opts={"type": "scroll"},
                )
                options.append(chart.options)
                labels.append(view)
            if options:
                chart = swanlab.echarts.Line()
                chart.options = {
                    # SwanLab checks the rendered series count against this top-level field.
                    "series": options[0]["series"],
                    "baseOption": {
                        "timeline": {
                            "axisType": "category",
                            "autoPlay": False,
                            "currentIndex": 0,
                            "data": labels,
                            "bottom": 0,
                            "left": "15%",
                            "right": "15%",
                            "controlStyle": {"show": False},
                            "replaceMerge": ["series"],
                        },
                        "grid": {"left": "12%", "right": "15%", "top": 70, "bottom": 90},
                    },
                    "options": options,
                }
                charts[f"evaluation/dev/{group}{metric}"] = chart
    return charts


def training_metrics(record):
    return {
        f"{'resources' if key in {'seconds', 'peak_memory_bytes', 'input_tokens_per_second'} else 'train'}/{key}": value
        for key, value in record.items()
        if isinstance(value, (int, float))
    } | {f"train/cumulative_{key}": value for key, value in record["cumulative"].items()}


def log_qa(run, metrics, rows, step, prefix, media=False, history_dir=None):
    if run is None:
        return
    values = training_curves(history_dir, step) if history_dir is not None else {}
    if media:
        values.update(
            {
                key: value
                for key, value in qa_media(metrics, rows, prefix).items()
                if history_dir is None or not key.startswith("evaluation/")
            }
        )
    run.log(values, step=step)


def publish_training_history(directory, media_step, mode, output_dir=None):
    config = json.loads((directory / "config.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    total_steps = provenance["target_steps"]
    completed = [
        json.loads(p.read_text())["completed_steps"]
        for p in directory.glob("resources-from-*.json")
    ]
    if total_steps not in completed:
        raise ValueError("history publishing requires a completed training run")
    if media_step is None or media_step <= total_steps:
        raise ValueError("media step must follow the completed training steps")
    output_dir = directory if output_dir is None else output_dir
    identity = json.loads((output_dir / "swanlab.json").read_text())
    if output_dir != directory:
        provenance["report_source"] = json.loads((output_dir / "republication.json").read_text())[
            "source"
        ]
    charts = training_curves(directory / "dev", total_steps)
    with swanlab_run(
        output_dir,
        provenance,
        mode,
        identity["project"],
        job_type="train",
        group=identity["group"],
        tags=tuple(identity["tags"]),
        fixed_tags=(),
    ) as run:
        if run is not None:
            run.log(charts, step=media_step)
    (output_dir / f"evaluation-history-{media_step:06d}.json").write_text(
        json.dumps(
            {
                "checkpoint_step": total_steps,
                "media_step": media_step,
                "eval_every": config["eval_every"],
                "eval_generation_every": config["eval_generation_every"],
                "reports": [p.name for p in sorted((directory / "dev").glob("dev-step-*.json"))],
            },
            indent=2,
        )
        + "\n"
    )


def rebuild_training_run(directory, output_dir, mode):
    provenance = json.loads((directory / "provenance.json").read_text())
    identity = json.loads((directory / "swanlab.json").read_text())
    total_steps = provenance["target_steps"]
    completed = [
        json.loads(path.read_text())["completed_steps"]
        for path in directory.glob("resources-from-*.json")
    ]
    if total_steps not in completed:
        raise ValueError("rebuilding requires a completed training run")
    records = sorted(
        (
            json.loads(line)
            for path in directory.glob("train-from-*.jsonl")
            for line in path.read_text().splitlines()
        ),
        key=lambda record: record["step"],
    )
    if [record["step"] for record in records] != list(range(1, total_steps + 1)):
        raise ValueError("rebuilding requires exactly one training record per step")
    reports = {
        int(path.stem.removeprefix("dev-step-")): (
            json.loads(path.read_text()),
            [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()],
        )
        for path in sorted((directory / "dev").glob("dev-step-*.json"))
    }
    if 0 not in reports or total_steps not in reports or max(reports) > total_steps:
        raise ValueError("rebuilding requires initial and final dev reports within training steps")
    source = {"training_run": str(directory.resolve()), "swanlab_id": identity["id"]}
    output_dir.mkdir(parents=True, exist_ok=False)
    with swanlab_run(
        output_dir,
        {**provenance, "report_source": source},
        mode,
        identity["project"],
        job_type="train",
        group=identity["group"],
        tags=tuple(identity["tags"]),
        fixed_tags=(),
    ) as run:
        for step in range(total_steps + 1):
            if run is not None and step:
                run.log(training_metrics(records[step - 1]), step=step)
            if step in reports:
                metrics, rows = reports[step]
                log_qa(
                    run,
                    metrics,
                    rows,
                    step,
                    "dev",
                    media=any("prediction" in row for row in rows),
                    history_dir=directory / "dev",
                )
    (output_dir / "republication.json").write_text(
        json.dumps(
            {
                "source": source,
                "training_steps": total_steps,
                "dev_steps": sorted(reports),
                "evaluation_charts": sorted(training_curves(directory / "dev", total_steps)),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--reports", type=Path, help="JSON list of {name, report}")
    inputs.add_argument(
        "--training-run",
        type=Path,
        help="Completed training run; add --output-dir to rebuild into a new run",
    )
    parser.add_argument(
        "--media-step", type=int, help="New media upload step for the completed run"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-tag", action="append", default=[])
    parser.add_argument(
        "--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    args = parser.parse_args()
    if args.training_run:
        if args.output_dir is not None and args.media_step is None:
            rebuild_training_run(args.training_run, args.output_dir, args.swanlab_mode)
        else:
            publish_training_history(
                args.training_run, args.media_step, args.swanlab_mode, args.output_dir
            )
        return
    if args.output_dir is None or not args.swanlab_group:
        parser.error("--reports requires --output-dir and --swanlab-group")
    entries = json.loads(args.reports.read_text())
    if not entries or len({e["name"] for e in entries}) != len(entries):
        raise ValueError("reports require unique run names")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    reports, reference = {}, None
    for entry in entries:
        path = (args.reports.parent / entry["report"]).resolve()
        rows = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
        identity = sorted(
            (
                r["capacity"],
                r["episode_id"],
                r["read_id"],
                r["prefix_end"],
                r["condition"],
                r["question"],
                tuple(r["references"]),
            )
            for r in rows
        )
        if reference is not None and identity != reference:
            raise ValueError("comparison requires the same evaluation texts and reads")
        reference = identity
        reports[entry["name"]] = aggregate_qa(rows)
    media = {}
    for metric in ("nll", "em", "f1", "hit_limit_rate"):
        media[f"evaluation/compare/{metric}"] = bar(
            list(CONDITIONS),
            {
                name: [report[f"overall/{c}/all"].get(metric) for c in CONDITIONS]
                for name, report in reports.items()
            },
        )
    media["tables/compare/overall"] = table(
        [
            {"run": name, "group": key, **values}
            for name, report in reports.items()
            for key, values in report.items()
            if key.startswith(("overall/", "paired/"))
        ]
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "reports.json").write_text(json.dumps(entries, indent=2) + "\n")
    (args.output_dir / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
    with swanlab_run(
        args.output_dir,
        {"reports": entries},
        args.swanlab_mode,
        "latent-working-memory-v1",
        job_type="compare" if len(entries) > 1 else "evaluate",
        group=args.swanlab_group,
        tags=tuple(args.swanlab_tag),
        fixed_tags=("scope:main", "method:latent-working-memory", "data:squad"),
    ) as run:
        if run is not None:
            run.log(media, step=0)


if __name__ == "__main__":
    main()
