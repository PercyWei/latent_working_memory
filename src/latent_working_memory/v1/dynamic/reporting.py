"""Publish dynamic training, incremental dev curves and final QA evaluation."""

import argparse
import json
from importlib.metadata import version
import subprocess
from pathlib import Path

import swanlab

from latent_working_memory.v1.reporting import configure_line_panels
from latent_working_memory.v1.dynamic.evaluation import aggregate_qa
from latent_working_memory.v1.reporting import CONDITION_COLORS, _bar, _shade
from latent_working_memory.v1.tracking import swanlab_run, swanlab_training_run


COLORS = {
    "memory": CONDITION_COLORS["memory"],
    "wrong_memory": CONDITION_COLORS["wrong_memory"],
    "no_memory": CONDITION_COLORS["no_memory"],
    "gold_paragraph": CONDITION_COLORS["full_context"],
    "gold_paragraph_base": CONDITION_COLORS["base_full_context"],
}
CONDITIONS = tuple(COLORS)
COLORS.update(no_memory_base="#BBC1C6", no_memory_pretrain="#9099A1")
DISPLAY_CONDITIONS = ("memory", "wrong_memory", "no_memory_base", "no_memory_pretrain",
                      "no_memory", "gold_paragraph", "gold_paragraph_base")
CORE_METRICS = ("nll", "em", "f1")


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
    return _bar(labels, series, labels, COLORS, "condition")


def qa_media(reports, prefix="evaluation/test/overview", charts=True):
    """Group final conditions by evaluation dataset; keep tables and examples paired."""
    values = {}
    datasets = list(reports)
    if charts:
        for metric in CORE_METRICS:
            conditions = [c for c in DISPLAY_CONDITIONS
                          if any(metric in metrics.get(f"overall/{c}/all", {})
                                 for metrics, _ in reports.values())]
            if not conditions:
                continue
            chart = swanlab.echarts.Bar().add_xaxis(datasets)
            for condition in conditions:
                points = []
                for index, (metrics, _) in enumerate(reports.values()):
                    value = metrics.get(f"overall/{condition}/all", {}).get(metric)
                    points.append(None if value is None else {"value": round(value, 4),
                        "itemStyle": {"color": _shade(COLORS[condition], index, len(datasets))}})
                chart.add_yaxis(condition, points, label_opts={"show": False},
                                itemstyle_opts={"color": COLORS[condition]})
            chart.set_global_opts(tooltip_opts={"trigger": "axis"},
                                  legend_opts={"type": "scroll", "top": 0},
                                  xaxis_opts={"axisLabel": {"interval": 0}},
                                  yaxis_opts={"minInterval": .001})
            chart.options["grid"] = {"left": "12%", "right": "4%", "top": "20%",
                                     "bottom": "12%", "containLabel": True}
            values[f"{prefix}/{metric}"] = chart
    for name, overall in (("summary", True), ("details", False)):
        records = [{"dataset": dataset, "group": key, **metrics[key]}
                   for dataset, (metrics, _) in reports.items()
                   for key in metrics if key.startswith("overall/") == overall]
        if records:
            values[f"{prefix}/{name}"] = table(records)
    examples = []
    for dataset, (_, rows) in reports.items():
        selected = {}
        for row in rows:
            if "prediction" not in row:
                continue
            key = row["capacity"], row["episode_id"], row["read_id"], row["prefix_end"]
            if key not in selected and len(selected) >= 10:
                continue
            selected.setdefault(key, {})[row["condition"]] = row
        for key, sample in selected.items():
            reference = next(iter(sample.values()))
            examples.append(swanlab.Text(
                reference["question"] + "\n\nReferences: "
                + json.dumps(reference["references"], ensure_ascii=False) + "\n\n"
                + "\n".join(f"{c}: {sample[c]['prediction']}"
                              for c in DISPLAY_CONDITIONS if c in sample),
                caption=f"dataset={dataset}, K={key[0]}, {reference['kind']}, "
                        f"delay={reference['delay_tokens']} tokens"))
    if examples:
        values[f"{prefix}/examples"] = examples
    return values


def dev_scalars(metrics, dataset):
    return {
        f"dev/{dataset}/{metric}/{condition}": metrics[f"overall/{condition}/all"][metric]
        for metric in CORE_METRICS
        for condition in CONDITIONS
        if metric in metrics.get(f"overall/{condition}/all", {})
    }


def dev_panels(datasets):
    return {
        f"dev/{dataset}/{metric}": {
            "title": f"dev/{dataset}/{metric}",
            "config": {
                "xAxis": {"key": "step", "name": "step", "type": "FLOAT", "class": "SYSTEM"},
                "yAxis": [
                    {
                        "key": f"dev/{dataset}/{metric}/{condition}",
                        "name": f"dev/{dataset}/{metric}/{condition}",
                        "type": "FLOAT",
                        "class": "CUSTOM",
                    }
                    for condition in CONDITIONS
                ],
                "xName": "optimizer step",
                "yName": metric,
            },
        }
        for dataset in datasets
        for metric in CORE_METRICS
    }


def dev_panel_style(panel, run_id):
    return {
        f"{run_id}-{axis['key']}": {
            "name": axis["key"].split("/")[-1],
            "colors": [COLORS[axis["key"].split("/")[-1]]] * 2,
        }
        for axis in panel["config"]["yAxis"]
    }


def configure_dynamic_panels(run, mode, datasets):
    if run is not None:
        configure_line_panels(run, dev_panels(datasets), mode, dev_panel_style)


def training_metrics(record):
    names = {
        "loss": "train/loss",
        "target_nll": "train/target_nll",
        "gradient_norm": "train/gradient_norm",
        "learning_rate": "train/learning_rate",
        "seconds": "resources/step_seconds",
        "input_tokens_per_second": "resources/input_tokens_per_second",
        "capacity": "sampling/capacity",
    }
    values = {name: record[key] for key, name in names.items() if key in record}
    if "peak_memory_bytes" in record:
        values["resources/peak_memory_gib"] = record["peak_memory_bytes"] / 1024**3
    values.update({f"progress/{key}": value for key, value in record["cumulative"].items()})
    return values


def log_qa(run, metrics, rows, step, split, dataset, media=False, final=False):
    if run is None:
        return
    if final:
        values = qa_media({dataset: (metrics, rows)}, f"evaluation/{split}/overview")
    else:
        if split != "dev":
            raise ValueError("incremental validation requires the dev split")
        values = dev_scalars(metrics, dataset)
        if media:
            values.update(qa_media({dataset: (metrics, rows)}, f"dev/{dataset}", charts=False))
    run.log(values, step=step)


def read_report(path):
    return (
        json.loads(path.read_text()),
        [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()],
    )


def validate_final_report(directory, report):
    provenance = json.loads((directory / "provenance.json").read_text())
    info = json.loads((report.parent / "evaluation.json").read_text())
    step = provenance["target_steps"]
    expected = directory / "checkpoints" / f"dynamic-step-{step:06d}.pt"
    if (
        info["split"] != "test"
        or info["checkpoint_step"] != step
        or Path(info["checkpoint"]).resolve() != expected.resolve()
        or info["config"] != provenance["config"]
        or report.name != f"test-step-{step:06d}.json"
    ):
        raise ValueError(
            "test report must use this training run's final checkpoint and configuration"
        )
    return step, info


def append_qa_reports(directory, reports, mode):
    """Append final test to its training run, preserving the cloud training config."""
    if mode == "disabled":
        return
    validated = {name: validate_final_report(directory, report) for name, report in reports.items()}
    if not validated or any(json.loads((report.parent / "evaluation.json").read_text())["name"] != name
                            for name, report in reports.items()):
        raise ValueError("test reports must be keyed by their recorded evaluation source names")
    step, _ = next(iter(validated.values()))
    media = qa_media({name: read_report(path) for name, path in reports.items()})
    identity = json.loads((directory / "swanlab.json").read_text())
    if mode == "offline" and identity["mode"] != "offline":
        raise ValueError("offline evaluation requires an offline training run")
    provenance = json.loads((directory / "provenance.json").read_text())
    receipt = directory / "evaluation-publications" / f"test-step-{step:06d}.json"
    if receipt.exists():
        raise ValueError(f"test already published: {receipt}")
    context = (
        swanlab_training_run(directory)
        if mode == "online"
        else swanlab_run(
            directory,
            provenance,
            mode,
            identity["project"],
            job_type="train",
            group=identity["group"],
            tags=tuple(identity["tags"]),
            fixed_tags=(),
        )
    )
    with context as run:
        run.log(media, step=step)
    receipt.parent.mkdir(exist_ok=True)
    receipt.write_text(
        json.dumps(
            {"training_run_id": identity["id"], "checkpoint_step": step,
             "reports": {name: {"report": str(path.resolve()), **validated[name][1]}
                         for name, path in reports.items()}}, indent=2
        )
        + "\n"
    )


def rebuild_training_run(directory, output_dir, mode, test_reports, previous_run_dir=None):
    """Replay saved observations in step order, then publish the final checkpoint's test."""
    provenance = json.loads((directory / "provenance.json").read_text())
    identity = json.loads((directory / "swanlab.json").read_text())
    previous = (
        identity
        if previous_run_dir is None
        else json.loads((previous_run_dir / "swanlab.json").read_text())
    )
    tests = {name: validate_final_report(directory, path) for name, path in test_reports.items()}
    if not tests or any(info["name"] != name for name, (_, info) in tests.items()):
        raise ValueError("test reports must use recorded evaluation source names")
    total_steps = next(iter(tests.values()))[0]
    completed = [
        json.loads(p.read_text())["completed_steps"]
        for p in directory.glob("resources-from-*.json")
    ]
    if total_steps not in completed:
        raise ValueError("rebuilding requires a completed training run")
    records = sorted(
        (
            json.loads(line)
            for path in directory.glob("train-from-*.jsonl")
            for line in path.read_text().splitlines()
        ),
        key=lambda r: r["step"],
    )
    if [r["step"] for r in records] != list(range(1, total_steps + 1)):
        raise ValueError("rebuilding requires exactly one training record per step")
    datasets = provenance["selection"]["evaluation"]["dev"]
    reports = {name: {
        int(path.stem.removeprefix("dev-step-")): read_report(path)
        for path in sorted((directory / "dev" / name).glob("dev-step-*.json"))
    } for name in datasets}
    if any(0 not in values or total_steps not in values or max(values) > total_steps
           for values in reports.values()):
        raise ValueError("rebuilding requires initial and final dev reports within training steps")
    media = qa_media({name: read_report(path) for name, path in test_reports.items()})
    rendered = {
        key: json.loads(chart.dump_options())
        for key, chart in media.items()
        if key.rsplit("/", 1)[-1] in CORE_METRICS
    }
    scalar_records = []
    for step in range(total_steps + 1):
        values = {} if step == 0 else training_metrics(records[step - 1])
        for name in datasets:
            if step in reports[name]:
                values.update(dev_scalars(reports[name][step][0], name))
        scalar_records.append({"step": step, "values": values})
    output_dir.mkdir(parents=True, exist_ok=False)
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
        configure_dynamic_panels(run, mode, datasets)
        for step in range(total_steps + 1):
            if step and run is not None:
                run.log(training_metrics(records[step - 1]), step=step)
            for name in datasets:
                if step in reports[name]:
                    metrics, rows = reports[name][step]
                    log_qa(run, metrics, rows, step, "dev", name,
                           media=any("prediction" in row for row in rows))
        if run is not None:
            run.log(media, step=total_steps)
    (output_dir / "scalar-records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in scalar_records)
    )
    (output_dir / "evaluation-charts.json").write_text(json.dumps(rendered, indent=2) + "\n")
    (output_dir / "dev-panels.json").write_text(
        json.dumps(
            {
                key: {**panel, "styles": dev_panel_style(panel, "RUN")}
                for key, panel in dev_panels(datasets).items()
            },
            indent=2,
        )
        + "\n"
    )
    manifest = {
        "source": {"training_run": str(directory.resolve()), "swanlab_id": identity["id"]},
        "previous_run": previous,
        "new_run": json.loads((output_dir / "swanlab.json").read_text())
        if mode != "disabled"
        else None,
        "training_steps": total_steps,
        "dev_steps": {name: sorted(values) for name, values in reports.items()},
        "test_reports": {name: str(path.resolve()) for name, path in test_reports.items()},
        "test_checkpoint": next(iter(tests.values()))[1]["checkpoint"],
        "test_step": total_steps,
        "evaluation_charts": sorted(rendered),
        "dev_panels": list(dev_panels(datasets)),
        "original_training_seconds": sum(r["seconds"] for r in records),
        "publication_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip(),
        "publication_packages": {name: version(name) for name in ("swanlab", "pyecharts")},
    }
    (output_dir / "republication.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--reports", type=Path, help="JSON list of {name, dataset, report}")
    inputs.add_argument("--training-run", type=Path, help="Completed source training directory")
    parser.add_argument("--test-reports", type=Path, help="JSON map of evaluation source to test report")
    parser.add_argument(
        "--previous-run-dir", type=Path, help="Previous display run identity directory"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-tag", action="append", default=[])
    parser.add_argument(
        "--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    args = parser.parse_args()
    if args.training_run:
        if args.test_reports is None:
            parser.error("--training-run requires --test-reports")
        rebuild_training_run(
            args.training_run,
            args.output_dir,
            args.swanlab_mode,
            {name: (args.test_reports.parent / path).resolve()
             for name, path in json.loads(args.test_reports.read_text()).items()},
            args.previous_run_dir,
        )
        return
    if not args.swanlab_group:
        parser.error("--reports requires --swanlab-group")
    entries = json.loads(args.reports.read_text())
    if not entries or len({(e["name"], e["dataset"]) for e in entries}) != len(entries):
        raise ValueError("reports require unique run and evaluation-source pairs")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    reports, references = {}, {}
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
        dataset = entry["dataset"]
        if dataset in references and identity != references[dataset]:
            raise ValueError("comparison requires the same evaluation texts and reads")
        references[dataset] = identity
        reports.setdefault(dataset, {})[entry["name"]] = aggregate_qa(rows)
    media = {}
    for dataset, results in reports.items():
        for metric in ("nll", "em", "f1", "hit_limit_rate"):
            conditions = [c for c in DISPLAY_CONDITIONS
                          if any(f"overall/{c}/all" in result for result in results.values())]
            media[f"evaluation/compare/{dataset}/{metric}"] = bar(
                conditions,
                {name: [report.get(f"overall/{c}/all", {}).get(metric) for c in conditions]
                 for name, report in results.items()},
            )
    media["tables/compare/overall"] = table(
        [{"dataset": dataset, "run": name, "group": key, **values}
         for dataset, results in reports.items()
         for name, report in results.items()
         for key, values in report.items() if key.startswith(("overall/", "paired/"))]
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
        fixed_tags=("scope:main", "method:latent-working-memory"),
    ) as run:
        if run is not None:
            run.log(media, step=0)


if __name__ == "__main__":
    main()
