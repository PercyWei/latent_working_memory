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


def qa_media(metrics, rows):
    media = {}
    for metric in ("nll", "em", "f1", "hit_limit_rate"):
        series = {
            kind: [metrics.get(f"overall/{c}/{kind}", {}).get(metric) for c in CONDITIONS]
            for kind in ("all", "arrival", "delayed")
        }
        if any(v is not None for values in series.values() for v in values):
            media[f"charts/{metric}"] = bar(list(CONDITIONS), series)
    for axis in ("delay-tokens-le", "delay-updates-le"):
        labels = sorted(
            {key.split("/")[0] for key in metrics if key.startswith(axis)},
            key=lambda s: int(s.removeprefix(axis)),
        )
        for metric in ("nll", "f1"):
            if labels:
                media[f"charts/{axis}/{metric}"] = bar(
                    labels,
                    {
                        c: [metrics.get(f"{label}/{c}/all", {}).get(metric) for label in labels]
                        for c in CONDITIONS
                    },
                )
    media["tables/capacity-and-ratio"] = table(
        [
            {"group": key, **value}
            for key, value in metrics.items()
            if key.startswith("k") and key.endswith("/memory/all")
        ]
    )
    media["tables/paired"] = table(
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
    media["examples/qa"] = [
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


def log_qa(run, metrics, rows, step, prefix, media=False):
    if run is None:
        return
    values = {
        f"evaluation/{prefix}/{group}/{key}": value
        for group, scores in metrics.items()
        if group.startswith(("overall/", "paired/"))
        or (group.startswith("k") and group.count("/") == 2 and "/memory/" in group)
        for key, value in scores.items()
    }
    if media:
        values.update({f"{key}/{prefix}": val for key, val in qa_media(metrics, rows).items()})
    run.log(values, step=step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True, help="JSON list of {name, report}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--swanlab-group", required=True)
    parser.add_argument("--swanlab-tag", action="append", default=[])
    parser.add_argument(
        "--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    args = parser.parse_args()
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
        media[f"charts/{metric}"] = bar(
            list(CONDITIONS),
            {
                name: [report[f"overall/{c}/all"].get(metric) for c in CONDITIONS]
                for name, report in reports.items()
            },
        )
    media["tables/overall"] = table(
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
