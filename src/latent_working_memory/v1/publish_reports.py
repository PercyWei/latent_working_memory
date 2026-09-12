from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_working_memory.v1.evaluation import aggregate_pretrain_metrics
from latent_working_memory.v1.reporting import (
    comparison_charts,
    build_evaluation_charts,
    reconstruction_media,
)
from latent_working_memory.v1.tracking import swanlab_run


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Publish saved evaluation reports without inference"
    )
    parser.add_argument(
        "--reports",
        required=True,
        type=Path,
        help="JSON list: training_source, evaluation_source, report (JSON path)",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--swanlab-project", required=True)
    parser.add_argument("--swanlab-group", required=True)
    parser.add_argument("--swanlab-tag", action="append", default=[])
    parser.add_argument("--swanlab-mode", choices=("offline", "online"), default="offline")
    parser.add_argument(
        "--evaluation-output",
        nargs=2,
        action="append",
        default=[],
        metavar=("TRAINING_SOURCE", "DIRECTORY"),
        help="Publish all test sources for one training source into one evaluation run",
    )
    args = parser.parse_args(argv)
    entries = json.loads(args.reports.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("reports must be a non-empty list")
    reports, seen = [], set()
    for entry in entries:
        train, test = entry["training_source"], entry["evaluation_source"]
        if not isinstance(train, str) or not train or not isinstance(test, str) or not test:
            raise ValueError("report labels must be non-empty strings")
        if (train, test) in seen:
            raise ValueError("duplicate training/evaluation source pair")
        seen.add((train, test))
        path = (args.reports.parent / entry["report"]).resolve()
        entry["report"] = str(path)
        # Re-aggregate saved reads for the current evaluation protocol.
        records = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
        records = [
            record
            for record in records
            if record["condition"] != "recent_context"
            and not (record["task"] == "ae" and record["condition"] == "no_memory")
        ]
        report = aggregate_pretrain_metrics(records)
        reports.append((train, test, report))
    charts = comparison_charts(reports)
    outputs = dict(args.evaluation_output)
    if len(outputs) != len(args.evaluation_output):
        raise ValueError("duplicate evaluation output source")
    if (
        len({Path(p).resolve() for p in outputs.values()} | {args.output_dir.resolve()})
        != len(outputs) + 1
    ):
        raise ValueError("each run needs its own output directory")
    bundles = []
    for train, directory in outputs.items():
        selected = [(test, report) for source, test, report in reports if source == train]
        if not selected:
            raise ValueError(f"unknown training source: {train}")
        selected_entries = [entry for entry in entries if entry["training_source"] == train]
        rendered = build_evaluation_charts(selected)
        for entry in selected_entries:
            rendered.update(
                reconstruction_media(
                    Path(entry["report"]).with_suffix(".jsonl"),
                    f"examples/{entry['evaluation_source']}",
                )
            )
        bundles.append((Path(directory), selected_entries, rendered))
    # Render every report before cloud writes; source reports keep full precision.
    for directory, selected_entries, rendered in bundles:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "reports.json").write_text(json.dumps(selected_entries, indent=2) + "\n")
        with swanlab_run(
            directory,
            {"reports": selected_entries},
            args.swanlab_mode,
            args.swanlab_project,
            job_type="evaluate",
            group=args.swanlab_group,
            tags=tuple(args.swanlab_tag),
        ) as run:
            run.log(rendered)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reports.json").write_text(json.dumps(entries, indent=2) + "\n")
    with swanlab_run(
        args.output_dir,
        {"reports": entries},
        args.swanlab_mode,
        args.swanlab_project,
        job_type="compare",
        group=args.swanlab_group,
        tags=tuple(args.swanlab_tag),
    ) as run:
        run.log(charts)


if __name__ == "__main__":
    main()
