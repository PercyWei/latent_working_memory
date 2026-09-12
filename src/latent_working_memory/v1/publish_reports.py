from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_working_memory.v1.reporting import comparison_charts, log_test_report
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
        "--publish-individual",
        action="store_true",
        help="Append report charts to each report directory's existing SwanLab run",
    )
    args = parser.parse_args(argv)
    entries = json.loads(args.reports.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("reports must be a non-empty list")
    reports, seen, identities = [], set(), []
    for entry in entries:
        train, test = entry["training_source"], entry["evaluation_source"]
        if not isinstance(train, str) or not train or not isinstance(test, str) or not test:
            raise ValueError("report labels must be non-empty strings")
        if (train, test) in seen:
            raise ValueError("duplicate training/evaluation source pair")
        seen.add((train, test))
        path = (args.reports.parent / entry["report"]).resolve()
        entry["report"] = str(path)
        report = json.loads(path.read_text())
        reports.append((train, test, report))
        if args.publish_individual:
            identity = json.loads((path.parent / "swanlab.json").read_text())
            if (identity["project"], identity["group"], identity["job_type"]) != (
                args.swanlab_project,
                args.swanlab_group,
                "evaluate",
            ):
                raise ValueError("report run must belong to the requested evaluation group")
            if not path.with_suffix(".jsonl").is_file():
                raise FileNotFoundError(path.with_suffix(".jsonl"))
            identities.append(identity)
    # Validate/render comparison before making any cloud writes.
    charts = comparison_charts(reports)
    if args.publish_individual:
        for entry, (_, _, report), identity in zip(entries, reports, identities, strict=True):
            path = Path(entry["report"])
            with swanlab_run(
                path.parent,
                {"report_path": str(path)},
                args.swanlab_mode,
                args.swanlab_project,
                identity["id"],
                job_type="evaluate",
                group=args.swanlab_group,
                tags=tuple(identity["tags"]),
            ) as run:
                log_test_report(run, report, path.with_suffix(".jsonl"))
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
