from __future__ import annotations

import argparse
import json
import random
import math
from statistics import NormalDist
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from latent_working_memory.v1.data import Episode


def sample_inspection(
    data_dir: Path,
    output_dir: Path,
    split: str = "train",
    examples: int = 200,
    seed: int = 20260909,
) -> dict[str, Any]:
    """Sample completed data into a separate, editable sentence-boundary inspection directory."""
    if split not in {"train", "dev", "test"} or examples <= 0:
        raise ValueError("use a valid split and a positive inspection sample count")
    if output_dir.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError("inspection output must be outside the prepared data directory")
    if output_dir.exists():
        raise FileExistsError("use a new inspection directory")
    metadata = json.loads((data_dir / "preparation.json").read_text())
    if metadata["boundary_variant"] != "semantic":
        raise ValueError("sentence-boundary inspection requires semantic data")
    bounds = metadata["length_bounds"]
    originals = {
        row["record"]["id"]: row["record"]["text"]
        for row in map(json.loads, (data_dir / "documents.jsonl").read_text().splitlines())
    }
    rows, cells = [], defaultdict(list)
    with (data_dir / f"{split}.jsonl").open() as handle:
        for line in handle:
            episode = Episode.from_record(json.loads(line))
            source = episode.sources[0]
            provenance = source.provenance
            size = len(episode.input_ids)
            bucket = next((b for b in bounds if size <= b), size)
            text = originals[source.document_id]
            cuts = {"x_start": provenance["x_char_span"][0], "x_end": provenance["x_char_span"][1]}
            if provenance["y_char_span"] is not None:
                cuts["y_end"] = provenance["y_char_span"][1]
            row = {
                "episode_id": episode.episode_id,
                "document_id": source.document_id,
                "split": split,
                "input_tokens": size,
                "length_up_to": bucket,
                "task": episode.reads[0].task,
                "boundary_variant": provenance["boundary_variant"],
                "input": originals[source.document_id][slice(*provenance["x_char_span"])],
                "continuation": episode.reads[0].references[0].text
                if episode.reads[0].task == "continuation"
                else None,
                "x_char_span": provenance["x_char_span"],
                "y_char_span": provenance["y_char_span"],
                "judgment": None,
                "review_reason": None,
                "reviewer": None,
                "boundaries": {
                    name: {
                        "offset": offset,
                        "left": text[max(0, offset - 400) : offset],
                        "right": text[offset : offset + 400],
                        "at_document_edge": offset in (0, len(text)),
                    }
                    for name, offset in cuts.items()
                },
            }
            rows.append(row)
            cells[(row["task"], bucket)].append(row)
    if not rows:
        raise ValueError("the selected split has no prepared views")
    random_panel = random.Random(seed).sample(rows, min(examples, len(rows)))
    rng = random.Random(seed + 1)
    for cell in cells.values():
        rng.shuffle(cell)
    keys = sorted(cells)
    rng.shuffle(keys)
    stratified_panel, seen_documents = [], set()
    while keys and len(stratified_panel) < examples:
        active = []
        for key in keys:
            cell = cells[key]
            while cell and cell[-1]["document_id"] in seen_documents:
                cell.pop()
            if cell:
                row = cell.pop()
                stratified_panel.append(row)
                seen_documents.add(row["document_id"])
                active.append(key)
            if len(stratified_panel) == examples:
                break
        keys = active
    report = {
        "preparation_id": metadata["preparation_id"],
        "data_directory": str(data_dir.resolve()),
        "split": split,
        "seed": seed,
        "requested_examples_per_panel": examples,
        "population_views": len(rows),
        "random_views": len(random_panel),
        "stratified_views": len(stratified_panel),
        "random_protocol": "uniform views without replacement; estimates complete-sample boundary correctness",
        "stratified_protocol": "rotate task/length cells; distinct documents; diagnostic",
        "judgment_protocol": "pass: all external cuts are sentence boundaries; fail: any cut inside a sentence; uncertain: unresolved. Content quality and context dependence are not defects.",
        "acceptance_threshold": 0.98,
    }
    output_dir.mkdir(parents=True)
    for name, panel in (("random-views", random_panel), ("stratified-views", stratified_panel)):
        with (output_dir / f"{name}.jsonl").open("w") as handle:
            for row in panel:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "inspection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def judgment_statistics(rows: list[dict[str, Any]], threshold: float = 0.98) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        judgment = row["judgment"]
        if judgment not in {None, "pass", "fail", "uncertain"}:
            raise ValueError("inspection judgment must be pass, fail, uncertain, or null")
        if judgment is not None:
            for key in ("reviewer", "review_reason"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError("a reviewed sample needs a reviewer and review_reason")
        counts[judgment or "unreviewed"] += 1
    total = len(rows)
    unresolved = counts["uncertain"] + counts["unreviewed"]
    rate = counts["pass"] / total if total and not unresolved else None
    z = NormalDist().inv_cdf(0.95)
    lower = (
        (
            rate
            + z * z / (2 * total)
            - z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        )
        / (1 + z * z / total)
        if rate is not None
        else None
    )
    return {
        "success_rate": rate,
        "wilson_one_sided_95_lower": lower,
        "above_threshold": rate > threshold if rate is not None else None,
        "lower_bound_above_threshold": lower > threshold if lower is not None else None,
        "samples": total,
        "pass": counts["pass"],
        "fail": counts["fail"],
        "uncertain": counts["uncertain"],
        "unreviewed": counts["unreviewed"],
        "complete": unresolved == 0,
        "failure_rate": counts["fail"] / total if total and not unresolved else None,
        "sample_failure_fraction_bounds": [
            counts["fail"] / total,
            (counts["fail"] + unresolved) / total,
        ]
        if total
        else None,
    }


def summarize_inspection(directory: Path) -> dict[str, Any]:
    metadata = json.loads((directory / "inspection.json").read_text())
    report = {
        "preparation_id": metadata["preparation_id"],
        "split": metadata["split"],
        "population_views": metadata["population_views"],
        "rate_scope": "random panel estimates uniform prepared-sample boundary defect rate; stratified panel is diagnostic",
        "bounds_scope": "failure fraction bounds include unresolved judgments; Wilson lower bound is approximate and does not account for shared-document dependence",
        "acceptance_threshold": metadata["acceptance_threshold"],
        "panels": {},
    }
    for name, count_key in (
        ("random-views", "random_views"),
        ("stratified-views", "stratified_views"),
    ):
        rows = [json.loads(line) for line in (directory / f"{name}.jsonl").read_text().splitlines()]
        if len(rows) != metadata[count_key] or len({r["episode_id"] for r in rows}) != len(rows):
            raise ValueError("inspection panel was truncated or contains duplicate views")
        groups = defaultdict(list)
        for row in rows:
            groups[f"task/{row['task']}"].append(row)
            groups[f"length_up_to/{row['length_up_to']}"].append(row)
        report["panels"][name] = {
            "all": judgment_statistics(rows),
            "groups": {key: judgment_statistics(value) for key, value in sorted(groups.items())},
        }
    (directory / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a completed pretraining dataset separately"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser(
        "sample", help="create independent sentence-boundary inspection panels"
    )
    sample.add_argument("--data-dir", type=Path, required=True)
    sample.add_argument("--output-dir", type=Path, required=True)
    sample.add_argument("--split", choices=("train", "dev", "test"), default="train")
    sample.add_argument("--examples", type=int, default=200)
    sample.add_argument("--seed", type=int, default=20260909)
    summarize = commands.add_parser("summarize", help="summarize judgments entered in the panels")
    summarize.add_argument("--inspection-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "sample":
        report = sample_inspection(
            args.data_dir, args.output_dir, args.split, args.examples, args.seed
        )
    else:
        report = summarize_inspection(args.inspection_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
