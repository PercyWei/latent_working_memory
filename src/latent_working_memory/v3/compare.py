"""导出同一评估题池上各方法的质量—容量点，不强制相同容量。"""

import argparse
import csv
import json
from pathlib import Path


def compare(summary_paths, output_dir):
    if not summary_paths:
        raise ValueError("at least one evaluation summary is required")
    points = []
    evaluation_key = None
    for path in summary_paths:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        key = (
            summary["split"],
            summary["dataset_signature"],
            summary["max_new_tokens"],
            summary["objective"]["qa_prompt"],
        )
        if evaluation_key is None:
            evaluation_key = key
        elif key != evaluation_key:
            raise ValueError(
                "comparison requires the same split, evaluation dataset, QA prompt and generation limit"
            )
        objective = summary["objective"]
        point = {
            "summary_path": str(Path(path).resolve()),
            "method": summary["method"],
            "stage": summary["stage"],
            "offline_oracle": summary["offline_oracle"],
            "split": summary["split"],
            "trajectories": summary["trajectories"],
            "questions": summary["quality"]["all"]["questions"],
            **{name: summary["quality"]["all"][name] for name in ("nll", "em", "f1")},
            **{
                f"{group}_{name}": summary["quality"][group][name]
                for group in ("old", "new")
                for name in ("nll", "em", "f1")
            },
            **summary["capacity"],
            **summary["costs"],
            "threshold_i": objective["threshold_i"]
            if summary["method"] == "memory_change"
            else None,
            **{
                name: objective[name] if summary["method"] == "information_loss" else None
                for name in ("threshold_d", "threshold_g", "eta")
            },
        }
        points.append(point)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "points.json").write_text(
        json.dumps(points, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (directory / "points.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    return points


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    points = compare(args.summaries, args.output_dir)
    print(
        f"[compare] runs={len(points)} output={Path(args.output_dir)} (points.json, points.csv)",
        flush=True,
    )


if __name__ == "__main__":
    main()
