"""绘制已完成的逐位置边界诊断；不加载模型或执行 QA。"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import textwrap
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import PercentFormatter
import numpy as np

from ..io import write_json


BIN_EDGES = np.array([index / 10 for index in range(11)])
QUANTILES = {"p10": 0.1, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p90": 0.9, "p95": 0.95}


def summarize_probabilities(probabilities, pending):
    counts, _ = np.histogram(probabilities, bins=BIN_EDGES)
    count = len(probabilities)
    by_pending = []
    for length in np.unique(pending):
        values = probabilities[pending == length]
        by_pending.append({
            "pending_tokens_before": int(length),
            "decision_count": len(values),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.1)),
            "p90": float(np.quantile(values, 0.9)),
            "at_or_above_half_fraction": float((values >= 0.5).mean()),
        })
    return {
        "mean": float(probabilities.mean()) if count else None,
        "minimum": float(probabilities.min()) if count else None,
        "maximum": float(probabilities.max()) if count else None,
        "quantiles": {name: float(np.quantile(probabilities, q)) if count else None
                      for name, q in QUANTILES.items()},
        "at_or_above_half_count": int((probabilities >= 0.5).sum()),
        "at_or_above_half_fraction": float((probabilities >= 0.5).mean()) if count else None,
        "histogram": {
            "edges": [round(float(edge), 1) for edge in BIN_EDGES],
            "counts": counts.tolist(),
            "fractions": (counts / count).tolist() if count else [0.0] * 10,
            "interval_rule": "[left, right), except the final interval [0.9, 1.0]",
        },
        "by_pending_length": by_pending,
    }


def _color(run_id, split):
    hue = int.from_bytes(hashlib.blake2b(run_id.encode(), digest_size=2).digest(), "big") / 65536
    color = hsv_to_rgb((hue, 0.72, 0.68))
    return color if split == "train" else 0.7 * color + 0.3


def _title(plan, split):
    protocol = plan["split_protocols"][split]
    run = textwrap.fill(plan["run_id"], width=87, break_on_hyphens=False)
    return (f"{run}\nCheckpoint {plan['checkpoint_step']}  |  {split} ({protocol['mode']})"
            f"  |  T = {protocol['temperature']:g}")


def _panels(plan, split, statistics):
    panels = [("commit_probability", "Actual probability", statistics["actual_probability"])]
    if plan["split_protocols"][split]["temperature"] != 1.0:
        panels.append(("commit_probability_t1", "Probability at T = 1", statistics["probability_t1"]))
    return panels


def _save_figure(figure, destination):
    for suffix in ("png", "svg"):
        figure.savefig(destination.with_suffix(f".{suffix}"), dpi=180, facecolor="white")
    plt.close(figure)


def plot_histogram(plan, split, statistics, destination):
    panels = _panels(plan, split, statistics)
    figure, axes = plt.subplots(1, len(panels), figsize=(12.6, 5.6), squeeze=False)
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.16, top=0.75, wspace=0.22)
    figure.suptitle(_title(plan, split), x=0.065, y=0.97, ha="left", fontsize=13)
    color = _color(plan["run_id"], split)
    for axis, (_, label, values) in zip(axes[0], panels, strict=True):
        fractions = values["histogram"]["fractions"]
        counts = values["histogram"]["counts"]
        axis.bar(BIN_EDGES[:-1], fractions, width=0.1, align="edge", color=color,
                 edgecolor="white", linewidth=1.0)
        maximum = max(fractions)
        axis.set_ylim(0, max(0.05, maximum * 1.25))
        for index, (count, fraction) in enumerate(zip(counts, fractions, strict=True)):
            axis.annotate(f"{fraction:.1%}\n{count:,}", (index * 0.1 + 0.05, fraction),
                          xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                          fontsize=8 if len(panels) == 2 else 9)
        axis.set_xlim(0, 1)
        axis.set_xticks(BIN_EDGES)
        axis.set_xlabel("Commit probability")
        axis.set_ylabel("Share of eligible decisions")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.axvline(0.5, color="#334155", linestyle="--", linewidth=1.0)
        if statistics["decision_count"]:
            axis.set_title(f"{label}: mean {values['mean']:.3f}, "
                           f"P(p >= 0.5) {values['at_or_above_half_fraction']:.1%}",
                           loc="left", fontsize=10, pad=10)
        else:
            axis.set_title(f"{label}: no eligible decisions", loc="left", fontsize=10, pad=10)
    footnote = (f"N = {statistics['decision_count']:,} decisions; bars show fraction and count. "
                "Bins are [left, right), with 1.0 in the last bin.\n"
                "Greedy uses raw logit >= 0; rounded p can equal 0.5 for a negative logit. "
                "T = 1 keeps the same visited states.")
    figure.text(0.065, 0.035, footnote, fontsize=9, color="#475569", linespacing=1.5)
    _save_figure(figure, destination)


def plot_scatter(plan, split, rows, statistics, destination):
    panels = _panels(plan, split, statistics)
    figure, axes = plt.subplots(1, len(panels), figsize=(12.6, 5.9), squeeze=False, sharey=True)
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.18, top=0.76, wspace=0.12)
    figure.suptitle(_title(plan, split), x=0.065, y=0.97, ha="left", fontsize=13)
    pending = np.array([row["pending_tokens_before"] for row in rows], dtype=np.int64)
    color = _color(plan["run_id"], split)
    for axis, (field, label, values) in zip(axes[0], panels, strict=True):
        probabilities = np.array([row[field] for row in rows], dtype=np.float64)
        axis.scatter(pending, probabilities, s=7, alpha=0.13, linewidths=0, color=color,
                     rasterized=True, label="All eligible decisions")
        by_pending = values["by_pending_length"]
        if by_pending:
            axis.plot([point["pending_tokens_before"] for point in by_pending],
                      [point["mean"] for point in by_pending], color="#172554", linewidth=1.7,
                      label="Mean at each pending length")
        axis.axhline(0.5, color="#64748B", linestyle="--", linewidth=1.0, label="p = 0.5")
        axis.set_xlim(0, max(1, int(pending.max())) * 1.03 if len(pending) else 1)
        axis.set_ylim(-0.025, 1.025)
        axis.set_yticks(np.linspace(0, 1, 5))
        axis.set_xlabel("Pending tokens before decision (excludes arriving unit)")
        axis.set_title(label, loc="left", fontsize=10, pad=10)
        axis.grid(color="#E2E8F0", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.legend(loc="upper right", fontsize=8, framealpha=0.92)
    axes[0, 0].set_ylabel("Commit probability")
    figure.text(0.065, 0.04,
                f"N = {statistics['decision_count']:,} decisions across "
                f"{statistics['documents_with_decisions']:,} documents. All points retained; no jitter.\n"
                "Greedy uses raw logit >= 0; rounded p can equal 0.5 for a negative logit. "
                "T = 1 keeps the same visited states.",
                fontsize=9, color="#475569", linespacing=1.5)
    _save_figure(figure, destination)


def render_diagnostic(input_dir):
    input_dir = Path(input_dir)
    plan = json.loads((input_dir / "plan.json").read_text(encoding="utf-8"))
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "axes.titleweight": "bold",
    })
    result = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S Asia/Shanghai"),
        "run_id": plan["run_id"], "checkpoint_step": plan["checkpoint_step"],
        "aggregation": "Eligible decisions have equal weights; sampled paths are new checkpoint diagnostics.",
        "greedy_rule": "raw_logit >= 0; rounded probabilities >= 0.5 are a separate statistic.",
        "splits": {},
    }
    for split in ("train", "dev"):
        folder = input_dir / split
        with (folder / "decisions.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        pending = np.array([row["pending_tokens_before"] for row in rows], dtype=np.int64)
        actual = np.array([row["commit_probability"] for row in rows], dtype=np.float64)
        reference = np.array([row["commit_probability_t1"] for row in rows], dtype=np.float64)
        statistics = {
            "decision_count": len(rows),
            "source_documents": plan["data"][split]["documents"],
            "documents_with_decisions": len({row["document_id"] for row in rows}),
            "paths_with_decisions": len({(row["document_id"], row["path_id"]) for row in rows}),
            "actual_commit_count": sum(row["action"] for row in rows),
            "actual_commit_fraction": sum(row["action"] for row in rows) / len(rows) if rows else None,
            "nonnegative_logit_count": sum(row["raw_logit"] >= 0 for row in rows),
            "nonnegative_logit_fraction": (sum(row["raw_logit"] >= 0 for row in rows) / len(rows)
                                           if rows else None),
            "protocol": plan["split_protocols"][split],
            "actual_probability": summarize_probabilities(actual, pending),
            "probability_t1": summarize_probabilities(reference, pending),
        }
        result["splits"][split] = statistics
        plot_histogram(plan, split, statistics, folder / "probability_histogram")
        plot_scatter(plan, split, rows, statistics, folder / "pending_probability_scatter")
    write_json(input_dir / "probability_statistics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="包含 plan.json 和 train/dev decisions.jsonl 的单个 checkpoint 诊断目录")
    arguments = parser.parse_args()
    result = render_diagnostic(arguments.input_dir)
    print(json.dumps({"run_id": result["run_id"], "checkpoint_step": result["checkpoint_step"],
                      "decisions": {split: row["decision_count"] for split, row in result["splits"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
