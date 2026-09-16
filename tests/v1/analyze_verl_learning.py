# /// script
# dependencies = ["matplotlib==3.10.0"]
# ///
"""Summarize completed paired learning runs and render their measured curves."""

import argparse
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_mean(values, window):
    return [statistics.mean(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    loaded = []
    for name in args.runs:
        directory = args.root / name
        settings = json.loads((directory / "settings.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        audit = json.loads((directory / "checkpoint-audit.json").read_text())
        records = [
            json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()
        ]
        assert [r["global_step"] for r in records] == list(range(1, settings["steps"] + 1))
        assert result["completed_steps"] == settings["steps"]
        assert all(math.isfinite(r[k]) for r in records for k in ("loss", "gradient_norm"))
        label = (
            "DDP"
            if settings["backend"] == "ddp"
            else (
                "FSDP2 reshard" if settings["reshard_after_forward"] else "FSDP2 no forward reshard"
            )
        )
        label += f" (seed {settings['model']['model_seed']})"
        loaded.append(
            {
                "name": name,
                "label": label,
                "settings": settings,
                "result": result,
                "records": records,
                "checkpoint_audit": audit,
            }
        )
    comparison = []
    for run in loaded:
        settings, result = run["settings"], run["result"]
        reference = next(
            x
            for x in loaded
            if x["settings"]["backend"] == "ddp"
            and x["settings"]["model"]["model_seed"] == settings["model"]["model_seed"]
        )
        baseline = reference["result"]
        actual_groups = result["evaluations"][-1]["groups"]
        reference_groups = baseline["evaluations"][-1]["groups"]
        deltas = {
            key: value["nll"] - reference_groups[key]["nll"] for key, value in actual_groups.items()
        }
        gain = 1 - result["mean_step_seconds"] / baseline["mean_step_seconds"]
        row = {
            "run": run["name"],
            "label": run["label"],
            "mean_step_seconds": result["mean_step_seconds"],
            "samples_per_second": result["samples_per_second"],
            "target_tokens_per_second": result["target_tokens_per_second"],
            "step_time_reduction": gain,
            "peak_allocated_gib": result["peak_allocated_bytes"] / 2**30,
            "peak_reserved_gib": result["peak_reserved_bytes"] / 2**30,
            "mean_warm_dev_seconds": statistics.mean(
                e["seconds"] for e in result["evaluations"][1:]
            ),
            "final_dev": {key: actual_groups[key]["nll"] for key in ("all", "ae", "continuation")},
            "dev_delta_vs_ddp": deltas,
            "quality_screen_passed": deltas["all"] <= 0.05
            and deltas["ae"] <= 0.10
            and deltas["continuation"] <= 0.10,
            "speed_screen_passed": None if settings["backend"] == "ddp" else gain >= 0.05,
            "checkpoint_complete": run["checkpoint_audit"]["checkpoint_complete"],
            "gradient_norm_max": max(r["gradient_norm"] for r in run["records"]),
            "gradient_norm_median": statistics.median(r["gradient_norm"] for r in run["records"]),
            "total_input_tokens": sum(r["input_tokens"] for r in run["records"]),
            "total_target_tokens": sum(r["target_tokens"] for r in run["records"]),
        }
        comparison.append(row)
    (args.root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    seeds = sorted({r["settings"]["model"]["model_seed"] for r in loaded})
    for run in loaded:
        rows, evaluations = run["records"], run["result"]["evaluations"]
        settings = run["settings"]
        variant = (
            0 if settings["backend"] == "ddp" else (1 if settings["reshard_after_forward"] else 2)
        )
        seed_index = seeds.index(settings["model"]["model_seed"])
        style = {"color": f"C{variant}", "linestyle": ("-", "--", ":", "-.")[seed_index % 4]}
        marker = "o" if seed_index == 0 else "s"
        steps = [r["global_step"] for r in rows]
        dev_steps = [e["global_step"] for e in evaluations]
        axes[0, 0].plot(
            steps, moving_mean([r["loss"] for r in rows], 20), label=run["label"], **style
        )
        axes[0, 1].plot(
            dev_steps, [e["groups"]["all"]["nll"] for e in evaluations], marker=marker, **style
        )
        axes[0, 2].plot(steps, moving_mean([r["seconds"] for r in rows], 10), **style)
        axes[1, 0].plot(
            dev_steps, [e["groups"]["ae"]["nll"] for e in evaluations], marker=marker, **style
        )
        axes[1, 1].plot(
            dev_steps,
            [e["groups"]["continuation"]["nll"] for e in evaluations],
            marker=marker,
            **style,
        )
        axes[1, 2].plot(steps, [r["peak_allocated_bytes"] / 2**30 for r in rows], **style)
    titles = [
        "Train NLL (20-step mean)",
        "Fixed dev NLL",
        "Step time (10-step mean)",
        "Dev AE NLL",
        "Dev continuation NLL",
        "Peak allocated GPU memory",
    ]
    units = ["NLL", "NLL", "seconds", "NLL", "NLL", "GiB / GPU"]
    for ax, title, unit in zip(axes.flat, titles, units, strict=True):
        ax.set(title=title, xlabel="Global step", ylabel=unit)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Qwen2.5-3B: paired real-data pretraining validation", fontsize=15)
    fig.savefig(args.root / "learning-curves.png", dpi=160)
    fig.savefig(args.root / "learning-curves.pdf")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "label",
                        "mean_step_seconds",
                        "final_dev",
                        "quality_screen_passed",
                        "speed_screen_passed",
                        "checkpoint_complete",
                    )
                }
                for r in comparison
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
