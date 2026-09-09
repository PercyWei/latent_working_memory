from __future__ import annotations

import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import swanlab

from latent_working_memory.v1.checkpoint import capture_rng_state, restore_rng_state
from latent_working_memory.v1.config import GRANULARITIES


@contextmanager
def swanlab_run(
    output_dir: Path,
    config: dict[str, Any],
    mode: str = "disabled",
    project: str = "latent-working-memory",
    run_id: str | None = None,
    job_type: str = "pretrain",
) -> Iterator[swanlab.Run | None]:
    if mode == "disabled":
        yield None
        return
    identity_path = output_dir / "swanlab.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        if identity["project"] != project or (run_id is not None and identity["id"] != run_id):
            raise ValueError("SwanLab project/run differs from the output directory")
        run_id = identity["id"]
    rng_state = capture_rng_state()
    try:
        run = swanlab.init(
            project=project,
            name=output_dir.name,
            config=config,
            mode=mode,
            public=False,
            job_type=job_type,
            log_dir=str(output_dir / "swanlab"),
            id=run_id,
            resume="allow" if run_id is not None else "never",
            settings=swanlab.Settings(
                interactive=False,
                terminal={"proxy_type": "none"},
                probe={"git": False, "monitor": False},
            ),
        )
    finally:
        restore_rng_state(rng_state)
    with run:
        identity_path.write_text(
            json.dumps(
                {
                    "id": run.id,
                    "project": project,
                    "mode": mode,
                    "url": run.url if mode == "online" else None,
                },
                indent=2,
            )
            + "\n"
        )
        yield run


def log_training(
    run: swanlab.Run | None,
    record: dict[str, Any],
    cumulative_input_tokens: int,
    cumulative_target_tokens: int,
) -> None:
    if run is None:
        return
    samples = record["samples"]
    metrics = {
        "train/loss": record["loss"],
        "train/gradient_norm": record["gradient_norm"],
        "resources/step_seconds": record["seconds"],
        "resources/input_tokens_per_second": record["input_tokens_per_second"],
        "resources/peak_memory_gib": record["peak_memory_bytes"] / 1024**3,
        "progress/input_tokens": cumulative_input_tokens,
        "progress/target_tokens": cumulative_target_tokens,
        "progress/distinct_documents": record["distinct_documents"],
        "progress/document_visits": record["document_visits"],
    }
    ae_samples = [s for s in samples if s["ae_nll"] is not None]
    metrics["batch/ae_fraction"] = len(ae_samples) / len(samples)
    if ae_samples:
        metrics["train/ae_nll"] = sum(s["ae_nll"] for s in ae_samples) / len(ae_samples)
    lm_samples = [s for s in samples if s["lm_nll"] is not None]
    metrics["batch/lm_fraction"] = len(lm_samples) / len(samples)
    if lm_samples:
        metrics["train/lm_nll"] = sum(s["lm_nll"] for s in lm_samples) / len(lm_samples)
    for field in ("input_tokens", "continuation_tokens", "capacity", "effective_ratio"):
        values = [s[field] for s in samples]
        metrics.update(
            {
                f"batch/{field}_mean": sum(values) / len(values),
                f"batch/{field}_min": min(values),
                f"batch/{field}_max": max(values),
            }
        )
    granularities = Counter(s["granularity"] for s in samples)
    metrics.update(
        {
            f"batch/{name}_fraction": granularities[name] / len(samples)
            for name in (*GRANULARITIES, "random")
        }
    )
    for upper in record["input_length_bounds"]:
        selected = [s for s in samples if s["length_bucket"] == upper]
        metrics[f"batch_by_length/{upper}/samples"] = len(selected)
        metrics[f"batch_by_length/{upper}/input_tokens"] = sum(s["input_tokens"] for s in selected)
        metrics[f"batch_by_length/{upper}/target_tokens"] = sum(
            (s["input_tokens"] + 1 if s["ae_nll"] is not None else 0)
            + (s["continuation_tokens"] + 1 if s["lm_nll"] is not None else 0)
            for s in selected
        )
    run.log(metrics, step=record["step"])


def log_evaluation(
    run: swanlab.Run | None,
    metrics: dict[str, Any],
    records_path: Path,
    step: int,
    split: str = "dev",
) -> None:
    if run is None:
        return
    values: dict[str, Any] = {"progress/input_tokens": metrics["training_input_tokens"]}
    generation_metrics = (
        "generated_reads",
        "sequence_match",
        "normalized_token_edit_distance",
        "correct_prefix_ratio",
        "bleu_4",
    )
    strata = {"granularity", "length_up_to", "ratio_up_to", "capacity"}
    for group, summary in metrics["groups"].items():
        category, name = group.split("/", 1)
        if category == "all":
            for metric in (
                "nll",
                "ppl",
                "token_accuracy",
                "reads",
                *generation_metrics,
            ):
                if metric in summary:
                    values[f"{split}/{name}/{metric}"] = summary[metric]
        elif category in strata:
            for metric in ("nll", "reads", *generation_metrics):
                if metric in summary:
                    values[f"{split}_by_{category}/{name}/{metric}"] = summary[metric]
    for group, comparison in metrics["comparisons"].items():
        category, name = group.split("/", 1)
        if category == "all" or category in strata:
            prefix = split if category == "all" else f"{split}_by_{category}"
            values.update(
                {f"{prefix}/{name}/{metric}": value for metric, value in comparison.items()}
            )
    examples = []
    for line in records_path.read_text().splitlines():
        record = json.loads(line)
        if "prediction" in record:
            examples.append(
                swanlab.Text(
                    f"Reference:\n{record['reference']}\n\nPrediction:\n{record['prediction']}",
                    caption=f"{record['input_tokens']} tokens, K={record['capacity']}, "
                    f"exact={record['sequence_match']}, prefix={record['correct_prefix_ratio']:.3f}",
                )
            )
    # SwanLab caps one media list at 108 items. Keep every generated read visible.
    for start in range(0, len(examples), 100):
        suffix = "" if start == 0 else f"/page_{start // 100 + 1}"
        values[f"{split}/reconstruction{suffix}"] = examples[start : start + 100]
    run.log(values, step=step)
