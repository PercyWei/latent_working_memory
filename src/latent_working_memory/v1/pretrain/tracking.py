"""预训练运行的来源标签、AE/LM 指标、面板与结果追加。"""

from __future__ import annotations
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import swanlab
from latent_working_memory.v1.tracking import (
    DEFAULT_SWANLAB_PROJECT,
    swanlab_run,
    swanlab_training_run,
)
from latent_working_memory.v1.pretrain.reporting import (
    configure_development_panels,
    development_scalars,
    evaluation_overview,
    paired_reconstructions,
    evaluation_tables,
)


@contextmanager
def pretraining_run(
    output_dir,
    config,
    mode="disabled",
    project=DEFAULT_SWANLAB_PROJECT,
    job_type="train",
    group=None,
    tags=(),
    new_run=False,
):
    fixed_tags = {"scope:main", "method:latent-working-memory", "data:fineweb"}
    if mode != "disabled" and "data_preparation" in config:
        metadata = config["data_preparation"]
        if "sources" in metadata:
            fixed_tags.update(f"data:{name}" for name in metadata["sources"])
        elif "source_weights" in metadata:
            fixed_tags.update(f"data:{name}" for name in metadata["source_weights"])
        else:
            fixed_tags.add(f"data:{metadata['boundary_variant']}")
    with swanlab_run(
        output_dir,
        config,
        mode,
        project,
        job_type=job_type,
        group=group,
        tags=tags,
        fixed_tags=tuple(fixed_tags),
        new_run=new_run,
    ) as run:
        if run is not None and job_type == "train" and "evaluation_preparations" in config:
            configure_development_panels(run, list(config["evaluation_preparations"]), mode)
        yield run


def pretraining_tracking_config(config, run_identity, preparation):
    return (
        config
        | run_identity
        | {
            "data_preparation": preparation,
            "global_batch_size": config["batch_size"]
            * config["gradient_accumulation_steps"]
            * run_identity["world_size"],
        }
    )


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
    if "learning_rate" in record:
        metrics["train/learning_rate"] = record["learning_rate"]
    if "epoch" in record:
        for key in (
            "epoch",
            "epoch_progress",
            "epoch_samples",
            "completed_epochs",
            "distinct_samples",
            "sample_visits",
        ):
            metrics[f"progress/{key}"] = record[key]
        metrics["resources/capacity_reads"] = record["capacity_reads"]
    total_weight = sum(sample.get("loss_weight", 1) for sample in samples)
    for task in ("ae", "lm"):
        field = f"{task}_nll"
        selected = [sample for sample in samples if sample[field] is not None]
        weight = sum(sample.get("loss_weight", 1) for sample in selected)
        metrics[f"batch/{task}_fraction"] = weight / total_weight
        if selected:
            metrics[f"train/{task}_nll"] = sum(
                sample[field] * sample.get("loss_weight", 1) for sample in selected
            ) / weight
    for field in ("input_tokens", "continuation_tokens", "capacity", "effective_ratio"):
        values = [s[field] for s in samples]
        metrics.update(
            {
                f"batch/{field}_mean": sum(s[field] * s.get("loss_weight", 1) for s in samples)
                / total_weight,
                f"batch/{field}_min": min(values),
                f"batch/{field}_max": max(values),
            }
        )
    for upper in record["input_length_bounds"]:
        selected = [s for s in samples if s["length_bucket"] == upper]
        metrics[f"batch_by_length/{upper}/samples"] = round(
            sum(s.get("loss_weight", 1) for s in selected)
        )
        metrics[f"batch_by_length/{upper}/input_tokens"] = sum(s["input_tokens"] for s in selected)
        metrics[f"batch_by_length/{upper}/target_tokens"] = sum(
            (s["input_tokens"] + 1 if s["ae_nll"] is not None else 0)
            + (s["continuation_tokens"] + 1 if s["lm_nll"] is not None else 0)
            for s in selected
        )
    run.log(metrics, step=record["step"])


def log_evaluation(run, metrics, records_paths, step):
    if run is None:
        return
    values = development_scalars(metrics)
    values.update(evaluation_tables(list(metrics.items()), "dev/overview"))
    values.update(paired_reconstructions(list(records_paths.items()), "dev/overview"))
    values["progress/input_tokens"] = next(iter(metrics.values()))["training_input_tokens"]
    run.log(values, step=step)


def append_evaluation_reports(training_dir: Path, entries: list[dict[str, Any]]) -> None:
    """Append saved reports to a finished online training run without changing its config."""
    identity = json.loads((training_dir / "swanlab.json").read_text())
    if identity["job_type"] != "train" or identity["mode"] != "online":
        raise ValueError("evaluation append requires an online training run")
    reports = [
        (entry["evaluation_source"], json.loads(Path(entry["report"]).read_text()))
        for entry in entries
    ]
    coordinates = {(report["split"], report["step"]) for _, report in reports}
    if len(coordinates) != 1 or len({source for source, _ in reports}) != len(reports):
        raise ValueError("append requires one split/checkpoint and distinct evaluation sources")
    split, step = coordinates.pop()
    if split not in {"dev", "test"} or not isinstance(step, int) or step < 0:
        raise ValueError("invalid evaluation split or checkpoint step")
    receipt = training_dir / "evaluation-publications" / f"{split}-step-{step:06d}.json"
    if receipt.exists():
        raise ValueError(f"evaluation already appended: {receipt}")
    prefix = f"evaluation/{split}/overview"
    values = evaluation_overview(reports, prefix)
    values.update(
        paired_reconstructions(
            [
                (entry["evaluation_source"], Path(entry["report"]).with_suffix(".jsonl"))
                for entry in entries
            ],
            prefix,
        )
    )
    metadata = {
        "training_run_id": identity["id"],
        "split": split,
        "checkpoint_step": step,
        "reports": entries,
        "protocols": {source: report.get("protocol", {}) for source, report in reports},
    }
    values[f"{prefix}/metadata"] = swanlab.Text(json.dumps(metadata, ensure_ascii=False))
    with swanlab_training_run(training_dir) as run:
        run.log(values, step=step)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
