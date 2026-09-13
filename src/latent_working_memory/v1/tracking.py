from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import swanlab

from latent_working_memory.v1.checkpoint import capture_rng_state, restore_rng_state
from latent_working_memory.v1.reporting import (
    evaluation_overview, paired_reconstructions, development_overview,
)


@contextmanager
def swanlab_run(
    output_dir: Path,
    config: dict[str, Any],
    mode: str = "disabled",
    project: str = "latent-working-memory",
    run_id: str | None = None,
    job_type: str = "train",
    group: str | None = None,
    tags: tuple[str, ...] = (),
    fixed_tags: tuple[str, ...] = ("scope:main", "method:latent-working-memory", "data:fineweb"),
) -> Iterator[swanlab.Run | None]:
    if mode == "disabled":
        yield None
        return
    if not group:
        raise ValueError("enabled SwanLab runs require a group")
    fixed_tags = set(fixed_tags)
    if "data_preparation" in config:
        metadata = config["data_preparation"]
        if "source_weights" in metadata:
            fixed_tags.update(f"data:{name}" for name in metadata["source_weights"])
        else:
            fixed_tags.add(f"data:{metadata['boundary_variant']}")
    tags = tuple(sorted(fixed_tags | set(tags)))
    identity_path = output_dir / "swanlab.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        if any(
            identity[k] != v
            for k, v in {
                "project": project,
                "group": group,
                "tags": list(tags),
                "job_type": job_type,
            }.items()
        ) or (run_id is not None and identity["id"] != run_id):
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
            group=group,
            tags=list(tags),
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
                    "group": group,
                    "tags": list(tags),
                    "job_type": job_type,
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
    if "learning_rate" in record:
        metrics["train/learning_rate"] = record["learning_rate"]
        metrics.update(
            {
                f"sampling/length_up_to_{bound}": weight
                for bound, weight in record["length_sampling_weights"].items()
            }
        )
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


def log_evaluation(run, metrics, records_paths, step):
    if run is None:
        return
    values = development_overview(list(records_paths.items()), step)
    values["progress/input_tokens"] = next(iter(metrics.values()))["training_input_tokens"]
    run.log(values, step=step)


def append_evaluation_reports(training_dir: Path, entries: list[dict[str, Any]]) -> None:
    """Append saved reports to a finished online training run without changing its config."""
    identity = json.loads((training_dir / "swanlab.json").read_text())
    if identity["job_type"] != "train" or identity["mode"] != "online":
        raise ValueError("evaluation append requires an online training run")
    reports = [(entry["evaluation_source"], json.loads(Path(entry["report"]).read_text()))
               for entry in entries]
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
    values.update(paired_reconstructions(
        [(entry["evaluation_source"], Path(entry["report"]).with_suffix(".jsonl"))
         for entry in entries], prefix,
    ))
    metadata = {"training_run_id": identity["id"], "split": split,
                "checkpoint_step": step, "reports": entries,
                "protocols": {source: report.get("protocol", {}) for source, report in reports}}
    values[f"{prefix}/metadata"] = swanlab.Text(json.dumps(metadata, ensure_ascii=False))
    with swanlab_training_run(training_dir) as run:
        run.log(values, step=step)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


@contextmanager
def swanlab_training_run(training_dir):
    """Resume a finished training run while preserving its identity and configuration."""
    identity = json.loads((training_dir / "swanlab.json").read_text())
    if identity["job_type"] != "train" or identity["mode"] != "online":
        raise ValueError("evaluation append requires an online training run")
    # The saved URL identifies the workspace as well as the project; IDs alone are not global.
    project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    workspace, project = project_path.split("/")
    if project != identity["project"]:
        raise ValueError("training run URL differs from its project")
    remote = swanlab.Api().run(f"{project_path}/{identity['id']}")
    if remote.state != "FINISHED":
        raise ValueError("append requires a finished training run; do not resume active training")
    # SwanLab's canonical API config is {key: {value, desc, sort}}.
    config = {key: item["value"] for key, item in sorted(
        remote.profile["config"].items(), key=lambda pair: pair[1]["sort"]
    )}
    rng_state = capture_rng_state()
    try:
        run = swanlab.init(
            project=project, workspace=workspace, name=remote.name, config=config,
            id=identity["id"], resume="must", mode="online",
            log_dir=str(training_dir / "swanlab"),
            settings=swanlab.Settings(interactive=False, terminal={"proxy_type": "none"},
                                      probe={"git": False, "monitor": False}),
        )
    finally:
        restore_rng_state(rng_state)
    with run:
        yield run
