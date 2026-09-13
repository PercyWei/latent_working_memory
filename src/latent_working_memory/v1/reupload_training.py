"""Rebuild a deleted cloud training run through the normal training/report loggers."""

import argparse
import json
import shutil
from pathlib import Path

import swanlab

from latent_working_memory.v1.tracking import (
    append_evaluation_reports, log_evaluation, log_training,
    pretraining_tracking_config, swanlab_run,
)


def prepare_replay(training_dir):
    provenance = json.loads((training_dir / "provenance.json").read_text())
    config = pretraining_tracking_config(
        json.loads((training_dir / "config.json").read_text()),
        provenance["run_identity"], provenance["preparation"],
    )
    records = []
    logs = sorted(training_dir.glob("train-from-*.jsonl"))
    for path in logs:
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    if not records:
        raise ValueError("no saved training records")
    resources = json.loads((training_dir / logs[-1].name.replace("train-from-", "resources-from-")
                            .replace(".jsonl", ".json")).read_text())
    steps = [record["step"] for record in records]
    if steps != list(range(steps[0], resources["completed_steps"] + 1)):
        raise ValueError("training records must be complete, ordered and non-overlapping")
    initial_input = resources["cumulative_input_tokens"] - sum(r["input_tokens"] for r in records)
    initial_target = resources["cumulative_target_tokens"] - sum(r["target_tokens"] for r in records)
    if min(initial_input, initial_target) < 0:
        raise ValueError("resource totals differ from training records")
    counters = {steps[0] - 1: initial_input}
    total = initial_input
    for record in records:
        total += record["input_tokens"]
        counters[record["step"]] = total
    dev = {}
    source_steps = []
    for name in provenance["run_identity"]["evaluation_preparations"]:
        directory = training_dir if name == "dev" else training_dir / name
        paths = sorted(directory.glob("dev-step-*.json"))
        source_steps.append([int(path.stem.removeprefix("dev-step-")) for path in paths])
        for path in paths:
            metrics = json.loads(path.read_text())
            step = int(path.stem.removeprefix("dev-step-"))
            if metrics["training_input_tokens"] != counters[step]:
                raise ValueError("dev input-token counter differs from training history")
            dev.setdefault(step, {})[name] = (metrics, path.with_suffix(".jsonl"))
    if not source_steps or any(steps != source_steps[0] for steps in source_steps):
        raise ValueError("dev sources must cover identical evaluation steps")
    return {"config": config, "records": records, "dev": dev,
            "initial_input_tokens": initial_input, "initial_target_tokens": initial_target,
            "final_step": resources["completed_steps"]}


def replay_training(run, prepared):
    inputs, targets = prepared["initial_input_tokens"], prepared["initial_target_tokens"]

    def dev(step):
        reports = prepared["dev"][step]
        log_evaluation(run, {name: value[0] for name, value in reports.items()},
                       {name: value[1] for name, value in reports.items()}, step)

    first = prepared["records"][0]["step"] - 1
    if first in prepared["dev"]:
        dev(first)
    for record in prepared["records"]:
        inputs += record["input_tokens"]
        targets += record["target_tokens"]
        log_training(run, record, inputs, targets)
        if record["step"] in prepared["dev"]:
            dev(record["step"])
        if record["step"] % 1000 == 0:
            print(f"replayed step {record['step']}", flush=True)


def reupload_training(training_dir):
    prepared = prepare_replay(training_dir)
    state_path = training_dir / "reupload.json"
    identity_path = training_dir / "swanlab.json"
    identity = json.loads(identity_path.read_text())
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    continuing = state is not None and state["status"] != "complete"
    if continuing:
        if state["new_id"] != identity["id"]:
            raise ValueError("incomplete reupload identity differs from training directory")
        archive = Path(state["archive"])
    else:
        project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
        response = swanlab.Api()._get(f"/project/{project_path}/runs/{identity['id']}")
        if response.ok or "Disabled_Resource" not in response.errmsg:
            raise ValueError("only an explicitly deleted cloud run can be replaced")
        archive = training_dir / "swanlab-archive" / identity["id"]
        archive.mkdir(parents=True, exist_ok=False)
        shutil.copy2(identity_path, archive / "swanlab.json")
        if state_path.exists():
            shutil.copy2(state_path, archive / "reupload.json")
        state = {"old_id": identity["id"], "new_id": None,
                 "archive": str(archive.resolve()), "status": "uploading"}
    with swanlab_run(
        training_dir, prepared["config"], "online", identity["project"],
        job_type="train", group=identity["group"], tags=tuple(identity["tags"]),
        new_run=not continuing,
    ) as run:
        state["new_id"] = run.id
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        replay_training(run, prepared)
    publications = training_dir / "evaluation-publications"
    if not (archive / "evaluation-publications").exists():
        publications.rename(archive / "evaluation-publications")
    receipt = json.loads((archive / "evaluation-publications" /
                          f"test-step-{prepared['final_step']:06d}.json").read_text())
    # The ordinary report publisher creates the same compact test panels and new identity receipt.
    if not (publications / f"test-step-{prepared['final_step']:06d}.json").exists():
        append_evaluation_reports(training_dir, receipt["reports"])
    state.update(status="complete", train_steps=len(prepared["records"]),
                 dev_steps=sorted(prepared["dev"]), final_step=prepared["final_step"])
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    reupload_training(parser.parse_args().training_run)


if __name__ == "__main__":
    main()
