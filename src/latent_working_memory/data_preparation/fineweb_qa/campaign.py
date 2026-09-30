"""Run frozen FineWeb batches to a train-trajectory target and report construction cost."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from latent_working_memory.data_preparation.fineweb_qa.pipeline import (
    _prompt_texts,
    _request_statistics,
    _without_endpoint,
)
from latent_working_memory.data_preparation.fineweb_qa.storage import load_json, save_json

STAGES = ("prepare", "annotate", "diagnose", "review", "finalize")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def batch_config(template: dict, index: int) -> dict:
    config = copy.deepcopy(template)
    config["batch_index"] = index
    for key in ("artifacts_dir", "dataset_dir"):
        config[key] = str(Path(template[key]).parent / f"batch-{index:03d}")
    return config


def _inputs(config: dict) -> tuple[dict, dict]:
    template = load_json(Path(config["batch_template"]))
    pool = load_json(Path(template["source_pool_dir"]) / "source-pool.json")
    target = config["target_train_trajectories"]
    if type(target) is not int or not 0 < target <= pool["split_counts"]["train"]:
        raise ValueError("train target must be positive and fit the frozen source pool")
    if template["batch_index"] != 0:
        raise ValueError("campaign template must start at batch zero")
    return template, pool


def _completed(template: dict, pool: dict) -> list[dict]:
    batches = []
    count = math.ceil(pool["split_counts"]["train"] / pool["config"]["batch_counts"]["train"])
    for index in range(count):
        batch = batch_config(template, index)
        path = Path(batch["dataset_dir"]) / "preparation.json"
        if not path.exists():
            continue
        preparation = load_json(path)
        if preparation["source_pool_id"] != pool["pool_id"] or preparation["batch_index"] != index:
            raise ValueError("published batch does not belong to this frozen source pool")
        frozen = load_json(Path(batch["artifacts_dir"]) / "config.json")
        if _without_endpoint(frozen) != _without_endpoint(batch):
            raise ValueError("published batch configuration differs from campaign template")
        batches.append(
            {
                "batch_index": index,
                "dataset_dir": batch["dataset_dir"],
                "artifacts_dir": batch["artifacts_dir"],
                "by_split": preparation["summary"]["by_split"],
            }
        )
    return batches


def report(config: dict) -> dict:
    """Only preparation.json marks published data; ongoing annotation is reported separately."""
    template, pool = _inputs(config)
    completed = _completed(template, pool)
    completed_indices = {b["batch_index"] for b in completed}
    by_split = {
        split: {
            "trajectories": sum(b["by_split"][split]["complete_documents"] for b in completed),
            "qas": sum(b["by_split"][split]["qas"] for b in completed),
        }
        for split in ("train", "dev", "test")
    }
    keys = ("input_tokens", "output_tokens", "network_attempts", "cache_hits", "usage_missing")
    all_usage = dict.fromkeys(keys, 0)
    completed_usage = dict.fromkeys(keys, 0)
    batches = []
    count = math.ceil(pool["split_counts"]["train"] / pool["config"]["batch_counts"]["train"])
    for index in range(count):
        batch = batch_config(template, index)
        root = Path(batch["artifacts_dir"])
        if not root.exists():
            continue
        stages = _request_statistics(root)
        usage = {key: sum(s[key] for s in stages.values()) for key in keys}
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        for key in keys:
            all_usage[key] += usage[key]
            if index in completed_indices:
                completed_usage[key] += usage[key]
        failures = [load_json(p) for p in (root / "failed-documents").glob("doc-*.json")]
        failed_indices = {f["document_index"] for f in failures if f["phase"] == "annotate"}
        finished = 0
        quota_complete = 0
        for path in (root / "documents").glob("doc-*.json"):
            if path.name.endswith(".error.json"):
                continue
            row = load_json(path)
            if row["finished"] and row["document_index"] not in failed_indices:
                finished += 1
                quota_complete += row["assembly"]["ok"]
        batches.append(
            {
                "batch_index": index,
                "published": index in completed_indices,
                "annotation_finished_documents": finished,
                "annotation_quota_complete_documents": quota_complete,
                "document_failures": len(failures),
                "usage": usage,
            }
        )
    for usage in (all_usage, completed_usage):
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    train_count = by_split["train"]["trajectories"]
    state_path = Path(config["artifacts_dir"]) / "status.json"
    state = load_json(state_path) if state_path.exists() else None
    if state is not None:
        state = {key: value for key, value in state.items() if key != "stages"}
    return {
        "observed_at": _now(),
        "target_train_trajectories": config["target_train_trajectories"],
        "remaining_train_trajectories": max(0, config["target_train_trajectories"] - train_count),
        "completed_batches": sorted(completed_indices),
        "completed_by_split": by_split,
        "completed_batches_usage": completed_usage,
        "all_batches_usage": all_usage,
        "completed_batch_tokens_per_train_trajectory": completed_usage["total_tokens"] / train_count
        if train_count
        else None,
        "usage_scope": "Actual requests for all splits and failed documents; completed_batches_usage includes only published batches, all_batches_usage also includes ongoing batches. Missing usage is not estimated.",
        "batches": batches,
        "status": state,
    }


def run(config: dict) -> dict:
    root = Path(config["artifacts_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "campaign.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        template, pool = _inputs(config)
        contract = {
            "config": config,
            "batch_template": _without_endpoint(template),
            "prompts": _prompt_texts(template),
            "source_pool_id": pool["pool_id"],
        }
        contract_path = root / "config.json"
        if contract_path.exists():
            if load_json(contract_path) != contract:
                raise ValueError(
                    "campaign configuration changed; keep the frozen campaign contract"
                )
        else:
            save_json(contract_path, contract)
        state_path = root / "status.json"
        state = (
            load_json(state_path) if state_path.exists() else {"created_at": _now(), "stages": []}
        )
        if state.get("stage_pid") is not None:
            try:
                os.kill(state["stage_pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(
                    "previous stage process still exists; inspect it before resuming"
                )
        state.update(status="running", pid=os.getpid(), updated_at=_now(), error=None)
        save_json(state_path, state)
        try:
            max_batches = math.ceil(
                pool["split_counts"]["train"] / pool["config"]["batch_counts"]["train"]
            )
            for index in range(max_batches + 1):
                completed = _completed(template, pool)
                train_count = sum(b["by_split"]["train"]["complete_documents"] for b in completed)
                state.update(
                    completed_train_trajectories=train_count,
                    target_train_trajectories=config["target_train_trajectories"],
                    updated_at=_now(),
                )
                save_json(state_path, state)
                if train_count >= config["target_train_trajectories"]:
                    save_json(
                        Path(config["dataset_dir"]) / "collection.json",
                        {
                            "created_at": _now(),
                            "source_pool_id": pool["pool_id"],
                            "target_train_trajectories": config["target_train_trajectories"],
                            "train_trajectories": train_count,
                            "batches": completed,
                        },
                    )
                    state.update(
                        status="complete", completed_at=_now(), current_stage=None, stage_pid=None
                    )
                    save_json(state_path, state)
                    return report(config)
                if index == max_batches:
                    raise RuntimeError(
                        "frozen train source pool exhausted before reaching the target"
                    )
                if index in {b["batch_index"] for b in completed}:
                    continue
                batch = batch_config(template, index)
                path = root / "batches" / f"batch-{index:03d}.json"
                save_json(path, batch)
                for stage in STAGES:
                    if _prompt_texts(template) != contract["prompts"]:
                        raise ValueError("prompts changed during campaign")
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "latent_working_memory.data_preparation.fineweb_qa",
                        stage,
                        "--config",
                        str(path),
                    ]
                    entry = {
                        "batch_index": index,
                        "stage": stage,
                        "command": command,
                        "started_at": _now(),
                        "status": "running",
                    }
                    state["stages"].append(entry)
                    state.update(current_batch_index=index, current_stage=stage, updated_at=_now())
                    save_json(state_path, state)
                    log_path = Path(batch["artifacts_dir"]) / f"{stage}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                        state["stage_pid"] = child.pid
                        save_json(state_path, state)
                        code = child.wait()
                    entry.update(
                        exit_code=code,
                        finished_at=_now(),
                        status="complete" if code == 0 else "failed",
                    )
                    state.update(updated_at=_now(), stage_pid=None)
                    save_json(state_path, state)
                    if code:
                        raise RuntimeError(
                            f"batch {index} stage {stage} exited with {code}; see {log_path}"
                        )
        except BaseException as error:
            state.update(
                status="failed", updated_at=_now(), error=f"{type(error).__name__}: {error}"
            )
            save_json(state_path, state)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "report"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    result = run(config) if args.action == "run" else report(config)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
