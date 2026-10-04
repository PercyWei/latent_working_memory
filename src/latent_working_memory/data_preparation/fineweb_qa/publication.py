"""Publish one dataset from completed batches and record only this dataset's source use."""

import json
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from latent_working_memory.data_preparation.fineweb_qa.pipeline import _request_statistics
from latent_working_memory.data_preparation.fineweb_qa.storage import load_json, save_json


def publish_dataset(config: dict, template: dict, pool: dict, batches: list[dict]) -> dict:
    root = Path(config["dataset_dir"])
    preparations = [load_json(Path(b["dataset_dir"]) / "preparation.json") for b in batches]
    by_split = {s: Counter() for s in ("train", "dev", "test")}
    stage_counts, request_statistics = defaultdict(Counter), defaultdict(Counter)
    documents, batch_reports, current_sources = [], [], []
    source_by_id, source_batch = {}, {}
    excluded_ids = {d["document_id"] for d in pool["excluded_sources"]}
    excluded_clusters = {d["dedup_cluster"] for d in pool["excluded_sources"]}
    for batch, preparation in zip(batches, preparations, strict=True):
        for split, bounds in batch["source_ranges"].items():
            candidates = [d for d in pool["documents"] if d["split"] == split]
            selected = candidates[bounds["start"] : bounds["stop"]]
            if len(selected) != preparation["summary"]["by_split"][split]["frozen_documents"]:
                raise ValueError("batch source count differs from its preparation")
            for document in selected:
                identity = document["document_id"]
                if (
                    identity in source_by_id
                    or identity in excluded_ids
                    or document["dedup_cluster"] in excluded_clusters
                ):
                    raise ValueError("a previously used source was assigned again")
                source_by_id[identity] = document
                source_batch[identity] = batch["batch_index"]
                current_sources.append(
                    {key: document[key] for key in ("document_id", "dedup_cluster", "source")}
                )
        for split, counts in preparation["summary"]["by_split"].items():
            by_split[split].update(counts)
        for segment, counts in preparation["stage_counts"].items():
            stage_counts[segment].update(counts)
        for stage, counts in _request_statistics(Path(batch["artifacts_dir"])).items():
            request_statistics[stage].update(counts)
        documents.extend(
            dict(d, batch_index=batch["batch_index"]) for d in preparation["documents"]
        )
        batch_reports.append(
            {
                "batch_index": batch["batch_index"],
                "artifacts_dir": batch["artifacts_dir"],
                "source_ranges": batch["source_ranges"],
                "summary": preparation["summary"],
                "supplementation": preparation["supplementation"],
            }
        )
    if by_split["train"]["complete_documents"] < config["target_train_trajectories"]:
        raise ValueError("cannot publish before the current run reaches its train target")
    if len(documents) != len(current_sources):
        raise ValueError("final document outcomes do not cover all assigned sources")

    # Validate all staged splits before replacing any public file. The metadata is
    # the completion marker and is written only after all three splits are ready.
    root.mkdir(parents=True, exist_ok=True)
    temporary = {s: root / f".{s}.{uuid.uuid4().hex}.tmp" for s in by_split}
    seen_documents, seen_trajectories = set(), set()
    try:
        for split, path in temporary.items():
            with path.open("xb") as output:
                for batch, preparation in zip(batches, preparations, strict=True):
                    count = qas = 0
                    with (Path(batch["dataset_dir"]) / f"{split}.jsonl").open("rb") as source:
                        for line in source:
                            row = json.loads(line)
                            identity = row["document_id"]
                            if (
                                row["split"] != split
                                or identity in seen_documents
                                or row["trajectory_id"] in seen_trajectories
                            ):
                                raise ValueError(
                                    "duplicate document or incorrect split in staged data"
                                )
                            expected = source_by_id[identity]
                            if source_batch[identity] != batch["batch_index"]:
                                raise ValueError("staged trajectory belongs to a different batch")
                            if any(
                                row[k] != expected[k]
                                for k in ("split", "source", "trajectory_id", "dedup_cluster")
                            ):
                                raise ValueError("staged trajectory differs from its frozen source")
                            seen_documents.add(identity)
                            seen_trajectories.add(row["trajectory_id"])
                            count += 1
                            qas += len(row["qas"])
                            output.write(line)
                    expected_counts = preparation["summary"]["by_split"][split]
                    if (count, qas) != (
                        expected_counts["complete_documents"],
                        expected_counts["qas"],
                    ):
                        raise ValueError("staged data counts differ from preparation")
                output.flush()
                os.fsync(output.fileno())

        complete = sum(d["complete_documents"] for d in by_split.values())
        usage_keys = (
            "input_tokens",
            "output_tokens",
            "network_attempts",
            "cache_hits",
            "usage_missing",
        )
        usage = {key: sum(s[key] for s in request_statistics.values()) for key in usage_keys}
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        marker = root / "preparation.json"
        result = {
            "created_at": load_json(marker)["created_at"]
            if marker.exists()
            else datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": root.name,
            "source_pool_id": pool["pool_id"],
            "source_pool_config": pool["config"],
            "previous_datasets": config["previous_datasets"],
            "target_train_trajectories": config["target_train_trajectories"],
            "qa": template["qa"],
            "annotation": {k: v for k, v in template["annotation"].items() if k != "endpoint"},
            "prompts": preparations[0]["prompts"],
            "summary": {
                "frozen_documents": len(current_sources),
                "complete_documents": complete,
                "failed_documents": len(current_sources) - complete,
                "failure_counts": dict(
                    Counter(d["failure_reason"] for d in documents if d["failure_reason"])
                ),
                "final_qas": sum(d["qas"] for d in by_split.values()),
                "by_split": {s: dict(counts) for s, counts in by_split.items()},
            },
            "used_sources": current_sources,
            "stage_counts": {k: dict(v) for k, v in stage_counts.items()},
            "request_statistics": {k: dict(v) for k, v in request_statistics.items()},
            "usage": usage,
            "documents": documents,
            "batches": batch_reports,
            "artifacts_dir": config["artifacts_dir"],
        }
        marker.unlink(missing_ok=True)
        for split, path in temporary.items():
            os.replace(path, root / f"{split}.jsonl")
        save_json(marker, result)
        return result
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
