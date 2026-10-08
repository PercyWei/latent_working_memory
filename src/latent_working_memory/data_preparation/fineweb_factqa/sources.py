"""Append frozen FineWeb candidates and allocate disjoint annotation batches."""

from __future__ import annotations

import copy
import hashlib
import itertools
import uuid
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from latent_working_memory.data_preparation.fineweb_factqa.storage import load_json, save_json
from latent_working_memory.data_preparation.fineweb_source import (
    SPLITS,
    load_previous_sources,
    source_files,
    split_fractions,
    write_used_sources,
)
from latent_working_memory.data_preparation.pretrain.config import PreparationConfig
from latent_working_memory.data_preparation.pretrain.dedup import (
    cluster_documents,
    matching_clusters,
)
from latent_working_memory.data_preparation.pretrain.fineweb import document_split
from latent_working_memory.data_preparation.pretrain.quality import document_rejection_reason
from latent_working_memory.data_preparation.pretrain.sources import (
    parquet_records,
    referenced_records,
)
from latent_working_memory.data_preparation.segmentation import SegmentationConfig, sample_windows


def pool_contract(config: dict) -> dict:
    return {
        key: copy.deepcopy(config[key])
        for key in (
            "source_dir",
            "source_batch_size",
            "source_seed",
            "selection_seed",
            "split_counts",
            "window",
            "batch_split_counts",
        )
    }


def _rank(*parts: object) -> bytes:
    return hashlib.blake2b("\0".join(map(str, parts)).encode(), digest_size=16).digest()


def _validate_pool_config(config: dict) -> SegmentationConfig:
    for key in ("source_batch_size", "source_seed", "selection_seed"):
        value = config[key]
        minimum = 1 if key == "source_batch_size" else 0
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    split_fractions(config["split_counts"])
    counts = config["batch_split_counts"]
    if (
        set(counts) != set(SPLITS)
        or any(type(value) is not int or value < 0 for value in counts.values())
        or any(config["split_counts"][split] > 0 and counts[split] == 0 for split in SPLITS)
    ):
        raise ValueError("batch_split_counts requires positive counts for requested splits")
    if "continuation_tokens" in config["window"]:
        raise ValueError("FactQA window has no configurable continuation_tokens")
    return SegmentationConfig(**config["window"], continuation_tokens=0)


def _write_ledger(directory: Path, pool: dict) -> None:
    # The source-pool commit precedes this ledger; deriving it repairs an interrupted write.
    offsets = {
        split: max((batch["ranges"][split]["stop"] for batch in pool["batches"]), default=0)
        for split in SPLITS
    }
    pool["next_offsets"] = offsets
    references = []
    for split in SPLITS:
        documents = [document for document in pool["documents"] if document["split"] == split]
        references.extend(
            {key: document[key] for key in ("document_id", "dedup_cluster", "source")}
            for document in documents[: offsets[split]]
        )
    write_used_sources(directory, references)


def prepare_pool(config: dict) -> dict:
    """Create or restore a pool; no source is consumed until a batch needs candidates."""
    _validate_pool_config(config)
    directory = Path(config["dataset_dir"])
    path = directory / "source-pool.json"
    contract = pool_contract(config)
    previous = config["previous_datasets"]
    if not isinstance(previous, list) or any(not isinstance(p, str) or not p for p in previous):
        raise ValueError("previous_datasets must be a list of dataset directories")
    if any(Path(p).resolve() == directory.resolve() for p in previous):
        raise ValueError("an appended campaign requires a new dataset directory")
    if path.exists():
        pool = load_json(path)
        if pool["config"] != contract or pool["previous_datasets"] != previous:
            raise ValueError("source pool configuration changed; create a new pool")
    else:
        recipe = PreparationConfig()
        pool = {
            "pool_id": str(uuid.uuid4()),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "config": contract,
            "previous_datasets": list(previous),
            "excluded_sources": load_previous_sources(previous),
            "source_files": [str(p) for p in source_files(config["source_dir"])],
            "source_recipe": {
                key: getattr(recipe, key)
                for key in (
                    "min_document_chars",
                    "near_duplicate_threshold",
                    "near_duplicate_min_words",
                )
            },
            "statistics": {
                "scanned_documents": 0,
                "source_batches": 0,
                "frozen_trajectories": 0,
                "frozen_source_documents": 0,
            },
            "exhausted": False,
            "documents": [],
            "split_counts": dict.fromkeys(SPLITS, 0),
            "segment_counts": {},
            "batches": [],
            "next_offsets": dict.fromkeys(SPLITS, 0),
        }
        save_json(path, pool)
    _write_ledger(directory, pool)
    return pool


def source_records(pool: dict):
    """Open one source stream per run and replay only its persisted prefix on restart."""
    with closing(
        parquet_records([Path(p) for p in pool["source_files"]], pool["config"]["source_seed"])
    ) as records:
        for _ in range(pool["statistics"]["scanned_documents"]):
            if next(records, None) is None:
                raise ValueError("FineWeb source ended before its saved scan position")
        yield from records


def _extend_pool(config: dict, pool: dict, records) -> None:
    batch = list(itertools.islice(records, config["source_batch_size"]))
    counts = Counter(pool["statistics"])
    counts["scanned_documents"] += len(batch)
    counts["source_batches"] += bool(batch)
    pool["exhausted"] = len(batch) < config["source_batch_size"]
    segmentation = SegmentationConfig(**config["window"], continuation_tokens=0)
    recipe = PreparationConfig(**pool["source_recipe"])
    candidates = []
    for record, location in batch:
        if document_rejection_reason(record, recipe.min_document_chars) is not None:
            counts["basic_rejected_documents"] += 1
        elif len(record["text"]) < segmentation.minimum_window_chars:
            counts["length_failed"] += 1
        else:
            candidates.append((record, location))
    raw = [record for record, _ in candidates]
    clusters = cluster_documents(raw, recipe)
    previous = matching_clusters(
        referenced_records(pool["excluded_sources"]), raw, clusters, recipe
    )
    frozen_sources = {document["document_id"]: document for document in pool["documents"]}
    frozen = matching_clusters(referenced_records(frozen_sources.values()), raw, clusters, recipe)
    available, seen = [], set()
    for (record, location), cluster in zip(candidates, clusters, strict=True):
        if cluster in previous:
            counts["previously_used_documents"] += 1
        elif cluster in frozen:
            counts["previous_pool_duplicate_documents"] += 1
        elif cluster in seen:
            counts["within_scan_duplicate_documents"] += 1
        else:
            seen.add(cluster)
            available.append((record, location, cluster))
    available.sort(key=lambda item: (_rank(config["selection_seed"], item[0]["id"]), item[0]["id"]))
    fractions = split_fractions(config["split_counts"])
    for record, location, cluster in available:
        split = document_split(cluster, config["source_seed"], fractions)
        for start, parts, end in sample_windows(
            len(record["text"]), record["id"], config["selection_seed"], segmentation
        ):
            cuts = [0, *itertools.accumulate(segmentation.reserved_chars(part) for part in parts)]
            pool["documents"].append(
                {
                    "pool_index": len(pool["documents"]),
                    "trajectory_id": f"{record['id']}:{start}:{end}",
                    "document_id": record["id"],
                    "dedup_cluster": cluster,
                    "split": split,
                    "source": {
                        "file": location["source_file"],
                        "row_group": location["row_group"],
                        "row_index": location["row_index"],
                    },
                    "window_char_span": [start, end],
                    "segments": [
                        {"segment_id": f"seg{i}", "char_span": [left, right]}
                        for i, (left, right) in enumerate(zip(cuts, cuts[1:]))
                    ],
                }
            )
            pool["split_counts"][split] += 1
    counts["frozen_trajectories"] = len(pool["documents"])
    counts["frozen_source_documents"] = len({d["document_id"] for d in pool["documents"]})
    pool["statistics"] = dict(counts)
    pool["segment_counts"] = dict(Counter(str(len(d["segments"])) for d in pool["documents"]))
    # Persist the cursor and all new frozen candidates together, before allocating any of them.
    save_json(Path(config["dataset_dir"]) / "source-pool.json", pool)


def allocate_batch(config: dict, pool: dict, records, requested_counts: dict) -> dict:
    """Allocate available candidates first; extend only when all still-needed splits are empty."""
    if (
        set(requested_counts) != set(SPLITS)
        or any(type(value) is not int or value < 0 for value in requested_counts.values())
        or not any(requested_counts.values())
    ):
        raise ValueError("requested_counts requires nonnegative split counts and a positive total")
    while not any(
        requested_counts[split] and pool["next_offsets"][split] < pool["split_counts"][split]
        for split in SPLITS
    ):
        if pool["exhausted"]:
            missing = ", ".join(
                f"{split}={count}" for split, count in requested_counts.items() if count
            )
            raise ValueError(f"FineWeb source exhausted; missing candidates: {missing}")
        _extend_pool(config, pool, records)
    ranges = {}
    for split in SPLITS:
        start = pool["next_offsets"][split]
        stop = min(start + requested_counts[split], pool["split_counts"][split])
        ranges[split] = {
            "start": start,
            "stop": stop,
            "requested": requested_counts[split],
            "selected": stop - start,
        }
        pool["next_offsets"][split] = stop
    entry = {"batch_index": len(pool["batches"]), "ranges": ranges}
    pool["batches"].append(entry)
    save_json(Path(config["dataset_dir"]) / "source-pool.json", pool)
    _write_ledger(Path(config["dataset_dir"]), pool)
    return entry


def batch_ranges(pool: dict, index: int) -> dict:
    if type(index) is not int or not 0 <= index < len(pool["batches"]):
        raise ValueError(f"batch_index is not an allocated source batch: {index}")
    return pool["batches"][index]["ranges"]


def prepare_selection(config: dict) -> dict:
    """Read the fixed text windows assigned before annotation, without reclustering."""
    pool = load_json(Path(config["source_pool_dir"]) / "source-pool.json")
    index = config["batch_index"]
    ranges = batch_ranges(pool, index)
    selected = []
    for split, bounds in ranges.items():
        candidates = [d for d in pool["documents"] if d["split"] == split]
        selected.extend(dict(d) for d in candidates[bounds["start"] : bounds["stop"]])
    grouped = defaultdict(list)
    for doc in selected:
        grouped[(doc["source"]["file"], doc["source"]["row_group"])].append(doc)
    for (filename, group), docs in grouped.items():
        with pq.ParquetFile(filename) as parquet:
            table = parquet.read_row_group(group, columns=["id", "text"])
        for doc in docs:
            row = table.slice(doc["source"]["row_index"], 1).to_pylist()[0]
            if row["id"] != doc["document_id"]:
                raise ValueError("source document differs from frozen pool")
            doc["text"] = row["text"][slice(*doc["window_char_span"])]
            if len(doc["text"]) != doc["window_char_span"][1] - doc["window_char_span"][0]:
                raise ValueError("source text is shorter than its frozen window")
    return {
        "source_pool_id": pool["pool_id"],
        "source_pool_config": pool["config"],
        "batch_index": index,
        "ranges": ranges,
        "documents": selected,
        "statistics": {
            "frozen_trajectories": len(selected),
            "frozen_source_documents": len({d["document_id"] for d in selected}),
            "selected_by_split": {key: value["selected"] for key, value in ranges.items()},
            "available_by_split": {
                split: pool["split_counts"][split] - bounds["start"]
                for split, bounds in ranges.items()
            },
        },
    }
