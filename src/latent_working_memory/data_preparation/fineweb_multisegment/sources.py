"""持续读取来源，配额满足后才固定累计合格候选的去重簇与划分。"""

from collections import Counter
from contextlib import closing
from itertools import islice

from latent_working_memory.data_preparation.fineweb_source import split_fractions
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


def collect_documents(paths, config, previous, recipe):
    """累计候选参与重聚类，避免后续桥接文档改变已写出的簇或 split。"""
    minimum_chars = config.window.minimum_window_chars
    fractions = split_fractions(config.split_counts)
    previous_records = list(referenced_records(previous))
    candidates, document_ids = [], set()
    statistics = Counter(
        scanned_documents=0,
        source_batches=0,
        basic_rejected=0,
        length_rejected=0,
        duplicate_id=0,
    )
    selected_by_split = dict.fromkeys(config.split_counts, 0)
    with closing(parquet_records(paths, config.source_seed)) as records:
        while True:
            batch = list(islice(records, config.source_batch_size))
            if not batch:
                deficits = ", ".join(
                    f"{split}={count - selected_by_split[split]}"
                    for split, count in config.split_counts.items()
                    if count > selected_by_split[split]
                )
                raise ValueError(f"FineWeb source exhausted; missing trajectories: {deficits}")
            statistics["source_batches"] += 1
            statistics["scanned_documents"] += len(batch)
            added = 0
            for record, location in batch:
                if document_rejection_reason(record, recipe.min_document_chars) is not None:
                    statistics["basic_rejected"] += 1
                elif len(record["text"]) < minimum_chars:
                    statistics["length_rejected"] += 1
                elif record["id"] in document_ids:
                    statistics["duplicate_id"] += 1
                else:
                    candidates.append((record, location))
                    document_ids.add(record["id"])
                    added += 1
            if not added:
                continue

            raw = [record for record, _ in candidates]
            clusters = cluster_documents(raw, recipe)
            excluded = (
                matching_clusters(previous_records, raw, clusters, recipe)
                if previous_records
                else set()
            )
            selected, seen = [], set()
            available_by_split = dict.fromkeys(config.split_counts, 0)
            selected_by_split = dict.fromkeys(config.split_counts, 0)
            duplicates = previously_used = 0
            for (record, location), cluster in zip(candidates, clusters, strict=True):
                if cluster in excluded:
                    previously_used += 1
                    continue
                if cluster in seen:
                    duplicates += 1
                    continue
                seen.add(cluster)
                split = document_split(cluster, config.source_seed, fractions)
                available_by_split[split] += 1
                if selected_by_split[split] < config.split_counts[split]:
                    selected.append(
                        {"record": record, "location": location, "cluster": cluster, "split": split}
                    )
                    selected_by_split[split] += 1
            if selected_by_split == config.split_counts:
                return selected, {
                    **statistics,
                    "candidate_documents": len(candidates),
                    "duplicate": duplicates,
                    "previously_used": previously_used,
                    "eligible": len(seen),
                    "selected": len(selected),
                    "available_by_split": available_by_split,
                    "selected_by_split": selected_by_split,
                }
