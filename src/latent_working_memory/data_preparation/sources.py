from __future__ import annotations

import itertools
import json
import random
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.data_preparation.dedup import cluster_documents
from latent_working_memory.data_preparation.fineweb import document_split
from latent_working_memory.data_preparation.quality import document_rejection_reason


def parquet_records(files: list[Path], seed: int) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    paths = files.copy()
    rng.shuffle(paths)
    # Interleave shuffled row groups across files so a small budget covers the corpus.
    with ExitStack() as stack:
        streams = []
        for path in paths:
            parquet = stack.enter_context(pq.ParquetFile(path))
            groups = list(range(parquet.num_row_groups))
            rng.shuffle(groups)
            streams.append(
                parquet.iter_batches(batch_size=256, row_groups=groups, use_threads=False)
            )
        while streams:
            active = []
            for stream in streams:
                batch = next(stream, None)
                if batch is None:
                    continue
                records = batch.to_pylist()
                rng.shuffle(records)
                yield from records
                active.append(stream)
            streams = active


def collect_sources(
    files: list[Path], data_seed: int, split_fractions: tuple[float, ...], recipe: PreparationConfig
) -> list[dict]:
    """Rebuild candidate assignments in memory from the original Parquet files."""
    with closing(parquet_records(files, data_seed)) as records:
        candidates = list(itertools.islice(records, recipe.max_documents))
    reasons = [document_rejection_reason(row, recipe.min_document_chars) for row in candidates]
    clusters = cluster_documents(candidates, recipe)
    seen, rows = set(), []
    for record, cluster, reason in zip(candidates, clusters, reasons, strict=True):
        status = "basic_rejected" if reason else "duplicate" if cluster in seen else "eligible"
        if status == "eligible":
            seen.add(cluster)
        rows.append(
            {
                "record": record,
                "cluster": cluster,
                "split": document_split(cluster, data_seed, split_fractions),
                "status": status,
                "reason": reason,
            }
        )
    return rows


def load_sources(metadata_path: Path) -> list[dict]:
    metadata = json.loads(metadata_path.read_text())
    recipe = PreparationConfig.from_mapping(metadata["recipe"])
    return collect_sources(
        [Path(p) for p in metadata["source_files"]],
        metadata["data_seed"],
        tuple(metadata["split_fractions"]),
        recipe,
    )
