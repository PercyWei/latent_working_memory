from __future__ import annotations

import random
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq


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
