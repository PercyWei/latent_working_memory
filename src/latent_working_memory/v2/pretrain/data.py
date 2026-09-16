"""启动时从原始文章构造固定轨迹；epoch 只改变索引顺序。"""

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from glob import glob
from itertools import accumulate
from pathlib import Path
import random

import torch

from latent_working_memory.data_preparation.pretrain.config import PreparationConfig
from latent_working_memory.data_preparation.pretrain.sources import collect_sources
from latent_working_memory.v2.pretrain.config import SelectionConfig


@dataclass(frozen=True)
class Document:
    document_id: str
    token_ids: torch.Tensor
    split: str


@dataclass(frozen=True)
class Trajectory:
    document_id: str
    source_start: int
    token_ids: torch.Tensor
    write_ends: tuple[int, ...]
    capacity: int


def segment_lengths(length, rounds, capacity, rng):
    if not rounds * capacity <= length <= 3 * rounds * capacity:
        raise ValueError("total length is not feasible for this number of segments")
    remaining, parts = length, []
    for count in range(rounds, 0, -1):
        low = max(capacity, remaining - 3 * capacity * (count - 1))
        high = min(3 * capacity, remaining - capacity * (count - 1))
        part = rng.randint(low, high)
        parts.append(part)
        remaining -= part
    rng.shuffle(parts)
    return parts


def load_documents(config: SelectionConfig, tokenizer):
    paths = [Path(p) for p in sorted(glob(config.source_glob, recursive=True))]
    if not paths:
        raise ValueError(f"no FineWeb Parquet files match {config.source_glob}")
    sources = collect_sources(
        paths,
        config.source_seed,
        config.split_fractions,
        PreparationConfig(max_documents=config.max_documents),
    )
    eligible = [row for row in sources if row["status"] == "eligible"]
    documents = []
    for start in range(0, len(eligible), 64):
        batch = eligible[start : start + 64]
        ids = tokenizer([row["record"]["text"] for row in batch], add_special_tokens=False)[
            "input_ids"
        ]
        for row, tokens in zip(batch, ids, strict=True):
            documents.append(
                Document(row["record"]["id"], torch.tensor(tokens, dtype=torch.long), row["split"])
            )
    return documents


def build_trajectories(documents, config: SelectionConfig, stage, split):
    if stage not in {"warmup", "multiround"}:
        raise ValueError("unknown stage")
    count = getattr(config, stage)[split]
    if not count:
        return ()
    rng = random.Random(f"{config.seed}:{stage}:{split}")
    pool = sorted((d for d in documents if d.split == split), key=lambda d: len(d.token_ids))
    lengths = [len(d.token_ids) for d in pool]
    k, q = config.capacity, config.continuation_tokens
    required = (8 if stage == "warmup" else 5) * k + q
    if not lengths or lengths[-1] < required:
        raise ValueError(f"{stage}/{split} needs source documents with at least {required} tokens")
    result = []
    for _ in range(count):
        if stage == "warmup":
            length = rng.randint(2 * k, 8 * k)
            rounds, minimum = 1, length + q
        else:
            rounds = rng.choice((3, 4, 5))
            minimum = rounds * k + q
        first = bisect_left(lengths, minimum)
        document = pool[rng.randrange(first, len(pool))]
        if stage == "multiround":
            length = rng.randint(rounds * k, min(8 * k, len(document.token_ids) - q))
        parts = [length] if stage == "warmup" else segment_lengths(length, rounds, k, rng)
        start = rng.randint(0, len(document.token_ids) - length - q)
        result.append(
            Trajectory(
                document.document_id,
                start,
                document.token_ids[start : start + length + q].clone(),
                tuple(accumulate(parts)),
                k,
            )
        )
    return tuple(result)


def build_datasets(documents, config, include_warmup, include_multiround_training=True):
    stages = ("warmup", "multiround") if include_warmup else ("multiround",)
    return {
        stage: {
            split: (
                ()
                if stage == "multiround" and split == "train" and not include_multiround_training
                else build_trajectories(documents, config, stage, split)
            )
            for split in ("train", "dev", "test")
        }
        for stage in stages
    }


def dataset_statistics(datasets):
    result = {}
    for stage, splits in datasets.items():
        result[stage] = {}
        for split, rows in splits.items():
            parts = [
                end - start
                for row in rows
                for start, end in zip((0,) + row.write_ends[:-1], row.write_ends, strict=True)
            ]
            result[stage][split] = {
                "trajectories": len(rows),
                "documents": len({r.document_id for r in rows}),
                "rounds": dict(Counter(len(r.write_ends) for r in rows)),
                "ratio_bins": dict(Counter((r.write_ends[-1] - 1) // r.capacity + 1 for r in rows)),
                "segment_min": min(parts, default=0),
                "segment_max": max(parts, default=0),
                "source_tokens": sum(r.write_ends[-1] for r in rows),
            }
    return result


def epoch_batches(rows, batch_size, seed, stage, epoch):
    indices = list(range(len(rows)))
    random.Random(f"{seed}:order:{stage}:{epoch}").shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [rows[i] for i in indices[start : start + batch_size]]
