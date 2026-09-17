"""独立构造 Parquet 位置与目标 token 切分计划，按字符估算为候选窗口留余量。"""

import argparse
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from glob import glob
from itertools import accumulate
import json
import math
import os
from pathlib import Path
import random
import uuid

from latent_working_memory.data_preparation.pretrain.config import PreparationConfig
from latent_working_memory.data_preparation.pretrain.sources import collect_sources


STAGE_DIRECTORIES = {"warmup": "single", "multiround": "multi"}


@dataclass(frozen=True)
class DataPreparationConfig:
    source_glob: str
    capacity: int = 512
    continuation_tokens: int = 512
    content_reserve_ratio: float = 1.5
    continuation_reserve_tokens: int = 768
    max_documents: int = 100000
    source_seed: int = 20260907
    seed: int = 20260916
    split_fractions: tuple = (0.9, 0.05, 0.05)
    warmup: dict = field(default_factory=lambda: {"train": 32000, "dev": 128, "test": 128})
    multiround: dict = field(default_factory=lambda: {"train": 32000, "dev": 128, "test": 128})

    def __post_init__(self):
        object.__setattr__(self, "split_fractions", tuple(self.split_fractions))
        if not self.source_glob:
            raise ValueError("source_glob is required")
        for name in (
            "capacity",
            "continuation_tokens",
            "continuation_reserve_tokens",
            "max_documents",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if (
            type(self.content_reserve_ratio) not in (int, float)
            or not math.isfinite(self.content_reserve_ratio)
            or self.content_reserve_ratio < 1
        ):
            raise ValueError("content_reserve_ratio must be finite and at least one")
        if self.continuation_reserve_tokens < self.continuation_tokens:
            raise ValueError("continuation_reserve_tokens must be at least continuation_tokens")
        for name in ("source_seed", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            len(self.split_fractions) != 3
            or any(not math.isfinite(x) or x <= 0 for x in self.split_fractions)
            or not math.isclose(sum(self.split_fractions), 1)
        ):
            raise ValueError("split_fractions must be three positive fractions summing to one")
        for counts in (self.warmup, self.multiround):
            if (
                set(counts) != {"train", "dev", "test"}
                or any(type(n) is not int or n < 0 for n in counts.values())
                or counts["train"] == 0
            ):
                raise ValueError("stage counts require positive train and nonnegative dev/test")

    def candidate_chars(self, content_tokens):
        return (
            math.ceil(4 * content_tokens * self.content_reserve_ratio)
            + 4 * self.continuation_reserve_tokens
        )


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    split: str
    source_file: str
    row_group: int
    row_index: int
    dedup_cluster: str


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


def build_indices(documents, config, stage, split):
    count = getattr(config, stage)[split]
    if not count:
        return []
    rng = random.Random(f"{config.seed}:{stage}:{split}")
    pool = sorted((d for d in documents if d.split == split), key=lambda d: len(d.text))
    lengths = [len(d.text) for d in pool]
    k, reserve = config.capacity, config.continuation_reserve_tokens
    required = config.candidate_chars((8 if stage == "warmup" else 5) * k)
    if not lengths or lengths[-1] < required:
        raise ValueError(
            f"{stage}/{split} needs source documents with at least {required} characters"
        )
    result = []
    for i in range(count):
        if stage == "warmup":
            length = rng.randint(2 * k, 8 * k)
            rounds, minimum = 1, config.candidate_chars(length)
        else:
            rounds = rng.choice((3, 4, 5))
            minimum = config.candidate_chars(rounds * k)
        first = bisect_left(lengths, minimum)
        document = pool[rng.randrange(first, len(pool))]
        if stage == "multiround":
            available = int(
                (len(document.text) - 4 * reserve) // (4 * config.content_reserve_ratio)
            )
            length = rng.randint(rounds * k, min(8 * k, available))
        parts = [length] if stage == "warmup" else segment_lengths(length, rounds, k, rng)
        chars = config.candidate_chars(length)
        start = rng.randint(0, len(document.text) - chars)
        result.append(
            {
                "sample_id": f"{stage}/{split}/{i:06d}",
                "document_id": document.document_id,
                "source_file": document.source_file,
                "row_group": document.row_group,
                "row_index": document.row_index,
                "dedup_cluster": document.dedup_cluster,
                "char_start": start,
                "char_end": start + chars,
                "write_token_ends": list(accumulate(parts)),
                "capacity": k,
            }
        )
    return result


def index_statistics(rows):
    parts = [
        end - start
        for row in rows
        for start, end in zip(
            [0] + row["write_token_ends"][:-1], row["write_token_ends"], strict=True
        )
    ]
    return {
        "trajectories": len(rows),
        "documents": len({r["document_id"] for r in rows}),
        "compressions": dict(Counter(len(r["write_token_ends"]) for r in rows)),
        "planned_ratio_bins": dict(
            Counter(math.ceil(r["write_token_ends"][-1] / r["capacity"]) for r in rows)
        ),
        "segment_tokens_min": min(parts, default=0),
        "segment_tokens_max": max(parts, default=0),
        "planned_source_tokens": sum(r["write_token_ends"][-1] for r in rows),
    }


def prepare_dataset(config, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"use a new dataset directory: {output_dir}")
    paths = [Path(p) for p in sorted(glob(config.source_glob, recursive=True))]
    if not paths:
        raise ValueError(f"no FineWeb Parquet files match {config.source_glob}")
    print("Reading, filtering and clustering source articles", flush=True)
    recipe = PreparationConfig(max_documents=config.max_documents)
    sources = collect_sources(paths, config.source_seed, config.split_fractions, recipe)
    documents = [
        Document(
            row["record"]["id"],
            row["record"]["text"],
            row["split"],
            os.path.relpath(Path(row["location"]["source_file"]).absolute(), output_dir.absolute()),
            row["location"]["row_group"],
            row["location"]["row_index"],
            row["cluster"],
        )
        for row in sources
        if row["status"] == "eligible"
    ]
    datasets = {
        stage: {
            split: build_indices(documents, config, stage, split)
            for split in ("train", "dev", "test")
        }
        for stage in STAGE_DIRECTORIES
    }
    used = {
        r["document_id"] for splits in datasets.values() for rows in splits.values() for r in rows
    }
    output_dir.mkdir(parents=True)
    for stage, splits in datasets.items():
        directory = output_dir / STAGE_DIRECTORIES[stage]
        directory.mkdir()
        for split, rows in splits.items():
            with (directory / f"{split}.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "preparation_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "length_estimation": "candidate characters = ceil(4 * planned content tokens * content_reserve_ratio) + 4 * continuation_reserve_tokens; write_token_ends are target token positions",
        "config": asdict(config),
        "source_recipe": {
            name: getattr(recipe, name)
            for name in (
                "min_document_chars",
                "near_duplicate_threshold",
                "near_duplicate_min_words",
            )
        },
        "source_files": [os.path.relpath(p.absolute(), output_dir.absolute()) for p in paths],
        "source_statistics": dict(Counter(row["status"] for row in sources)),
        "referenced_documents": len(used),
        "statistics": {
            stage: {split: index_statistics(rows) for split, rows in splits.items()}
            for stage, splits in datasets.items()
        },
    }
    (output_dir / "preparation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"Saved single/multi indices referencing {len(used)} source articles to {output_dir}",
        flush=True,
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reconstruction Parquet references and token plans without a tokenizer"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_dataset(DataPreparationConfig(**json.loads(args.config.read_text())), args.output_dir)


if __name__ == "__main__":
    main()
