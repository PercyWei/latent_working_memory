"""按实验配置从基础原文选择内存索引，不生成派生数据副本。"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from latent_working_memory.data_preparation.pretrain.text_samples import (
    TextSample,
    tokenizer_identity,
)
from latent_working_memory.v1.config import load_config
from latent_working_memory.v1.data import EpisodeIndex
from latent_working_memory.v1.pretrain.tokenization import TokenizationPool, text_blocks
from latent_working_memory.v1.pretrain.curriculum import validate_curriculum


class SelectedIndex(EpisodeIndex):
    def __init__(self, sources, entries, tokenizer, config):
        self.sources, self.entries = sources, entries
        self.tokenizer, self.config = tokenizer, config
        self.offsets, self.ids, self.input_lengths, self.tasks = [], [], [], []
        self.groups, self.source_ids, self.cluster_ids = {}, set(), set()
        for name, offset, sample_id, document_id, source_id, cluster, task, size in entries:
            self.groups.setdefault(document_id, []).append(len(self.offsets))
            self.offsets.append(offset)
            self.ids.append(sample_id)
            self.input_lengths.append(size)
            self.tasks.append(task)
            self.source_ids.add(source_id)
            self.cluster_ids.add(cluster)
        if not entries or len(set(self.ids)) != len(self.ids):
            raise ValueError("selection is empty or contains duplicate sample IDs")

    def __getitem__(self, index):
        name, offset, *_ = self.entries[index]
        path, variant = self.sources[name]
        with path.open("rb") as handle:
            handle.seek(offset)
            sample = TextSample(**json.loads(handle.readline()))
        return sample.to_episode(self.tokenizer, self.config, variant)

    def batch_entries(self, indices):
        return [
            (
                self.sources[self.entries[i][0]][0],
                self.entries[i][1],
                self.sources[self.entries[i][0]][1],
            )
            for i in indices
        ]


def validate_selection(spec):
    if set(spec) != {"sources", "seed", "training", "evaluation"}:
        raise ValueError("selection requires sources, seed, training and evaluation")
    if not spec["sources"] or any(
        not n or Path(n).name != n or n in {".", "..", "train"} for n in spec["sources"]
    ):
        raise ValueError("sources require simple names other than train")
    if type(spec["seed"]) is not int or spec["seed"] < 0:
        raise ValueError("selection seed must be a non-negative integer")
    evaluation = spec["evaluation"]
    if (
        set(evaluation) != {"balance_task_lengths", "samples_per_source"}
        or type(evaluation["balance_task_lengths"]) is not bool
    ):
        raise ValueError("evaluation requires balance_task_lengths and samples_per_source")
    counts = evaluation["samples_per_source"]
    if set(counts) != {"dev", "test"} or any(
        n is not None and (type(n) is not int or n <= 0) for n in counts.values()
    ):
        raise ValueError("evaluation sample counts must be positive or null")


def select_experiment(
    spec,
    config,
    tokenizer,
    splits=("train", "dev", "test"),
    tokenization=None,
    tokenization_batch_size=256,
):
    validate_selection(spec)
    sources = {name: Path(path) for name, path in spec["sources"].items()}
    validate_curriculum(spec["training"], sources, config.input_length_bounds)
    if config.input_length_bounds[-1] < config.max_input_tokens:
        raise ValueError("length bounds must cover max_input_tokens")
    balanced = spec["evaluation"]["balance_task_lengths"]
    requested = spec["evaluation"]["samples_per_source"]
    bounds = config.input_length_bounds
    keys = [(t, b) for t in ("ae", "continuation") for b in bounds] if balanced else [("all", 0)]
    cell_counts = {}
    for split, n in requested.items():
        if n is not None and n % len(keys):
            raise ValueError(f"{split} sample count must divide evenly across task/length cells")
        cell_counts[split] = n // len(keys) if n is not None else None
    metadata = {
        name: json.loads((directory / "preparation.json").read_text())
        for name, directory in sources.items()
    }
    tokenization = tokenization or TokenizationPool(tokenizer, config)
    cells, registry, seen_content = {}, {}, {}
    rejected = Counter()
    for name, directory in sources.items():
        meta = metadata[name]
        method = "pysbd_conservative" if meta["boundary_variant"] == "semantic" else "random_token"
        for split in splits:
            groups = defaultdict(list)
            seen_ids = set()
            blocks = text_blocks(directory / f"{split}.jsonl", tokenization_batch_size)
            for rows in tokenization.inspect_blocks(blocks):
                for (
                    offset,
                    sample_id,
                    document_id,
                    source_id,
                    cluster,
                    task,
                    boundary,
                    size,
                    content,
                ) in rows:
                    if sample_id in seen_ids or boundary != method:
                        raise ValueError("duplicate sample ID or inconsistent boundary source")
                    seen_ids.add(sample_id)
                    for kind, key in [
                        ("document", document_id),
                        ("source", source_id),
                        ("cluster", cluster),
                    ]:
                        if registry.setdefault((kind, key), split) != split:
                            raise ValueError("source crosses splits")
                    if size is None:
                        rejected[f"{name}/{split}/length_or_window"] += 1
                        continue
                    if content in seen_content:
                        if seen_content[content] != split:
                            raise ValueError("duplicate content crosses splits")
                        rejected[f"{name}/{split}/duplicate_content"] += 1
                        continue
                    seen_content[content] = split
                    key = (
                        (task, next(b for b in bounds if size <= b))
                        if split != "train" and balanced
                        else ("all", 0)
                    )
                    groups[key].append(
                        (name, offset, sample_id, document_id, source_id, cluster, task, size)
                    )
            cells[name, split] = groups
    indices, summaries, resolved_quotas = {}, {}, {}
    if "train" in splits:
        entries = [
            entry for name in sources for pool in cells[name, "train"].values() for entry in pool
        ]
        paths = {
            name: (directory / "train.jsonl", metadata[name]["boundary_variant"])
            for name, directory in sources.items()
        }
        indices["train", "train"] = SelectedIndex(paths, entries, tokenizer, config)
        summaries["train/train"] = dict(Counter(f"{e[0]}/{e[6]}" for e in entries))
    for name, directory in sources.items():
        for split in ("dev", "test"):
            if split not in splits:
                continue
            quota = cell_counts[split]
            if quota is None and balanced:
                quota = min(len(cells[name, split][key]) for key in keys)
            if quota == 0:
                raise ValueError(f"insufficient data for {name}/{split}")
            entries = []
            for key in keys:
                pool = list(cells[name, split][key])
                random.Random(f"{spec['seed']}:{name}:{split}:{key}").shuffle(pool)
                n = len(pool) if quota is None else quota
                if len(pool) < n:
                    raise ValueError(
                        f"insufficient data: {name}/{split}/{key} needs {n}, has {len(pool)}"
                    )
                entries.extend(pool[:n])
            random.Random(spec["seed"]).shuffle(entries)
            indices[name, split] = SelectedIndex(
                {name: (directory / f"{split}.jsonl", metadata[name]["boundary_variant"])},
                entries,
                tokenizer,
                config,
            )
            summaries[f"{name}/{split}"] = dict(Counter(f"{e[0]}/{e[6]}" for e in entries))
            resolved_quotas[f"{name}/{split}"] = quota
    report = {
        "selection": spec,
        "tokenizer": tokenizer_identity(config),
        "source_preparations": {name: meta["preparation_id"] for name, meta in metadata.items()},
        "counts": {
            f"{name}/{split}": len(index.offsets) for (name, split), index in indices.items()
        },
        "task_source_counts": summaries,
        "rejected": dict(rejected),
        "samples_per_cell": resolved_quotas,
    }
    return indices, report


def selection_metadata(report, name):
    identity = {
        "source_preparations": report["source_preparations"],
        "selection": report["selection"],
        "dataset": name,
    }
    return {
        "preparation_id": hashlib.blake2b(
            json.dumps(identity, sort_keys=True).encode(), digest_size=16
        ).hexdigest(),
        "sources": sorted(
            {
                source
                for point in report["selection"]["training"]["source_schedule"]
                for source, weight in point["weights"].items()
                if weight > 0
            }
        )
        if name == "train"
        else [name],
        "selection": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-workers", type=int, default=4)
    parser.add_argument("--tokenization-batch-size", type=int, default=256)
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, revision=config.model_revision, local_files_only=True
    )
    with TokenizationPool(tokenizer, config, args.tokenizer_workers) as tokenization:
        _, report = select_experiment(
            json.loads(args.spec.read_text()),
            config,
            tokenizer,
            tokenization=tokenization,
            tokenization_batch_size=args.tokenization_batch_size,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
