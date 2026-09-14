"""按实验配置从基础原文选择内存索引，不生成派生数据副本。"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from latent_working_memory.data_preparation.pretrain.text_samples import (
    TextSample,
    input_text_key,
    tokenizer_identity,
)
from latent_working_memory.v1.config import load_config
from latent_working_memory.v1.data import EpisodeIndex
from latent_working_memory.v1.pretrain.prepared_data import eligible_input_length


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


def select_experiment(spec, config, tokenizer, splits=("train", "dev", "test")):
    sources = {name: Path(path) for name, path in spec["sources"].items()}
    runs = spec["runs"]
    if (
        not sources
        or not runs
        or any(
            not name or Path(name).name != name or name in {".", ".."}
            for name in sources.keys() | runs.keys()
        )
    ):
        raise ValueError("sources and runs require simple non-empty dataset names")
    balanced = spec["balance_task_lengths"]
    requested = spec["samples_per_split"]
    if type(spec["seed"]) is not int or spec["seed"] < 0:
        raise ValueError("selection seed must be a non-negative integer")
    if type(balanced) is not bool or set(requested) != {"train", "dev", "test"}:
        raise ValueError("specify balance_task_lengths and train/dev/test sample counts")
    if any(n is not None and (type(n) is not int or n <= 0) for n in requested.values()):
        raise ValueError("sample counts must be positive integers or null for all")
    if balanced and any(n is None for n in requested.values()):
        raise ValueError("balanced selection requires explicit sample counts")
    for name, weights in runs.items():
        if (
            not weights
            or set(weights) - sources.keys()
            or any(w <= 0 for w in weights.values())
            or not math.isclose(sum(weights.values()), 1)
        ):
            raise ValueError("run source weights must be positive and sum to one")
        if name in sources and weights != {name: 1}:
            raise ValueError("a dataset named after a source must use only that source")
        if requested["train"] is None and len(weights) > 1:
            raise ValueError("mixed training requires a sample count to enforce source proportions")
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
    prompt_lengths = {
        t: len(tokenizer.encode(prompt, add_special_tokens=False))
        for t, prompt in [("ae", config.ae_prompt), ("continuation", config.lm_prompt)]
    }
    cells, registry, seen_content = {}, {}, {}
    rejected = Counter()
    for name, directory in sources.items():
        meta = metadata[name]
        reuse_lengths = meta["tokenizer"] == tokenizer_identity(config)
        method = "pysbd_conservative" if meta["boundary_variant"] == "semantic" else "random_token"
        for split in splits:
            groups = defaultdict(list)
            seen_ids = set()
            with (directory / f"{split}.jsonl").open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    sample = TextSample(**json.loads(line))
                    if sample.sample_id in seen_ids or sample.boundary_method != method:
                        raise ValueError("duplicate sample ID or inconsistent boundary source")
                    seen_ids.add(sample.sample_id)
                    for kind, key in [
                        ("document", sample.document_id),
                        ("source", sample.source_id),
                        ("cluster", sample.dedup_cluster),
                    ]:
                        if registry.setdefault((kind, key), split) != split:
                            raise ValueError("source crosses splits")
                    size = eligible_input_length(
                        sample, tokenizer, config, reuse_lengths, prompt_lengths
                    )
                    if size is None:
                        rejected[f"{name}/{split}/length_or_window"] += 1
                        continue
                    target = sample.text if sample.task == "ae" else sample.continuation
                    content = hashlib.blake2b(
                        json.dumps(
                            (sample.task, input_text_key(sample.text), " ".join(target.split()))
                        ).encode()
                    ).hexdigest()
                    if content in seen_content:
                        if seen_content[content] != split:
                            raise ValueError("duplicate content crosses splits")
                        rejected[f"{name}/{split}/duplicate_content"] += 1
                        continue
                    seen_content[content] = split
                    key = (
                        (sample.task, next(b for b in bounds if size <= b)) if balanced else keys[0]
                    )
                    groups[key].append(
                        (
                            name,
                            offset,
                            sample.sample_id,
                            sample.document_id,
                            sample.source_id,
                            sample.dedup_cluster,
                            sample.task,
                            size,
                        )
                    )
            cells[name, split] = groups
    indices, summaries = {}, {}
    weights_by_name = {name: {name: 1} for name in sources} | runs
    for name, weights in weights_by_name.items():
        for split in (("train",) if name in runs else ()) + (
            ("dev", "test") if name in sources else ()
        ):
            if split not in splits:
                continue
            entries = []
            for source, weight in weights.items():
                for key in keys:
                    pool = list(cells[source, split][key])
                    # Keep the historical per-cell shuffle and final shuffle exactly.
                    random.Random(f"{spec['seed']}:{source}:{split}:{key}").shuffle(pool)
                    quota = cell_counts[split]
                    n = len(pool) if quota is None else quota * weight
                    if not math.isclose(n, round(n)):
                        raise ValueError("sample counts must allow exact source proportions")
                    n = round(n)
                    if len(pool) < n:
                        raise ValueError(
                            f"insufficient data: {source}/{split}/{key} needs {n}, has {len(pool)}"
                        )
                    entries.extend(pool[:n])
            random.Random(spec["seed"]).shuffle(entries)
            paths = {
                source: (sources[source] / f"{split}.jsonl", metadata[source]["boundary_variant"])
                for source in weights
            }
            indices[name, split] = SelectedIndex(paths, entries, tokenizer, config)
            summaries[f"{name}/{split}"] = dict(
                Counter(f"{source}/{task}" for source, _, _, _, _, _, task, _ in entries)
            )
    report = {
        "selection": spec,
        "tokenizer": tokenizer_identity(config),
        "source_preparations": {name: meta["preparation_id"] for name, meta in metadata.items()},
        "counts": {
            f"{name}/{split}": len(index.offsets) for (name, split), index in indices.items()
        },
        "task_source_counts": summaries,
        "rejected": dict(rejected),
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
        "source_weights": report["selection"]["runs"].get(name, {name: 1}),
        "selection": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, revision=config.model_revision, local_files_only=True
    )
    _, report = select_experiment(json.loads(args.spec.read_text()), config, tokenizer)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
