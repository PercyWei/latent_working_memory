"""从既有 FineWeb 来源池构造短文本 AE/LM 目标对比的共享数据。"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import uuid
import os

from transformers import AutoTokenizer

from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.data_preparation.fineweb import SemanticSpans
from latent_working_memory.data_preparation.truncation import RandomSpans
from latent_working_memory.v1.config import load_config
from latent_working_memory.data_preparation.text_samples import TextSample, tokenizer_identity
from latent_working_memory.data_preparation.audit import length_statistics


def initialize_worker(config_path, recipe_mapping):
    global tokenizer, config, recipe
    config = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, revision=config.model_revision, local_files_only=True
    )
    recipe = PreparationConfig.from_mapping(recipe_mapping)


def candidates_for_document(source):
    record = source["record"]
    result = []
    for name, constructor in (("semantic", SemanticSpans), ("random", RandomSpans)):
        sampler = constructor(record, tokenizer, config, recipe)
        for task in ("ae", "continuation"):
            for lower, upper in recipe.length_intervals():
                if not sampler.available(task, lower, upper):
                    continue
                rng = random.Random(f"{config.data_seed}:{record['id']}:{name}:{task}:{upper}")
                for _ in range(8):
                    episode = sampler.sample(task, lower, upper, rng)
                    if episode is None:
                        continue
                    provenance = episode.sources[0].provenance
                    x = record["text"][slice(*provenance["x_char_span"])]
                    input_key = hashlib.blake2b(" ".join(x.split()).encode()).hexdigest()
                    provenance.update(dedup_cluster=source["cluster"], input_text_key=input_key)
                    y = " ".join(episode.reads[0].references[0].text.split())
                    key = hashlib.blake2b(json.dumps((task, input_key, y)).encode()).hexdigest()
                    sample = TextSample.from_episode(episode, record["text"], tokenizer)
                    result.append((name, task, upper, key, sample.to_record()))
                    break
    return source["split"], result


def prepare(spec, config_path, output):
    if output.exists():
        raise FileExistsError(output)
    cfg = load_config(config_path)
    root = Path(spec["source_root"])
    source_metadata = json.loads((root / "source-pool.json").read_text())
    if source_metadata["data_seed"] != cfg.data_seed or source_metadata["split_fractions"] != list(
        cfg.split_fractions
    ):
        raise ValueError("source split protocol differs")
    rows = []
    with (root / "sources.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["status"] == "eligible":
                rows.append(row)
    random.Random(spec["seed"]).shuffle(rows)
    recipe_config = PreparationConfig.from_mapping(spec["recipe"])
    bounds = recipe_config.length_bounds
    train_quota = spec["train_per_source_task"]
    eval_quota = spec["evaluation_per_source_task"]
    if train_quota % len(bounds) or eval_quota % len(bounds):
        raise ValueError("quotas must divide across length bins")
    quotas = {
        split: (train_quota if split == "train" else eval_quota) // len(bounds)
        for split in ("train", "dev", "test")
    }
    cells = {
        (split, name, task, upper): []
        for split in quotas
        for name in ("semantic", "random")
        for task in ("ae", "continuation")
        for upper in bounds
    }
    seen_content, fragment_splits = set(), {}
    eval_documents = set()
    considered = Counter()
    with ProcessPoolExecutor(
        max_workers=spec["workers"],
        initializer=initialize_worker,
        initargs=(str(config_path), recipe_config.to_dict()),
    ) as pool:
        # Bound submitted work; stop once all data quotas are met.
        for offset in range(0, len(rows), 64):
            batch = [
                row
                for row in rows[offset : offset + 64]
                if any(len(v) < quotas[k[0]] for k, v in cells.items() if k[0] == row["split"])
            ]
            for split, candidates in pool.map(candidates_for_document, batch):
                considered[split] += 1
                # Prefer the least-filled relative cell, independent of source iteration order.
                candidates.sort(key=lambda c: len(cells[split, c[0], c[1], c[2]]))
                for name, task, upper, content_key, episode in candidates:
                    cell = cells[split, name, task, upper]
                    if len(cell) >= quotas[split]:
                        continue
                    document = episode["document_id"]
                    if split != "train" and document in eval_documents:
                        continue
                    input_key = hashlib.blake2b(
                        " ".join(episode["text"].split()).encode()
                    ).hexdigest()
                    if (
                        content_key in seen_content
                        or fragment_splits.get(input_key, split) != split
                    ):
                        continue
                    seen_content.add(content_key)
                    fragment_splits[input_key] = split
                    cell.append(episode)
                    if split != "train":
                        eval_documents.add(document)
            if offset % 1024 == 0:
                print(
                    json.dumps(
                        {
                            "candidate_documents": dict(considered),
                            "selected": {
                                s: sum(len(v) for k, v in cells.items() if k[0] == s)
                                for s in quotas
                            },
                        }
                    ),
                    flush=True,
                )
            if all(len(v) == quotas[k[0]] for k, v in cells.items()):
                break
    missing = {
        "/".join(map(str, k)): quotas[k[0]] - len(v)
        for k, v in cells.items()
        if len(v) != quotas[k[0]]
    }
    if missing:
        raise ValueError(f"source pool cannot meet requested short-text quotas: {missing}")
    output.mkdir(parents=True)
    report = {
        "construction": spec,
        "source_pool_id": source_metadata["source_pool_id"],
        "candidate_documents": dict(considered),
        "datasets": {},
    }
    for name in ("semantic", "random"):
        directory = output / name
        directory.mkdir()
        counts, histogram, lengths = {}, {}, {}
        for split in ("train", "dev", "test"):
            examples = [
                e for k, values in cells.items() for e in values if k[0] == split and k[1] == name
            ]
            random.Random(f"{spec['seed']}:{name}:{split}").shuffle(examples)
            with (directory / f"{split}.jsonl").open("w") as handle:
                for row in examples:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            documents = {e["document_id"] for e in examples}
            if split != "train" and len(documents) != 2 * eval_quota:
                raise ValueError("evaluation panel does not meet independent-document quota")
            counts[f"{split}/documents"] = len(documents)
            histogram[split] = {}
            for task in ("ae", "continuation"):
                rows_for_task = [e for e in examples if e["task"] == task]
                counts[f"{split}/{task}"] = len(rows_for_task)
                histogram[split][task] = {str(b): len(cells[split, name, task, b]) for b in bounds}
                lengths[f"{split}/{task}/input"] = length_statistics(
                    [e["reference_input_tokens"] for e in rows_for_task]
                )
                if task == "continuation":
                    lengths[f"{split}/{task}/target"] = length_statistics(
                        [e["reference_target_tokens"] for e in rows_for_task]
                    )
        metadata = {
            "preparation_id": str(uuid.uuid4()),
            "source_pool_id": source_metadata["source_pool_id"],
            "source_pool": os.path.relpath(root / "source-pool.json", directory),
            "boundary_variant": name,
            "tokenizer": tokenizer_identity(cfg),
            "recipe": recipe_config.to_dict()
            | {"samples_per_task": [train_quota, eval_quota, eval_quota]},
            "construction": spec,
            "statistics": counts,
            "input_histogram": histogram,
            "lengths": lengths,
            "checks": {"source_split_isolation": True, "independent_evaluation_documents": True},
        }
        (directory / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
        report["datasets"][name] = metadata
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(json.loads(args.spec.read_text()), args.config, args.output_dir)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
