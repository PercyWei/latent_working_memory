"""从已构造数据选择等规模实验样本，保留原始 split 与文本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import uuid
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from latent_working_memory.v1.config import load_config
from latent_working_memory.v1.data import Episode
from latent_working_memory.v1.sampling import capacity_weights, read_tokens
from latent_working_memory.data_preparation.fineweb import data_contract


def prepare_experiment(spec: dict, config, tokenizer, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    sources = {name: Path(path) for name, path in spec["sources"].items()}
    runs = spec["runs"]
    for weights in runs.values():
        if (
            not weights
            or set(weights) - sources.keys()
            or any(w <= 0 for w in weights.values())
            or not math.isclose(sum(weights.values()), 1)
        ):
            raise ValueError("run source weights must be positive and sum to one")
    for name in sources.keys() & runs.keys():
        if runs[name] != {name: 1}:
            raise ValueError("a dataset named after a source must use only that source")
    bounds = config.input_length_bounds
    cells = {}
    identities = {}
    registry = {}
    rejected = defaultdict(int)
    seen_content = {}
    for name, directory in sources.items():
        meta = json.loads((directory / "preparation.json").read_text())
        if meta["contract"] != data_contract(config):
            raise ValueError("source data contract differs from training config")
        identities[name] = meta["preparation_id"]
        for split in ("train", "dev", "test"):
            groups = defaultdict(list)
            path = directory / f"{split}.jsonl"
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    episode = Episode.from_record(json.loads(line))
                    source = episode.sources[0]
                    for kind, key in [
                        ("document", source.document_id),
                        ("source", source.source_id),
                        ("cluster", source.provenance["dedup_cluster"]),
                    ]:
                        previous = registry.setdefault((kind, key), split)
                        if previous != split:
                            raise ValueError("source crosses splits")
                    if len(episode.input_ids) > config.max_input_tokens:
                        rejected[f"{name}/{split}/input_length"] += 1
                        continue
                    ae, lm = read_tokens(episode, tokenizer)
                    capacities = capacity_weights(config, len(episode.input_ids), ae, lm, 0)
                    # All configured ratios must fit, including the full-context LM baseline.
                    expected = {
                        max(
                            config.pretrain_k_min,
                            min(config.k_limit, math.ceil(len(episode.input_ids) / r)),
                        )
                        for r in config.pretrain_compression_ratios
                    }
                    if set(capacities) != expected or (
                        lm
                        and 1 + len(episode.input_ids) + len(lm.prompt_ids) + len(lm.target_ids)
                        > config.read_context_tokens
                    ):
                        rejected[f"{name}/{split}/window_or_target_length"] += 1
                        continue
                    content_key = hashlib.blake2b(
                        json.dumps(
                            (
                                episode.reads[0].task,
                                source.provenance["input_text_key"],
                                " ".join(episode.reads[0].references[0].text.split()),
                            )
                        ).encode()
                    ).hexdigest()
                    if content_key in seen_content:
                        if seen_content[content_key] != split:
                            raise ValueError("duplicate content crosses splits")
                        rejected[f"{name}/{split}/duplicate_content"] += 1
                        continue
                    seen_content[content_key] = split
                    bucket = next(b for b in bounds if len(episode.input_ids) <= b)
                    groups[(episode.reads[0].task, bucket)].append((offset, episode.episode_id))
            cells[name, split] = groups
            print(
                json.dumps(
                    {"source": name, "split": split, "eligible": sum(map(len, groups.values()))}
                ),
                flush=True,
            )
    keys = [(task, b) for task in ("ae", "continuation") for b in bounds]
    quotas = {
        split: min(len(cells[name, split][key]) for name in sources for key in keys)
        for split in ("train", "dev", "test")
    }
    # Exact source proportions are required at each task/length cell.
    while quotas["train"] and any(
        not math.isclose(quotas["train"] * w, round(quotas["train"] * w))
        for weights in runs.values()
        for w in weights.values()
    ):
        quotas["train"] -= 1
    if min(quotas.values()) <= 0:
        raise ValueError("some source/task/length cells have no eligible examples")
    output.mkdir(parents=True)
    report = {
        "source_preparations": identities,
        "selection": spec,
        "quota_per_task_length": quotas,
        "rejected": dict(rejected),
        "available": {
            f"{name}/{split}": {f"{t}/{b}": len(cells[name, split][t, b]) for t, b in keys}
            for name in sources
            for split in ("train", "dev", "test")
        },
        "checks": {
            "source_split_isolation": True,
            "all_ratios_fit": True,
            "full_context_fits": True,
        },
    }

    def write_selection(destination, split, weights):
        destination.mkdir(parents=True, exist_ok=True)
        selected = []
        for name, weight in weights.items():
            for key in keys:
                pool = list(cells[name, split][key])
                random.Random(f"{spec['seed']}:{name}:{split}:{key}").shuffle(pool)
                selected.extend(
                    (name, offset, identity)
                    for offset, identity in pool[: round(quotas[split] * weight)]
                )
        random.Random(spec["seed"]).shuffle(selected)
        seen = set()
        with (destination / f"{split}.jsonl").open("wb") as out:
            handles = {name: (sources[name] / f"{split}.jsonl").open("rb") for name in weights}
            try:
                for name, offset, identity in selected:
                    if identity in seen:
                        raise ValueError("duplicate selected episode ID")
                    seen.add(identity)
                    handles[name].seek(offset)
                    out.write(handles[name].readline())
            finally:
                for handle in handles.values():
                    handle.close()
        return len(selected)

    dataset_weights = {name: {name: 1} for name in sources} | runs
    report["runs"] = {}
    for name, weights in dataset_weights.items():
        dest = output / name
        counts = {}
        if name in runs:
            counts["train"] = write_selection(dest, "train", weights)
            report["runs"][name] = {
                "data_dir": str(dest.resolve()),
                "samples": counts["train"],
            }
        if name in sources:
            counts.update(
                {split: write_selection(dest, split, weights) for split in ("dev", "test")}
            )
        metadata = {
            "preparation_id": str(uuid.uuid4()),
            "contract": data_contract(config),
            "source_preparations": identities,
            "source_weights": weights,
            "counts": counts,
        }
        (dest / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report["evaluation_dirs"] = {
        name: str((output / name).resolve()) for name in sources
    }
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, revision=config.model_revision, local_files_only=True
    )
    print(
        json.dumps(
            prepare_experiment(
                json.loads(args.spec.read_text()), config, tokenizer, args.output_dir
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
