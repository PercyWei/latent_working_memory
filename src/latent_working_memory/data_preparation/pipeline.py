from __future__ import annotations

import hashlib
import itertools
import json
import random
import uuid
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Mapping

from transformers import PreTrainedTokenizerBase

from latent_working_memory.data_preparation.audit import audit_preparation, compare_preparations
from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.data_preparation.dedup import cluster_documents
from latent_working_memory.data_preparation.fineweb import (
    data_contract,
    SemanticSpans,
    document_split,
)
from latent_working_memory.data_preparation.quality import document_rejection_reason
from latent_working_memory.data_preparation.scoring import SampleScorer
from latent_working_memory.data_preparation.truncation import RandomSpans
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.data import Episode

SPLITS = ("train", "dev", "test")
TASKS = ("ae", "continuation")


def prepare_sources(
    records: Iterable[Mapping[str, Any]],
    config: ExperimentConfig,
    output_dir: Path,
    preparation: PreparationConfig,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"use a new source pool directory: {output_dir}")
    candidates = list(itertools.islice(records, preparation.max_documents))
    reasons = [document_rejection_reason(row, preparation.min_document_chars) for row in candidates]
    clusters = cluster_documents(candidates, preparation)
    output_dir.mkdir(parents=True)
    seen = set()
    counts = Counter()
    with (output_dir / "sources.jsonl").open("w") as handle:
        for record, cluster, reason in zip(candidates, clusters, reasons, strict=True):
            status = "basic_rejected" if reason else "duplicate" if cluster in seen else "eligible"
            if status == "eligible":
                seen.add(cluster)
            split = document_split(cluster, config)
            counts[status] += 1
            counts[f"{split}/{status}"] += 1
            handle.write(
                json.dumps(
                    {
                        "record": dict(record),
                        "cluster": cluster,
                        "split": split,
                        "status": status,
                        "reason": reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    metadata = {
        "source_pool_id": str(uuid.uuid4()),
        "data_seed": config.data_seed,
        "split_fractions": list(config.split_fractions),
        "dataset": config.pretrain_dataset,
        "subset": config.pretrain_subset,
        "recipe": preparation.to_dict(),
        "statistics": dict(counts),
    }
    (output_dir / "source-pool.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


class VariantBuilder:
    """Account for post-review task/bin quotas and write one independent dataset."""

    def __init__(
        self, root, variant, tokenizer, config, preparation, scorer, source_pool, reference
    ):
        self.root, self.variant = root, variant
        self.directory = root / variant
        if self.directory.exists():
            raise FileExistsError(f"use a new variant directory: {self.directory}")
        self.tokenizer, self.config = tokenizer, config
        self.recipe, self.scorer, self.source_pool = preparation, scorer, source_pool
        self.quotas = dict(zip(SPLITS, preparation.samples_per_task, strict=True))
        self.counts = Counter()
        self.histogram = {
            split: {task: {str(b): 0 for b in preparation.length_bounds} for task in TASKS}
            for split in SPLITS
        }
        self.reference = reference
        if reference is not None and (
            reference["source_pool_id"] != source_pool["source_pool_id"]
            or reference["contract"] != data_contract(config)
            or reference["samples_per_task"] != list(preparation.samples_per_task)
            or reference["length_bounds"] != list(preparation.length_bounds)
        ):
            raise ValueError(
                "random construction must use the semantic source pool, quotas and bins"
            )
        self.targets = preparation.balanced_histogram()
        recipe_record = json.loads(json.dumps(preparation.to_dict()))
        if reference is not None and (
            reference["input_histogram"] != self.targets
            or any(
                reference["recipe"][key] != recipe_record[key]
                for key in ("min_sample_tokens", "max_sample_tokens", "lm_prefix_fraction")
            )
        ):
            raise ValueError("datasets must share length constraints and balanced input quotas")
        self.seen_samples = set()
        self.fragment_splits = {}
        if reference is not None:
            for split in SPLITS:
                with (root / "semantic" / f"{split}.jsonl").open() as handle:
                    for line in handle:
                        episode = Episode.from_record(json.loads(line))
                        key = episode.sources[0].provenance["input_text_key"]
                        if len(episode.input_ids) >= preparation.dedup_input_min_tokens:
                            self.fragment_splits[key] = split

    def remaining(self, split: str, task: str, bucket: str | None = None) -> int:
        if bucket is not None:
            return self.targets[split][task][bucket] - self.histogram[split][task][bucket]
        return self.quotas[split] - self.counts[f"{split}/{task}"]

    def consider(self, episodes: list[Episode], source: dict, handles: dict) -> None:
        record, cluster, split = source["record"], source["cluster"], source["split"]
        pending, batch_keys = [], set()
        for episode in episodes:
            task = episode.reads[0].task
            bucket = str(next(b for b in self.recipe.length_bounds if len(episode.input_ids) <= b))
            if self.remaining(split, task, bucket) <= 0:
                continue
            p = episode.sources[0].provenance
            x = " ".join(record["text"][slice(*p["x_char_span"])].split())
            y = " ".join(episode.reads[0].references[0].text.split())
            key = hashlib.blake2b(json.dumps((task, x, y)).encode()).hexdigest()
            input_key = hashlib.blake2b(x.encode()).hexdigest()
            if key in self.seen_samples or key in batch_keys:
                self.counts["duplicate_samples"] += 1
                continue
            if len(episode.input_ids) >= self.recipe.dedup_input_min_tokens and (
                input_key in self.fragment_splits and self.fragment_splits[input_key] != split
            ):
                self.counts["cross_split_fragments"] += 1
                continue
            batch_keys.add(key)
            pending.append((episode, task, bucket, key, input_key))
        reviews = self.scorer.score_batch(
            [
                {
                    "boundary_variant": self.variant,
                    "task": task,
                    "X": record["text"][slice(*episode.sources[0].provenance["x_char_span"])],
                    "Y": episode.reads[0].references[0].text if task == "continuation" else None,
                }
                for episode, task, *_ in pending
            ]
        )
        for row, review in zip(pending, reviews, strict=True):
            episode, task, bucket, key, input_key = row
            self.counts[f"review/{review['decision']}"] += 1
            status = review["decision"]
            if status == "keep":
                status = "accepted" if self.remaining(split, task, bucket) > 0 else "quota_filled"
            handles["sample-decisions"].write(
                json.dumps(
                    {
                        "episode_id": episode.episode_id,
                        "document_id": record["id"],
                        "split": split,
                        "task": task,
                        "input_length_up_to": int(bucket),
                        "status": status,
                        "review": review,
                        "x_char_span": episode.sources[0].provenance["x_char_span"],
                        "y_char_span": episode.sources[0].provenance["y_char_span"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if status != "accepted":
                continue
            if not self.counts[f"document/{record['id']}"]:
                handles["documents"].write(
                    json.dumps(
                        {
                            "record": record,
                            "cluster": cluster,
                            "split": split,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self.counts[f"{split}/documents"] += 1
            self.counts[f"document/{record['id']}"] += 1
            self.seen_samples.add(key)
            if len(episode.input_ids) >= self.recipe.dedup_input_min_tokens:
                self.fragment_splits[input_key] = split
            episode.sources[0].provenance.update(
                dedup_cluster=cluster,
                input_text_key=input_key,
                quality_review=review,
            )
            handles[split].write(json.dumps(episode.to_record(), ensure_ascii=False) + "\n")
            self.counts[f"{split}/{task}"] += 1
            self.histogram[split][task][bucket] += 1

    def run(self, sources: list[dict], topic_annotations: dict | None) -> dict:
        self.directory.mkdir()
        rng = random.Random(f"{self.config.data_seed}:{self.variant}:sources")
        rng.shuffle(sources)
        with ExitStack() as stack:
            handles = {
                name: stack.enter_context((self.directory / f"{name}.jsonl").open("w"))
                for name in (*SPLITS, "documents", "sample-decisions")
            }
            for i, source in enumerate(sources):
                split, record = source["split"], source["record"]
                if all(self.remaining(split, task) == 0 for task in TASKS):
                    continue
                annotation = (
                    topic_annotations.get(record["id"]) if topic_annotations is not None else None
                )
                sampler = (SemanticSpans if self.variant == "semantic" else RandomSpans)(
                    record, self.tokenizer, self.config, self.recipe, annotation
                )
                cell_rng = random.Random(
                    f"{self.config.data_seed}:{record['id']}:{self.variant}:cells"
                )
                task_rngs = {
                    task: random.Random(
                        f"{self.config.data_seed}:{record['id']}:{self.variant}:{task}"
                    )
                    for task in TASKS
                }
                for offset in range(
                    0, self.recipe.candidates_per_document, self.recipe.scoring_batch_size
                ):
                    cells = [
                        (task, lower, upper, self.remaining(split, task, str(upper)))
                        for task in TASKS
                        for lower, upper in self.recipe.length_intervals()
                        if self.remaining(split, task, str(upper)) > 0
                    ]
                    if not cells:
                        break
                    episodes = []
                    attempts = min(
                        self.recipe.scoring_batch_size, self.recipe.candidates_per_document - offset
                    )
                    for _ in range(attempts):
                        task, lower, upper, _ = cell_rng.choices(cells, [c[3] for c in cells])[0]
                        episode = sampler.sample(task, lower, upper, task_rngs[task])
                        if episode is not None:
                            episodes.append(episode)
                    self.consider(episodes, source, handles)
                if (i + 1) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "variant": self.variant,
                                "sources_visited": i + 1,
                                "accepted": {
                                    s: {t: self.counts[f"{s}/{t}"] for t in TASKS} for s in SPLITS
                                },
                            }
                        ),
                        flush=True,
                    )
        missing = {
            s: {t: self.remaining(s, t) for t in TASKS if self.remaining(s, t)} for s in SPLITS
        }
        missing = {s: tasks for s, tasks in missing.items() if tasks}
        if missing:
            gaps = {
                s: {
                    t: {
                        b: self.remaining(s, t, b)
                        for b in self.histogram[s][t]
                        if self.remaining(s, t, b)
                    }
                    for t in missing[s]
                }
                for s in missing
            }
            raise ValueError(
                f"{self.variant} sample quotas not reached: {missing}; bin gaps: {gaps}"
            )
        audit = audit_preparation(
            self.directory, self.tokenizer, self.config, self.recipe, self.root
        )
        metadata = {
            "preparation_id": str(uuid.uuid4()),
            "source_pool_id": self.source_pool["source_pool_id"],
            "boundary_variant": self.variant,
            "contract": data_contract(self.config),
            "recipe": self.recipe.to_dict(),
            "samples_per_task": list(self.recipe.samples_per_task),
            "length_bounds": list(self.recipe.length_bounds),
            "target_input_histogram": self.targets,
            "input_histogram": self.histogram,
            "statistics": {k: v for k, v in self.counts.items() if not k.startswith("document/")},
            "scoring_protocol": self.scorer.protocol,
            "audit": audit,
        }
        if self.reference is not None:
            metadata["reference_preparation_id"] = self.reference["preparation_id"]
        return metadata


def prepare_variant(
    root: Path,
    variant: str,
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    preparation: PreparationConfig,
    scorer: SampleScorer,
    topic_annotations: dict | None = None,
) -> dict[str, Any]:
    if variant not in {"semantic", "random"}:
        raise ValueError("variant must be semantic or random")
    pool = json.loads((root / "source-pool.json").read_text())
    if (
        pool["data_seed"] != config.data_seed
        or pool["split_fractions"] != list(config.split_fractions)
        or pool["dataset"] != config.pretrain_dataset
        or pool["subset"] != config.pretrain_subset
        or any(
            pool["recipe"][key] != preparation.to_dict()[key]
            for key in (
                "max_documents",
                "min_document_chars",
                "near_duplicate_threshold",
                "near_duplicate_min_words",
            )
        )
    ):
        raise ValueError("source pool configuration differs; reuse its original split definition")
    reference = (
        json.loads((root / "semantic/preparation.json").read_text())
        if variant == "random"
        else None
    )
    sources = [json.loads(line) for line in (root / "sources.jsonl").read_text().splitlines()]
    builder = VariantBuilder(root, variant, tokenizer, config, preparation, scorer, pool, reference)
    result = builder.run([row for row in sources if row["status"] == "eligible"], topic_annotations)
    if variant == "random":
        comparison = compare_preparations(root, tokenizer, config, result)
        (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (root / variant / "preparation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


def prepare_fineweb(
    records: Iterable[Mapping[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    output_dir: Path,
    preparation: PreparationConfig,
    scorer: SampleScorer,
    topic_annotations: dict | None = None,
) -> dict[str, Any]:
    prepare_sources(records, config, output_dir, preparation)
    semantic = prepare_variant(
        output_dir, "semantic", tokenizer, config, preparation, scorer, topic_annotations
    )
    random_data = prepare_variant(
        output_dir, "random", tokenizer, config, preparation, scorer, topic_annotations
    )
    return {"semantic": semantic, "random": random_data}
