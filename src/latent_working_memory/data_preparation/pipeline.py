from __future__ import annotations

import hashlib
import itertools
import json
import random
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
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
from latent_working_memory.data_preparation.resume import FILES, load_progress, save_progress
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
    """Account for task/bin quotas and write one independent dataset."""

    def __init__(
        self,
        root,
        variant,
        tokenizer,
        config,
        preparation,
        source_pool,
        reference,
        resume=False,
    ):
        self.root, self.variant = root, variant
        self.directory = root / variant
        self.resume = resume
        if (self.directory / "preparation.json").exists():
            raise FileExistsError(f"variant is already complete: {self.directory}")
        if self.directory.exists() and not resume:
            raise FileExistsError(f"use a new variant directory: {self.directory}")
        self.tokenizer, self.config = tokenizer, config
        self.recipe, self.source_pool = preparation, source_pool
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
        self.considered_episodes = set()
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

    def consider(self, candidates: list[tuple[Episode, dict]], handles: dict) -> None:
        pending, batch_keys, input_splits = [], set(), {}
        for episode, source in candidates:
            record, cluster, split = source["record"], source["cluster"], source["split"]
            if episode.episode_id in self.considered_episodes:
                continue
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
            if len(episode.input_ids) >= self.recipe.dedup_input_min_tokens:
                owner = input_splits.get(input_key, self.fragment_splits.get(input_key))
                if owner is not None and owner != split:
                    self.counts["cross_split_fragments"] += 1
                    continue
                input_splits[input_key] = split
            batch_keys.add(key)
            pending.append((episode, task, bucket, key, input_key, source))
        for episode, task, bucket, key, input_key, source in pending:
            record, cluster, split = source["record"], source["cluster"], source["split"]
            self.considered_episodes.add(episode.episode_id)
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
            )
            handles[split].write(json.dumps(episode.to_record(), ensure_ascii=False) + "\n")
            self.counts[f"{split}/{task}"] += 1
            self.histogram[split][task][bucket] += 1

    def progress_contract(self, topic_annotations):
        recipe = self.recipe.to_dict()
        return json.loads(
            json.dumps(
                {
                    "source_pool_id": self.source_pool["source_pool_id"],
                    "variant": self.variant,
                    "data": data_contract(self.config),
                    "recipe": recipe,
                    "sentence_boundaries": "pysbd_conservative",
                    "topic_annotations": topic_annotations,
                }
            )
        )

    def restore_rows(self):
        documents = {}
        with (self.directory / "documents.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                documents[row["record"]["id"]] = row["record"]
        for split in SPLITS:
            with (self.directory / f"{split}.jsonl").open() as handle:
                for line in handle:
                    episode = Episode.from_record(json.loads(line))
                    source = episode.sources[0]
                    record = documents[source.document_id]
                    p = source.provenance
                    x = " ".join(record["text"][slice(*p["x_char_span"])].split())
                    y = " ".join(episode.reads[0].references[0].text.split())
                    task = episode.reads[0].task
                    key = hashlib.blake2b(json.dumps((task, x, y)).encode()).hexdigest()
                    input_key = hashlib.blake2b(x.encode()).hexdigest()
                    if key in self.seen_samples:
                        raise ValueError("duplicate accepted sample in paused data")
                    self.seen_samples.add(key)
                    if len(episode.input_ids) >= self.recipe.dedup_input_min_tokens:
                        if (
                            input_key in self.fragment_splits
                            and self.fragment_splits[input_key] != split
                        ):
                            raise ValueError("cross-split fragment in paused data")
                        self.fragment_splits[input_key] = split
                    bucket = str(
                        next(b for b in self.recipe.length_bounds if len(episode.input_ids) <= b)
                    )
                    self.histogram[split][task][bucket] += 1
                    self.counts[f"{split}/{task}"] += 1
                    if not self.counts[f"document/{record['id']}"]:
                        self.counts[f"{split}/documents"] += 1
                    self.counts[f"document/{record['id']}"] += 1
        accepted = 0
        with (self.directory / "sample-decisions.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                self.considered_episodes.add(row["episode_id"])
                accepted += row["status"] == "accepted"
        if accepted != sum(self.counts[f"{s}/{t}"] for s in SPLITS for t in TASKS):
            raise ValueError("paused decisions and accepted rows disagree")
        if any(
            self.remaining(s, t, str(b)) < 0
            for s in SPLITS
            for t in TASKS
            for b in self.recipe.length_bounds
        ):
            raise ValueError("paused rows exceed target quotas")

    def build_samplers(self, sources, topic_annotations):
        samplers = []
        for source in sources:
            record = source["record"]
            annotation = (
                topic_annotations.get(record["id"]) if topic_annotations is not None else None
            )
            sampler = (
                SemanticSpans(record, self.tokenizer, self.config, self.recipe, annotation)
                if self.variant == "semantic"
                else RandomSpans(record, self.tokenizer, self.config, self.recipe)
            )
            samplers.append((source, sampler))
        return samplers

    def prepare_window(self, samplers, remaining):
        candidates = []
        for source, sampler in samplers:
            split, record = source["split"], source["record"]
            cells = [
                (task, lower, upper, remaining[split][task][str(upper)])
                for task in TASKS
                for lower, upper in self.recipe.length_intervals()
                if remaining[split][task][str(upper)] > 0 and sampler.available(task, lower, upper)
            ]
            if not cells:
                continue
            rng = random.Random(f"{self.config.data_seed}:{record['id']}:{self.variant}:cells")
            task_rngs = {
                t: random.Random(f"{self.config.data_seed}:{record['id']}:{self.variant}:{t}")
                for t in TASKS
            }
            for _ in range(self.recipe.candidates_per_document):
                task, lower, upper, _ = rng.choices(cells, [c[3] for c in cells])[0]
                episode = sampler.sample(task, lower, upper, task_rngs[task])
                if episode is not None:
                    candidates.append((episode, source))
        return candidates

    def run(self, sources: list[dict], topic_annotations: dict | None) -> dict:
        rng = random.Random(f"{self.config.data_seed}:{self.variant}:sources")
        rng.shuffle(sources)
        contract = self.progress_contract(topic_annotations)
        start = 0
        if self.resume:
            state = load_progress(self.directory, contract)
            self.restore_rows()
            self.counts = Counter(state["counts"])
            start = state["next_source"]
        else:
            self.directory.mkdir()
        with ExitStack() as stack:
            handles = {
                name: stack.enter_context((self.directory / f"{name}.jsonl").open("a"))
                for name in FILES
            }
            save_progress(self.directory, handles, contract, start, dict(self.counts))
            cpu = stack.enter_context(ThreadPoolExecutor(max_workers=1))
            width = self.recipe.candidate_window_documents
            future = cpu.submit(
                self.build_samplers, sources[start : start + width], topic_annotations
            )
            for i in range(start, len(sources), width):
                # Fixed windows preserve proposal and commit order during CPU prefetch.
                remaining = {
                    s: {
                        t: {str(b): self.remaining(s, t, str(b)) for b in self.recipe.length_bounds}
                        for t in TASKS
                    }
                    for s in SPLITS
                }
                if not any(
                    v
                    for tasks in remaining.values()
                    for bins in tasks.values()
                    for v in bins.values()
                ):
                    break
                began = perf_counter()
                samplers = future.result()
                future = cpu.submit(
                    self.build_samplers, sources[i + width : i + 2 * width], topic_annotations
                )
                candidates = self.prepare_window(samplers, remaining)
                prepared = perf_counter() - began
                self.consider(candidates, handles)
                next_source = min(i + width, len(sources))
                save_progress(self.directory, handles, contract, next_source, dict(self.counts))
                print(
                    json.dumps(
                        {
                            "variant": self.variant,
                            "sources_visited": next_source,
                            "candidate_count": len(candidates),
                            "preparation_seconds": prepared,
                            "window_seconds": perf_counter() - began,
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
    topic_annotations: dict | None = None,
    resume: bool = False,
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
    builder = VariantBuilder(root, variant, tokenizer, config, preparation, pool, reference, resume)
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
    topic_annotations: dict | None = None,
) -> dict[str, Any]:
    prepare_sources(records, config, output_dir, preparation)
    semantic = prepare_variant(
        output_dir, "semantic", tokenizer, config, preparation, topic_annotations
    )
    random_data = prepare_variant(
        output_dir, "random", tokenizer, config, preparation, topic_annotations
    )
    return {"semantic": semantic, "random": random_data}
