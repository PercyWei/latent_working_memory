from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from sacrebleu.metrics import BLEU
from transformers import PreTrainedTokenizerBase

from latent_working_memory.v1.backbone import LatentMemoryBackbone, ReadTokens
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.data import EpisodeIndex
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.objectives import ReaderOutput
from latent_working_memory.v1.pretrain.sampling import capacity_weights, read_tokens
from latent_working_memory.v1.pretrain.evaluation_batching import (
    EvaluationBatching,
    ReadJob,
    GenerationJob,
    read_batches,
    generation_batches,
)


def normalize_exact_match_text(text: str) -> str:
    return " ".join(text.strip().split())


def correct_prefix_ratio(prediction: tuple[int, ...], reference: tuple[int, ...]) -> float:
    """Fraction of reference content tokens matched before the first mismatch or stop."""
    if not reference:
        raise ValueError("prefix accuracy needs a non-empty reference")
    matched = 0
    for predicted, expected in zip(prediction, reference):
        if predicted != expected:
            break
        matched += 1
    return matched / len(reference)


@torch.no_grad()
def evaluate_pretraining(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
    backbone: LatentMemoryBackbone,
    writer: JointMemoryWriter,
    index: EpisodeIndex,
    output_dir: Path,
    step: int,
    training_input_tokens: int,
    split: str = "dev",
    prefix_tokens: tuple[int, ...] = (),
    batching: EvaluationBatching | None = None,
) -> dict[str, Any]:
    if split not in {"dev", "test"}:
        raise ValueError("evaluation split must be dev or test")
    panel = [
        index[i]
        for i in index.evaluation_panel(
            config.eval_examples, config.data_seed + 1, config.input_length_bounds
        )
    ]
    if len({e.sources[0].source_id for e in panel}) < 2:
        raise ValueError(
            "evaluation needs at least two independent sources for wrong-memory controls"
        )
    if any(type(n) is not int or n <= 0 for n in prefix_tokens):
        raise ValueError("diagnostic prefix lengths must be positive integers")
    batching = EvaluationBatching() if batching is None else batching
    prefix_records = []
    was_training = backbone.training, writer.training
    backbone.eval()
    writer.eval()
    records, read_jobs, generation_jobs = [], [], []
    generated_ae_views = 0
    try:
        for episode_number, episode in enumerate(panel):
            source = episode.sources[0]
            ae, lm = read_tokens(episode, tokenizer)
            if ae is not None:
                generated_ae_views += 1
            # Evaluation enumerates every legal capacity, independently of training weights.
            capacities = capacity_weights(config, len(episode.input_ids), ae, lm, 1)
            if not capacities:
                raise ValueError(f"no legal evaluation capacity for {episode.episode_id}")
            donor = min(
                (e for e in panel if e.sources[0].source_id != source.source_id),
                key=lambda e: (abs(len(e.input_ids) - len(episode.input_ids)), e.episode_id),
            )
            # Raw-context controls are independent of K and are evaluated once per task.
            empty = writer.initialize_state().values
            raw_records = {"full_context": [], "base_full_context": []}
            generate_ae = (
                ae is not None
                and generated_ae_views <= config.eval_generation_examples
                and (
                    step % config.eval_generation_every == 0 or step in config.eval_generation_steps
                )
            )
            for task in (ae, lm):
                if task is not None:
                    for condition, use_lora in (
                        ("full_context", True),
                        ("base_full_context", False),
                    ):
                        read_jobs.append(
                            ReadJob(
                                raw_records[condition], empty, task, episode.input_ids, use_lora
                            )
                        )
            features = backbone.text_features([episode.input_ids, donor.input_ids], [0, 0])
            for capacity in sorted(capacities):
                common = {
                    "episode_id": episode.episode_id,
                    "document_id": source.document_id,
                    "boundary_method": source.provenance["boundary_method"],
                    "input_tokens": len(episode.input_ids),
                    "capacity": capacity,
                    "effective_ratio": len(episode.input_ids) / capacity,
                }
                memories = writer.update_batch(
                    [writer.initialize_state(dtype=f.dtype) for f in features],
                    features,
                    [0, 0],
                    [capacity, capacity],
                )
                conditions = {
                    "memory": memories[0].values,
                    "wrong_memory": memories[1].values,
                }
                for task_name, task in (("ae", ae), ("continuation", lm)):
                    if task is None:
                        continue
                    task_conditions = dict(conditions)
                    if task_name == "continuation":
                        task_conditions["no_memory"] = memories[0].values[:0]
                    for condition, memory in task_conditions.items():
                        record = common | {
                            "condition": condition,
                            "task": task_name,
                            "memory_bytes": memory.numel() * memory.element_size(),
                            "memory_tokens": len(memory),
                            "text_context_tokens": 0,
                            "reader_lora": True,
                            "wrong_memory_episode_id": donor.episode_id
                            if condition == "wrong_memory"
                            else None,
                        }
                        read_jobs.append(ReadJob([record], memory, task))
                        if task_name == "ae" and generate_ae:
                            record["reference"] = episode.reads[0].references[0].text
                            generation_jobs.append(
                                GenerationJob([record], memory.clone(), task.prompt_ids, task, True)
                            )
                            for prefix_length in prefix_tokens:
                                if prefix_length >= len(task.target_ids) - 1:
                                    raise ValueError(
                                        "diagnostic prefix must leave a nonempty suffix"
                                    )
                                suffix = task.target_ids[prefix_length:]
                                prompt = task.prompt_ids + task.target_ids[:prefix_length]
                                diagnostic = common | {
                                    "condition": condition,
                                    "prefix_tokens": prefix_length,
                                    "reference": tokenizer.decode(
                                        suffix[:-1], skip_special_tokens=True
                                    ),
                                    "target_tokens": len(suffix) - 1,
                                }
                                prefix_records.append(diagnostic)
                                generation_jobs.append(
                                    GenerationJob(
                                        [diagnostic],
                                        memory.clone(),
                                        prompt,
                                        ReadTokens(prompt, suffix),
                                        True,
                                    )
                                )
                        records.append(record)
                    for condition, use_lora in (
                        ("full_context", True),
                        ("base_full_context", False),
                    ):
                        record = common | {
                            "condition": condition,
                            "task": task_name,
                            "memory_bytes": 0,
                            "memory_tokens": 0,
                            "text_context_tokens": len(episode.input_ids),
                            "reader_lora": use_lora,
                            "wrong_memory_episode_id": None,
                        }
                        records.append(record)
                        raw_records[condition].append(record)
                        if task_name == "ae" and generate_ae:
                            record["reference"] = episode.reads[0].references[0].text
            if generate_ae:
                for condition, use_lora in (("full_context", True), ("base_full_context", False)):
                    generation_jobs.append(
                        GenerationJob(
                            raw_records[condition],
                            empty,
                            (*episode.input_ids, *ae.prompt_ids),
                            ae,
                            use_lora,
                        )
                    )
            # Keep GPU memories bounded while still packing requests across examples.
            if (episode_number + 1) % 16 == 0 or episode_number + 1 == len(panel):
                _score_read_jobs(backbone, read_jobs, batching)
                read_jobs.clear()
        for batch in generation_batches(generation_jobs, batching):
            predictions = backbone.greedy_students(
                [job.memory for job in batch],
                [job.prompt for job in batch],
                [job.target_length for job in batch],
                use_reader_lora=batch[0].use_lora,
            )
            for job, prediction in zip(batch, predictions, strict=True):
                content = (
                    prediction[:-1]
                    if prediction and prediction[-1] == tokenizer.eos_token_id
                    else prediction
                )
                prefix_ratio = correct_prefix_ratio(content, job.task.target_ids[:-1])
                text = tokenizer.decode(prediction, skip_special_tokens=True)
                for record in job.records:
                    record["correct_prefix_ratio"] = prefix_ratio
                    record["prediction"] = text
                    record["exact_match"] = content == job.task.target_ids[:-1]
    finally:
        backbone.train(was_training[0])
        writer.train(was_training[1])
    metrics = aggregate_pretrain_metrics(records)
    metrics.update(
        split=split,
        step=step,
        training_input_tokens=training_input_tokens,
        protocol={
            "nll": "target-token weighted; content excludes EOS; nll_with_eos includes EOS",
            "panel": "one view per independent document; every unique legal capacity",
            "control_weighting": "paired document/view/capacity; full-context reads reused across K",
            "reader_prefix": "BOS + memory or raw X + identical task prompt; targets are identical",
            "reader_lora": "enabled except base_full_context",
            "generation": "greedy; EOS stop; at most reference content length + 1 tokens",
            "correct_prefix_ratio": (
                "matched initial content tokens / reference content length; macro average"
            ),
        },
    )
    if prefix_records:
        diagnostic_groups = defaultdict(list)
        for row in prefix_records:
            diagnostic_groups[f"{row['condition']}/prefix-{row['prefix_tokens']}"].append(row)
        bleu = BLEU(tokenize="13a", lowercase=False, smooth_method="exp", effective_order=True)
        metrics["prefix_diagnostics"] = {
            key: {
                "generated_reads": len(rows),
                "correct_prefix_ratio": sum(r["correct_prefix_ratio"] for r in rows) / len(rows),
                "exact_match": sum(r["exact_match"] for r in rows) / len(rows),
                "bleu_4": bleu.corpus_score(
                    [r["prediction"] for r in rows], [[r["reference"] for r in rows]]
                ).score,
            }
            for key, rows in diagnostic_groups.items()
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    if prefix_records:
        with (output_dir / f"{split}-step-{step:06d}-prefix.jsonl").open("w") as handle:
            for row in prefix_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / f"{split}-step-{step:06d}.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / f"{split}-step-{step:06d}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    return metrics


def _score_read_jobs(backbone, jobs, batching):
    for batch in read_batches(jobs, batching):
        contexts = [job.text_context for job in batch]
        outputs = backbone.read_batch(
            [job.memory for job in batch],
            [job.task for job in batch],
            contexts if any(contexts) else None,
            batch[0].use_lora,
        )
        for job, output in zip(batch, outputs, strict=True):
            stats = _read_statistics(output, job.task)
            for record in job.records:
                record.update(stats)


def _read_statistics(output: ReaderOutput, task: ReadTokens) -> dict[str, Any]:
    return {
        "target_tokens": len(task.target_ids) - 1,
        "nll_sum": float(output.token_nll[:-1].sum()),
        "eos_nll": float(output.token_nll[-1]),
    }


def aggregate_pretrain_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for record in records:
        task_condition = f"{record['task']}/{record['condition']}"
        length_bin = 2 ** math.ceil(math.log2(record["input_tokens"]))
        ratio_bin = 2 ** math.ceil(math.log2(max(record["effective_ratio"], 1)))
        for group in (
            f"all/{task_condition}",
            f"length_ratio/{length_bin}/{ratio_bin}/{task_condition}",
        ):
            groups[group].append(record)
    summaries = {}
    # Effective order lets sentence-only groups with fewer than four words be scored.
    # corpus_score pools n-gram counts; averaging sentence BLEU would change the metric.
    bleu = BLEU(tokenize="13a", lowercase=False, smooth_method="exp", effective_order=True)
    bleu_signature = None
    for group, values in sorted(groups.items()):
        tokens = sum(r["target_tokens"] for r in values)
        total = sum(r["nll_sum"] for r in values)
        mean = total / tokens
        summary = {
            "reads": len(values),
            "target_tokens": tokens,
            "nll": mean,
            "ppl": math.exp(mean),
            "nll_with_eos": (total + sum(r["eos_nll"] for r in values)) / (tokens + len(values)),
        }
        generation = [r for r in values if "prediction" in r]
        if generation:
            summary["generated_reads"] = len(generation)
            for metric in ("correct_prefix_ratio",):
                summary[metric] = sum(r[metric] for r in generation) / len(generation)
            if all("exact_match" in r for r in generation):
                summary["exact_match"] = sum(r["exact_match"] for r in generation) / len(generation)
            summary["bleu_4"] = bleu.corpus_score(
                [r["prediction"] for r in generation], [[r["reference"] for r in generation]]
            ).score
            bleu_signature = str(bleu.get_signature())
        summaries[group] = summary
    comparisons = {}
    for group, summary in summaries.items():
        if not group.endswith("/memory"):
            continue
        prefix = group.removesuffix("/memory")
        comparison = {}
        for control in ("no_memory", "wrong_memory"):
            control_group = f"{prefix}/{control}"
            if control_group in summaries:
                comparison[f"gain_vs_{control}"] = summaries[control_group]["nll"] - summary["nll"]
        for control in ("full_context", "base_full_context"):
            control_group = f"{prefix}/{control}"
            if control_group in summaries:
                gap = summary["nll"] - summaries[control_group]["nll"]
                comparison[f"nll_gap_to_{control}"] = gap
                comparison[f"ppl_ratio_to_{control}"] = math.exp(gap)
        comparisons[prefix] = comparison
    return {
        "episodes": len({r["episode_id"] for r in records}),
        "documents": len({r["document_id"] for r in records}),
        "groups": summaries,
        "comparisons": comparisons,
        "bleu": {"scale": "0-100", "aggregation": "corpus", "signature": bleu_signature},
    }
