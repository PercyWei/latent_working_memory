from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from sacrebleu.metrics import BLEU
from transformers import PreTrainedTokenizerBase

from latent_working_memory.v1.backbone import LatentMemoryBackbone, ReadTokens
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.data import EpisodeIndex
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.objectives import ReaderOutput
from latent_working_memory.v1.sampling import capacity_weights, read_tokens
from latent_working_memory.v1.state import MemoryState


@dataclass(frozen=True, slots=True)
class NllSummary:
    total_nll: float
    target_tokens: int

    def __post_init__(self) -> None:
        if self.total_nll < 0:
            raise ValueError("total_nll must be non-negative")
        if type(self.target_tokens) is not int or self.target_tokens <= 0:
            raise ValueError("target_tokens must be a positive integer")

    @property
    def mean_nll(self) -> float:
        return self.total_nll / self.target_tokens

    @property
    def perplexity(self) -> float:
        return math.exp(self.mean_nll)


@dataclass(frozen=True, slots=True)
class MemoryFootprintPoint:
    prefix_end: int
    persistent_bytes: int

    def __post_init__(self) -> None:
        if type(self.prefix_end) is not int or self.prefix_end < 0:
            raise ValueError("prefix_end must be a non-negative integer")
        if type(self.persistent_bytes) is not int or self.persistent_bytes < 0:
            raise ValueError("persistent_bytes must be a non-negative integer")


def normalize_exact_match_text(text: str) -> str:
    return " ".join(text.strip().split())


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_exact_match_text(prediction) == normalize_exact_match_text(reference)


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


def aggregate_nll(items: Iterable[NllSummary]) -> NllSummary:
    summaries = tuple(items)
    if not summaries:
        raise ValueError("at least one NLL summary is required")
    return NllSummary(
        total_nll=sum(summary.total_nll for summary in summaries),
        target_tokens=sum(summary.target_tokens for summary in summaries),
    )


def persistent_memory_bytes(state: MemoryState, metadata_bytes: int = 0) -> int:
    if type(metadata_bytes) is not int or metadata_bytes < 0:
        raise ValueError("metadata_bytes must be a non-negative integer")
    return state.values.numel() * state.values.element_size() + metadata_bytes


def byte_token_area(points: Iterable[MemoryFootprintPoint], stream_end: int) -> int:
    samples = tuple(points)
    if not samples:
        raise ValueError("at least one footprint point is required")
    if type(stream_end) is not int or stream_end < 0:
        raise ValueError("stream_end must be a non-negative integer")
    if samples[0].prefix_end != 0:
        raise ValueError("the first footprint point must start at prefix 0")
    if samples[-1].prefix_end > stream_end:
        raise ValueError("footprint points must not extend past stream_end")
    if any(first.prefix_end >= second.prefix_end for first, second in zip(samples, samples[1:])):
        raise ValueError("footprint points must have strictly increasing prefixes")

    area = 0
    for index, point in enumerate(samples):
        next_prefix = samples[index + 1].prefix_end if index + 1 < len(samples) else stream_end
        area += point.persistent_bytes * (next_prefix - point.prefix_end)
    return area


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
    was_training = backbone.training, writer.training
    backbone.eval()
    writer.eval()
    records, generation_jobs = [], []
    generated_ae_views = 0
    try:
        for episode in panel:
            source = episode.sources[0]
            ae, lm = read_tokens(episode, tokenizer)
            if ae is not None:
                generated_ae_views += 1
            capacities = capacity_weights(config, len(episode.input_ids), ae, lm, step)
            if not capacities:
                raise ValueError(f"no legal evaluation capacity for {episode.episode_id}")
            donor = min(
                (e for e in panel if e.sources[0].source_id != source.source_id),
                key=lambda e: (abs(len(e.input_ids) - len(episode.input_ids)), e.episode_id),
            )
            # Raw-context controls are independent of K and are evaluated once per task.
            empty = writer.initialize_state().values
            raw_statistics = {}
            raw_generation_records = {"full_context": [], "base_full_context": []}
            generate_ae = (
                ae is not None
                and generated_ae_views <= config.eval_generation_examples
                and step % config.eval_generation_every == 0
            )
            for task_name, task in (("ae", ae), ("continuation", lm)):
                if task is not None:
                    raw_statistics[task_name] = {
                        condition: _read_statistics(
                            backbone.read_batch(
                                [empty], [task], [episode.input_ids], use_reader_lora=use_lora
                            )[0],
                            task,
                        )
                        for condition, use_lora in (
                            ("full_context", True),
                            ("base_full_context", False),
                        )
                    }
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
                    outputs = backbone.read_batch(
                        list(task_conditions.values()), [task] * len(task_conditions)
                    )
                    for (condition, memory), output in zip(
                        task_conditions.items(), outputs, strict=True
                    ):
                        record = (
                            common
                            | _read_statistics(output, task)
                            | {
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
                        )
                        if task_name == "ae" and generate_ae:
                            record["reference"] = episode.reads[0].references[0].text
                            generation_jobs.append(
                                ([record], memory.clone(), task.prompt_ids, task, True)
                            )
                        records.append(record)
                    for condition, use_lora in (
                        ("full_context", True),
                        ("base_full_context", False),
                    ):
                        record = (
                            common
                            | raw_statistics[task_name][condition]
                            | {
                                "condition": condition,
                                "task": task_name,
                                "memory_bytes": 0,
                                "memory_tokens": 0,
                                "text_context_tokens": len(episode.input_ids),
                                "reader_lora": use_lora,
                                "wrong_memory_episode_id": None,
                            }
                        )
                        records.append(record)
                        if task_name == "ae" and generate_ae:
                            record["reference"] = episode.reads[0].references[0].text
                            raw_generation_records[condition].append(record)
            if generate_ae:
                for condition, use_lora in (("full_context", True), ("base_full_context", False)):
                    generation_jobs.append(
                        (
                            raw_generation_records[condition],
                            empty,
                            (*episode.input_ids, *ae.prompt_ids),
                            ae,
                            use_lora,
                        )
                    )
        for use_lora in (True, False):
            jobs = sorted(
                (job for job in generation_jobs if job[4] == use_lora),
                key=lambda job: len(job[1]) + len(job[2]) + len(job[3].target_ids),
            )
            for start in range(0, len(jobs), 8):
                batch = jobs[start : start + 8]
                predictions = backbone.greedy_students(
                    [job[1] for job in batch],
                    [job[2] for job in batch],
                    [len(job[3].target_ids) for job in batch],
                    use_reader_lora=use_lora,
                )
                for (paired_records, _, _, task, _), prediction in zip(
                    batch, predictions, strict=True
                ):
                    content = (
                        prediction[:-1]
                        if prediction and prediction[-1] == tokenizer.eos_token_id
                        else prediction
                    )
                    prefix_ratio = correct_prefix_ratio(content, task.target_ids[:-1])
                    text = tokenizer.decode(prediction, skip_special_tokens=True)
                    for record in paired_records:
                        record["correct_prefix_ratio"] = prefix_ratio
                        record["prediction"] = text
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
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{split}-step-{step:06d}.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / f"{split}-step-{step:06d}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    return metrics


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
