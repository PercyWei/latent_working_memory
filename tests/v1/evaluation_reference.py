"""Original evaluation scheduling, used only as a paired regression reference."""

import json
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
from latent_working_memory.v1.pretrain.sampling import capacity_weights, read_tokens
from latent_working_memory.v1.pretrain.evaluation import (
    _read_statistics,
    aggregate_pretrain_metrics,
    correct_prefix_ratio,
)


@torch.no_grad()
def evaluate_pretraining_reference(
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
    prefix_records = []
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
            raw_statistics = {}
            raw_generation_records = {"full_context": [], "base_full_context": []}
            generate_ae = (
                ae is not None
                and generated_ae_views <= config.eval_generation_examples
                and (
                    step % config.eval_generation_every == 0 or step in config.eval_generation_steps
                )
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
                                    (
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
                        record["exact_match"] = content == task.target_ids[:-1]
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
