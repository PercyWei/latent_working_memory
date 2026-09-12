from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
import json
from importlib.metadata import version
import math
import os
import time
from pathlib import Path
import random
import re
import string
import subprocess
import sys
import uuid

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from latent_working_memory.devices import validate_device
from latent_working_memory.v1.backbone import ReadTokens, load_backbone
from latent_working_memory.v1.checkpoint import (
    capture_rng_state,
    load_model_checkpoint,
    restore_rng_state,
    save_model_checkpoint,
)
from latent_working_memory.v1.distributed import synchronize_gradients
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.v1.squad import SquadDataset, sample_reads
from latent_working_memory.v1.dynamic_data import (
    DynamicConfig,
    DynamicTextSampler,
    write_boundaries,
)
from latent_working_memory.v1.tracking import swanlab_run
from latent_working_memory.v1.training import (
    load_trainable_model_state,
    precision_context,
    trainable_model_state,
)


def read_schedule(episode, tokenizer, recipe, model_config, seed, capacity, generation=False):
    if capacity > model_config.k_limit:
        raise ValueError("capacity exceeds model slot limit")
    previous = 0
    rng, visits, schedule = random.Random(seed), {}, {}
    boundaries = write_boundaries(episode, capacity)
    for end in boundaries:
        if end - previous + 1 > model_config.write_context_tokens:
            raise ValueError("initial compression or paragraph exceeds write context budget")
        previous = end
        jobs = []
        if end == boundaries[0]:
            schedule[end] = jobs
            continue
        for read in sample_reads(
            episode, end, recipe.new_count, recipe.history_count, rng, visits, recipe.max_visits
        ):
            tokens = ReadTokens(
                tuple(tokenizer.encode(read.prompt, add_special_tokens=False)),
                tuple(tokenizer.encode(read.references[0].text, add_special_tokens=False))
                + (tokenizer.eos_token_id,),
            )
            target_budget = (
                max(len(tokens.target_ids), recipe.generation_tokens)
                if generation
                else len(tokens.target_ids)
            )
            if (
                1 + capacity + len(tokens.prompt_ids) + target_budget
                > model_config.read_context_tokens
            ):
                raise ValueError(f"QA read exceeds context budget: {read.read_id}")
            jobs.append((read, tokens))
        schedule[end] = jobs
    if not any(schedule.values()):
        raise ValueError("episode has no selected reads")
    return schedule


class DynamicTrainer:
    def __init__(self, backbone, writer, model_config, recipe, device):
        self.backbone, self.writer = backbone, writer
        self.model_config, self.recipe, self.device = model_config, recipe, device
        self.parameters = list(backbone.trainable_parameters()) + list(writer.parameters())
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=recipe.learning_rate, weight_decay=recipe.weight_decay
        )

    def step(self, episodes, tokenizer, seeds, capacity):
        batch_size = len(episodes)
        if len(seeds) != batch_size or batch_size != self.recipe.batch_size:
            raise ValueError("one optimizer step requires batch_size samples and seeds")
        if capacity not in self.recipe.capacities:
            raise ValueError("capacity is not configured for dynamic training")
        self.backbone.train()
        self.writer.train()
        self.optimizer.zero_grad(set_to_none=True)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        articles = [
            self._backward_episode(episodes[i], tokenizer, seeds[i], batch_size, capacity)
            for i in range(rank, batch_size, world_size)
        ]
        if world_size > 1:
            synchronize_gradients(self.parameters)
            gathered = [None] * world_size
            dist.all_gather_object(gathered, articles)
            articles = [gathered[i % world_size][i // world_size] for i in range(batch_size)]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.recipe.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        totals = {
            key: sum(article[key] for article in articles)
            for key in (
                "target_tokens",
                "input_tokens",
                "reads",
                "writes",
                "updates",
                "truncations",
            )
        }
        return {
            "samples": batch_size,
            "capacity": capacity,
            "sample_metrics": articles,
            "loss": sum(a["loss"] for a in articles) / batch_size,
            "target_nll": sum(a["target_nll"] * a["target_tokens"] for a in articles)
            / totals["target_tokens"],
            **totals,
            "gradient_norm": float(grad_norm),
        }

    def _read_loss(self, memory, tokens):
        return self.backbone.read_batch([memory], [tokens])[0].mean_nll

    def _backward_episode(self, episode, tokenizer, seed, batch_size, capacity):
        schedule = read_schedule(episode, tokenizer, self.recipe, self.model_config, seed, capacity)
        count = sum(len(jobs) for jobs in schedule.values())
        state = None
        previous = segment_start = 0
        pending, loss_value, token_nll, target_tokens = [], 0.0, 0.0, 0
        truncations = segment_updates = 0
        segments = []
        for end in schedule:
            with precision_context(self.device):
                features = self.backbone.text_features(
                    [episode.input_ids[previous:end]], [previous]
                )[0]
                if state is None:
                    state = self.writer(
                        self.writer.initialize_state(features.dtype),
                        features,
                        first_slots=capacity,
                    )
                else:
                    state = self.writer(state, features)
                for _, tokens in schedule[end]:
                    mean_nll = (
                        activation_checkpoint(
                            self._read_loss, state.values, tokens, use_reentrant=False
                        )
                        if self.recipe.gradient_checkpointing
                        else self._read_loss(state.values, tokens)
                    )
                    pending.append(mean_nll / count / batch_size)
                    loss_value += float(mean_nll.detach()) / count
                    token_nll += float(mean_nll.detach()) * len(tokens.target_ids)
                    target_tokens += len(tokens.target_ids)
            segment_updates += 1
            span = end - segment_start if self.recipe.bptt_unit == "tokens" else segment_updates
            boundary = self.recipe.bptt_span and span >= self.recipe.bptt_span
            if not segments and not pending:
                boundary = False
            if boundary or end == episode.write_ends[-1]:
                if pending:
                    torch.stack(pending).sum().backward()
                    pending.clear()
                state = state.detached()
                segments.append({"tokens": end - segment_start, "updates": segment_updates})
                segment_updates = 0
                segment_start = end
                truncations += int(end != episode.write_ends[-1])
            previous = end
        return {
            "episode_id": episode.episode_id,
            "loss": loss_value,
            "target_nll": token_nll / target_tokens,
            "target_tokens": target_tokens,
            "input_tokens": len(episode.input_ids),
            "reads": count,
            "writes": len(schedule),
            "updates": len(schedule) - 1,
            "initial_tokens": next(iter(schedule)),
            "truncations": truncations,
            "bptt_segments": segments,
        }


def answer_scores(prediction, references):
    def normalize(text):
        text = "".join(c for c in text.lower() if c not in string.punctuation)
        return re.sub(r"\s+", " ", re.sub(r"\b(a|an|the)\b", " ", text)).strip()

    pred = normalize(prediction)
    em, f1 = 0.0, 0.0
    for reference in references:
        gold = normalize(reference)
        em = max(em, float(pred == gold))
        p, g = pred.split(), gold.split()
        common = sum((Counter(p) & Counter(g)).values())
        score = 2 * common / (len(p) + len(g)) if common else 0.0
        f1 = max(f1, score)
    return em, f1


def encode_episode(backbone, writer, episode, capacity):
    state, previous = None, 0
    for end in write_boundaries(episode, capacity):
        f = backbone.text_features([episode.input_ids[previous:end]], [previous])[0]
        state = (
            writer(writer.initialize_state(f.dtype), f, first_slots=capacity)
            if state is None
            else writer(state, f)
        )
        previous = end
    return state


@torch.no_grad()
def evaluate_qa(
    backbone,
    writer,
    tokenizer,
    model_config,
    recipe,
    episodes,
    device,
    capacity,
    conditions=("memory", "no_memory", "wrong_memory", "gold_paragraph", "gold_paragraph_base"),
):
    if not episodes or not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("non-empty episodes and unique conditions required")
    if set(conditions) - {
        "memory",
        "no_memory",
        "wrong_memory",
        "gold_paragraph",
        "gold_paragraph_base",
    }:
        raise ValueError("unsupported QA condition")
    if "wrong_memory" in conditions and len({e.sources[0].document_id for e in episodes}) < 2:
        raise ValueError("wrong-memory comparison requires two independent documents")
    backbone.eval()
    writer.eval()
    records = []
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    for index in range(rank, len(episodes), world_size):
        episode = episodes[index]
        schedule = read_schedule(
            episode,
            tokenizer,
            recipe,
            model_config,
            f"{recipe.seed}:eval:{episode.episode_id}",
            capacity,
            generation=True,
        )
        with precision_context(device):
            wrong = None
            if "wrong_memory" in conditions:
                donor = next(
                    e
                    for e in episodes[index + 1 :] + episodes[: index + 1]
                    if e.sources[0].document_id != episode.sources[0].document_id
                )
                # Validate the donor's write budget too; donor QA targets are not used.
                if any(
                    b - a + 1 > model_config.write_context_tokens
                    for a, b in zip(
                        (0, *write_boundaries(donor, capacity)[:-1]),
                        write_boundaries(donor, capacity),
                    )
                ):
                    raise ValueError("donor paragraph exceeds write context budget")
                wrong = encode_episode(backbone, writer, donor, capacity).values
            state, previous = None, 0
            boundaries = tuple(schedule)
            for end in boundaries:
                f = backbone.text_features([episode.input_ids[previous:end]], [previous])[0]
                state = (
                    writer(writer.initialize_state(f.dtype), f, first_slots=capacity)
                    if state is None
                    else writer(state, f)
                )
                previous = end
                for read, tokens in schedule[end]:
                    for condition in conditions:
                        memory = state.values[:0]
                        condition_tokens = tokens
                        use_lora = condition != "gold_paragraph_base"
                        if condition == "memory":
                            memory = state.values
                        elif condition == "wrong_memory":
                            memory = wrong
                        elif condition in {"gold_paragraph", "gold_paragraph_base"}:
                            evidence_span = read.references[0].evidence_spans[0]
                            source = next(
                                source
                                for source in episode.sources
                                if (source.token_start, source.token_end) == evidence_span
                            )
                            prompt = (
                                "Text:\n"
                                + source.provenance["context"]
                                + "\n\n"
                                + read.prompt.replace(
                                    "information stored in memory", "provided text", 1
                                )
                            )
                            condition_tokens = ReadTokens(
                                tuple(tokenizer.encode(prompt, add_special_tokens=False)),
                                tokens.target_ids,
                            )
                        if (
                            1
                            + len(memory)
                            + len(condition_tokens.prompt_ids)
                            + max(len(condition_tokens.target_ids), recipe.generation_tokens)
                            > model_config.read_context_tokens
                        ):
                            raise ValueError(
                                f"{condition} QA read exceeds context budget: {read.read_id}"
                            )
                        result = backbone.read_batch(
                            [memory], [condition_tokens], use_reader_lora=use_lora
                        )[0]
                        generated = backbone.greedy_students(
                            [memory],
                            [condition_tokens.prompt_ids],
                            [recipe.generation_tokens],
                            use_reader_lora=use_lora,
                        )[0]
                        prediction = tokenizer.decode(generated, skip_special_tokens=True)
                        em, f1 = answer_scores(prediction, [r.text for r in read.references])
                        evidence_end = max(
                            b for ref in read.references for _, b in ref.evidence_spans
                        )
                        records.append(
                            {
                                "episode_id": episode.episode_id,
                                "read_id": read.read_id,
                                "prefix_end": end,
                                "condition": condition,
                                "delay_tokens": end - evidence_end,
                                "delay_writes": boundaries.index(end)
                                - bisect_left(boundaries, evidence_end),
                                "capacity": capacity,
                                "input_tokens": len(episode.input_ids),
                                "compression_ratio": end / capacity,
                                "final_compression_ratio": len(episode.input_ids) / capacity,
                                "kind": "arrival" if end == evidence_end else "delayed",
                                "prediction": prediction,
                                "references": [r.text for r in read.references],
                                "em": em,
                                "f1": f1,
                                "nll_sum": float(result.token_nll.sum()),
                                "target_tokens": result.target_length,
                                "hit_limit": len(generated) == recipe.generation_tokens
                                and generated[-1] != tokenizer.eos_token_id,
                            }
                        )
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, records)
        records = [row for rank_rows in gathered for row in rank_rows]
    records = sorted(
        records,
        key=lambda row: (
            row["episode_id"],
            row["prefix_end"],
            row["read_id"],
            row["condition"],
        ),
    )
    metrics = {}
    for condition in conditions:
        for kind in ("all", "arrival", "delayed"):
            rows = [
                r
                for r in records
                if r["condition"] == condition and (kind == "all" or r["kind"] == kind)
            ]
            if rows:
                metrics[f"{condition}/{kind}"] = qa_summary(rows)
    return metrics, records


def load_components(checkpoint, device):
    config = checkpoint.config
    tokenizer, backbone = load_backbone(
        config, device, torch.bfloat16 if device.type == "cuda" else torch.float32
    )
    writer = JointMemoryWriter(
        config.d_mem, config.num_layers, config.num_heads, config.ffn_dim, config.k_limit
    ).to(device)
    value = GrowthValueNetwork(config.d_mem).to(device)
    load_trainable_model_state(checkpoint.model_state, backbone, writer, value)
    value.requires_grad_(False)
    if (
        max(config.write_context_tokens, config.read_context_tokens)
        > backbone.max_position_embeddings
    ):
        raise ValueError("configured windows exceed backbone context")
    return tokenizer, backbone, writer, value


def evaluate_panel(backbone, writer, tokenizer, model_config, recipe, data, panel, device):
    metrics, records = {}, []
    for capacity, texts in panel.items():
        episodes = [text.episode(data) for text in texts]
        groups, rows = evaluate_qa(
            backbone, writer, tokenizer, model_config, recipe, episodes, device, capacity
        )
        metadata = {episode.episode_id: text for episode, text in zip(episodes, texts, strict=True)}
        metrics.update({f"k{capacity}/{key}": values for key, values in groups.items()})
        for row in rows:
            text = metadata[row["episode_id"]]
            row["target_ratio"] = text.ratio
            row["document_id"] = text.document_id
            row["paragraph_start"] = text.paragraph_start
            row["paragraph_end"] = text.paragraph_end
        buckets = {}
        for row in rows:
            labels = (
                f"target-r{row['target_ratio']}",
                f"length-le{2 ** (row['input_tokens'] - 1).bit_length()}",
                f"final-r-le{math.ceil(row['final_compression_ratio'])}",
            )
            for label in labels:
                for kind in ("all", row["kind"]):
                    key = f"k{capacity}/{label}/{row['condition']}/{kind}"
                    buckets.setdefault(key, []).append(row)
        metrics.update({key: qa_summary(values) for key, values in buckets.items()})
        records.extend(rows)
    return metrics, records


def qa_summary(rows):
    return {
        "reads": len(rows),
        "em": sum(r["em"] for r in rows) / len(rows),
        "f1": sum(r["f1"] for r in rows) / len(rows),
        "nll": sum(r["nll_sum"] for r in rows) / sum(r["target_tokens"] for r in rows),
        "hit_limit_rate": sum(r["hit_limit"] for r in rows) / len(rows),
    }


def write_evaluation(output_dir, name, metrics, rows):
    (output_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def runtime_info():
    source = Path(__file__).resolve()
    repository = source.parents[3]
    return {
        "python_executable": sys.executable,
        "environment": sys.prefix,
        "python_version": sys.version.split()[0],
        "source_file": str(source),
        "repository": str(repository),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "packages": {
            name: version(name) for name in ("torch", "transformers", "peft", "swanlab", "pyarrow")
        },
    }


def run_dynamic(
    checkpoint_path,
    index_path,
    output_dir,
    recipe,
    device,
    steps=None,
    save_every=100,
    eval_every=100,
    eval_texts_per_ratio=2,
    resume=False,
    swanlab_mode="disabled",
    swanlab_group=None,
    epochs=3,
):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    primary = rank == 0
    for name, count in (
        ("epochs", epochs),
        ("save_every", save_every),
        ("eval_every", eval_every),
        ("eval_texts_per_ratio", eval_texts_per_ratio),
    ):
        if type(count) is not int or count <= 0:
            raise ValueError(f"{name} must be a positive integer")
    total_steps = epochs * recipe.micro_epochs_per_epoch * recipe.steps_per_micro_epoch
    if steps is not None and (type(steps) is not int or not 0 < steps <= total_steps):
        raise ValueError("steps must be positive and within the configured epoch schedule")
    stop_step = total_steps if steps is None else steps
    if output_dir.exists() and not resume:
        raise FileExistsError("use a new output directory")
    if resume and output_dir.resolve() != checkpoint_path.resolve().parent:
        raise ValueError("resume requires the original run directory")
    checkpoint = load_model_checkpoint(checkpoint_path)
    if checkpoint.phase != ("dynamic" if resume else "pretrain"):
        raise ValueError("initialization needs pretrain; resume needs dynamic checkpoint")
    checkpoint = replace(
        checkpoint, config=replace(checkpoint.config, gradient_checkpointing=False)
    )
    if max(recipe.capacities) > checkpoint.config.k_limit:
        raise ValueError("capacity exceeds model slot limit")
    data = SquadDataset(index_path)
    sampler = DynamicTextSampler(data, recipe, checkpoint.config.write_context_tokens)
    dev = sampler.evaluation_texts("dev", eval_texts_per_ratio)
    test = sampler.evaluation_texts("test", eval_texts_per_ratio)
    identity = {
        "recipe": asdict(recipe),
        "data_index": data.index,
        "epochs": epochs,
        "eval_texts_per_ratio": eval_texts_per_ratio,
        "world_size": world_size,
        "dev_texts": {k: [asdict(t) for t in texts] for k, texts in dev.items()},
        "test_texts": {k: [asdict(t) for t in texts] for k, texts in test.items()},
    }
    if resume and checkpoint.progress["identity"] != identity:
        raise ValueError("resume data, schedule or dynamic configuration differs")
    next_step = checkpoint.progress["next_step"] if resume else 0
    if stop_step <= next_step:
        raise ValueError("steps must exceed completed steps")
    random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    tokenizer, backbone, writer, value = load_components(checkpoint, device)
    if tokenizer.get_vocab() != data.tokenizer.get_vocab() or (
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    ) != (data.tokenizer.bos_token_id, data.tokenizer.eos_token_id):
        raise ValueError("SQuAD tokenizer differs from checkpoint tokenizer")
    trainer = DynamicTrainer(backbone, writer, checkpoint.config, recipe, device)
    if resume:
        trainer.optimizer.load_state_dict(checkpoint.optimizer_state)
        restore_rng_state(checkpoint.progress["rank_rng_states"][rank])
    if world_size > 1:
        dist.barrier()
    output_dir.mkdir(parents=True, exist_ok=True)
    plans_dir = output_dir / "data_plans"
    plans_dir.mkdir(exist_ok=True)
    origin = checkpoint.progress["initial_checkpoint"] if resume else str(checkpoint_path.resolve())
    run_info = identity | {
        "initial_checkpoint": origin,
        "target_steps": total_steps,
        "stop_step": stop_step,
        "nll_includes_eos": True,
        "model_config": checkpoint.config.to_dict(),
        "runtime": runtime_info(),
    }
    if primary:
        (output_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    final = None
    with (
        swanlab_run(
            output_dir,
            run_info,
            mode=swanlab_mode if primary else "disabled",
            project="latent-working-memory-v1",
            job_type="train",
            group=swanlab_group,
            fixed_tags=("scope:main", "method:joint", "data:squad"),
        ) as tracking,
        (
            (output_dir / f"train-{uuid.uuid4().hex}.jsonl").open("x") if primary else nullcontext()
        ) as log,
    ):

        def evaluate(split, panel, step):
            metrics, rows = evaluate_panel(
                backbone, writer, tokenizer, checkpoint.config, recipe, data, panel, device
            )
            if primary:
                write_evaluation(output_dir, f"{split}-{step:06d}", metrics, rows)
            if tracking is not None:
                tracking.log(
                    {
                        f"evaluation/{split}/{group}/{key}": val
                        for group, values in metrics.items()
                        for key, val in values.items()
                    },
                    step=step,
                )

        if next_step == 0:
            evaluate("dev", dev, 0)
        active_micro = None
        micro_totals = Counter(checkpoint.progress["micro_totals"]) if resume else Counter()
        for step in range(next_step, stop_step):
            micro_index, batch_index = divmod(step, recipe.steps_per_micro_epoch)
            epoch, micro = divmod(micro_index, recipe.micro_epochs_per_epoch)
            if active_micro != micro_index:
                if batch_index == 0:
                    micro_totals.clear()
                capacity, texts, report = sampler.micro_epoch(epoch, micro, epochs)
                if primary:
                    (plans_dir / f"micro-{epoch:06d}-{micro:04d}.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                active_micro = micro_index
            offset = batch_index * recipe.batch_size
            batch = texts[offset : offset + recipe.batch_size]
            episodes = [text.episode(data) for text in batch]
            seeds = [f"{recipe.seed}:read:{epoch}:{micro}:{offset + i}" for i in range(len(batch))]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            begin = time.perf_counter()
            result = trainer.step(episodes, tokenizer, seeds, capacity)
            for row, text in zip(result["sample_metrics"], batch, strict=True):
                row.update(asdict(text))
                row["actual_ratio"] = text.input_tokens / capacity
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            resources = torch.tensor(
                [
                    time.perf_counter() - begin,
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                ],
                dtype=torch.float64,
                device=device,
            )
            if world_size > 1:
                dist.all_reduce(resources, op=dist.ReduceOp.MAX)
            result["seconds"], peak_memory = resources.tolist()
            result.update(
                peak_memory_bytes=int(peak_memory),
                step=step + 1,
                epoch=epoch,
                micro_epoch=micro,
                batch_in_micro_epoch=batch_index,
                samples_seen=(step + 1) * recipe.batch_size,
            )
            result["input_tokens_per_second"] = result["input_tokens"] / result["seconds"]
            micro_totals.update(
                {
                    key: result[key]
                    for key in (
                        "samples",
                        "input_tokens",
                        "target_tokens",
                        "reads",
                        "writes",
                        "updates",
                        "truncations",
                    )
                }
            )
            if primary:
                log.write(json.dumps(result) + "\n")
                log.flush()
                print(
                    json.dumps({k: v for k, v in result.items() if k != "sample_metrics"}),
                    flush=True,
                )
            if tracking is not None:
                tracking.log(
                    {
                        f"{'resources' if k in {'seconds', 'peak_memory_bytes', 'input_tokens_per_second'} else 'train'}/{k}": v
                        for k, v in result.items()
                        if isinstance(v, (int, float))
                    },
                    step=step + 1,
                )
            if batch_index + 1 == recipe.steps_per_micro_epoch and primary:
                report["training_totals"] = dict(micro_totals)
                (plans_dir / f"micro-{epoch:06d}-{micro:04d}.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                if micro + 1 == recipe.micro_epochs_per_epoch:
                    counts = Counter()
                    for m in range(recipe.micro_epochs_per_epoch):
                        plan = json.loads(
                            (plans_dir / f"micro-{epoch:06d}-{m:04d}.json").read_text()
                        )
                        for r, n in plan["used_counts"].items():
                            counts[f"k{plan['capacity']}/r{r}"] += n
                    total = sum(counts.values())
                    (plans_dir / f"epoch-{epoch:06d}.json").write_text(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "used_counts": dict(counts),
                                "sample_proportions": {k: n / total for k, n in counts.items()},
                            },
                            indent=2,
                        )
                        + "\n"
                    )
            if (step + 1) % eval_every == 0 or step + 1 == stop_step:
                evaluate("dev", dev, step + 1)
            if step + 1 == total_steps:
                evaluate("test", test, step + 1)
            if (step + 1) % save_every == 0 or step + 1 == stop_step:
                final = output_dir / f"dynamic-step-{step + 1:06d}.pt"
                rank_rng_states = [capture_rng_state()]
                if world_size > 1:
                    rank_rng_states = [None] * world_size
                    dist.all_gather_object(rank_rng_states, capture_rng_state())
                next_micro, next_batch = divmod(step + 1, recipe.steps_per_micro_epoch)
                next_epoch, next_micro = divmod(next_micro, recipe.micro_epochs_per_epoch)
                if primary:
                    save_model_checkpoint(
                        final,
                        "dynamic",
                        checkpoint.config,
                        trainable_model_state(backbone, writer, value),
                        trainer.optimizer.state_dict(),
                        {
                            "next_step": step + 1,
                            "epoch": next_epoch,
                            "micro_epoch": next_micro,
                            "batch_in_micro_epoch": next_batch,
                            "samples_seen": (step + 1) * recipe.batch_size,
                            "micro_totals": dict(micro_totals) if next_batch else {},
                            "identity": identity,
                            "initial_checkpoint": origin,
                            "rank_rng_states": rank_rng_states,
                        },
                        capture_rng_state(),
                    )
                if world_size > 1:
                    dist.barrier()
    return final


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate dynamic SQuAD memory")
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--steps", type=int, help="Stop after this global step within the epoch schedule"
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-texts-per-ratio", type=int, default=2)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--swanlab-group")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=device)
    else:
        device = torch.device(args.device)
    validate_device(device)
    recipe = DynamicConfig(**json.loads(args.recipe.read_text()))
    if args.mode == "train":
        run_dynamic(
            args.checkpoint,
            args.index,
            args.output_dir,
            recipe,
            device,
            args.steps,
            args.save_every,
            args.eval_every,
            args.eval_texts_per_ratio,
            args.resume,
            args.swanlab_mode,
            args.swanlab_group,
            args.epochs,
        )
    else:
        if args.resume or args.output_dir.exists() or args.eval_texts_per_ratio < 1:
            raise ValueError("evaluation needs a new directory and positive text count")
        checkpoint = load_model_checkpoint(args.checkpoint)
        checkpoint = replace(
            checkpoint, config=replace(checkpoint.config, gradient_checkpointing=False)
        )
        data = SquadDataset(args.index)
        sampler = DynamicTextSampler(data, recipe, checkpoint.config.write_context_tokens)
        panel = sampler.evaluation_texts(args.split, args.eval_texts_per_ratio)
        tokenizer, backbone, writer, _ = load_components(checkpoint, device)
        if tokenizer.get_vocab() != data.tokenizer.get_vocab() or (
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ) != (data.tokenizer.bos_token_id, data.tokenizer.eos_token_id):
            raise ValueError("SQuAD tokenizer differs from checkpoint tokenizer")
        metrics, rows = evaluate_panel(
            backbone, writer, tokenizer, checkpoint.config, recipe, data, panel, device
        )
        primary = not dist.is_initialized() or dist.get_rank() == 0
        if primary:
            args.output_dir.mkdir(parents=True)
            write_evaluation(args.output_dir, "metrics", metrics, rows)
            (args.output_dir / "evaluation.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(args.checkpoint.resolve()),
                        "index": str(args.index.resolve()),
                        "split": args.split,
                        "runtime": runtime_info(),
                        "recipe": asdict(recipe),
                        "texts": {k: [asdict(t) for t in texts] for k, texts in panel.items()},
                    },
                    indent=2,
                )
                + "\n"
            )
        with swanlab_run(
            args.output_dir,
            asdict(recipe) | {"checkpoint": str(args.checkpoint), "split": args.split},
            mode=args.swanlab_mode if primary else "disabled",
            project="latent-working-memory-v1",
            job_type="evaluate",
            group=args.swanlab_group,
            fixed_tags=("scope:main", "method:joint", "data:squad"),
        ) as tracking:
            if tracking is not None:
                tracking.log(
                    {
                        f"evaluation/{group}/{key}": val
                        for group, values in metrics.items()
                        for key, val in values.items()
                    }
                )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
