from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import json
import os
import time
from pathlib import Path
import random
import re
import string
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
from latent_working_memory.v1.tracking import swanlab_run
from latent_working_memory.v1.training import (
    load_trainable_model_state,
    precision_context,
    trainable_model_state,
)


@dataclass(frozen=True)
class DynamicConfig:
    capacity: int
    min_tokens: int
    max_tokens: int
    new_count: int = 1
    history_count: int = 1
    max_visits: int = 2
    bptt_unit: str = "tokens"
    bptt_span: int = 0  # Zero uses full BPTT.
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 0.00003
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    generation_tokens: int = 64
    seed: int = 42

    def __post_init__(self):
        for name in (
            "capacity",
            "max_tokens",
            "max_visits",
            "generation_tokens",
            "gradient_accumulation_steps",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("min_tokens", "new_count", "history_count", "bptt_span", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")
        if self.bptt_unit not in {"tokens", "updates"}:
            raise ValueError("bptt_unit must be tokens or updates")
        if self.min_tokens > self.max_tokens or self.new_count + self.history_count == 0:
            raise ValueError("invalid length interval or empty reading policy")
        if not (
            0 < self.learning_rate < float("inf")
            and 0 < self.gradient_clip < float("inf")
            and 0 <= self.weight_decay < float("inf")
        ):
            raise ValueError("invalid optimizer parameters")


def read_schedule(episode, tokenizer, recipe, model_config, seed, generation=False):
    if recipe.capacity > model_config.k_limit:
        raise ValueError("capacity exceeds model slot limit")
    previous = 0
    rng, visits, schedule = random.Random(seed), {}, {}
    for end in episode.write_ends:
        if end - previous + 1 > model_config.write_context_tokens:
            raise ValueError("complete paragraph exceeds write context budget")
        previous = end
        jobs = []
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
                1 + recipe.capacity + len(tokens.prompt_ids) + target_budget
                > model_config.read_context_tokens
            ):
                raise ValueError(f"QA read exceeds context budget: {read.read_id}")
            jobs.append((read, tokens))
        schedule[end] = jobs
    if not any(schedule.values()):
        raise ValueError("episode has no selected reads")
    return schedule


def shuffled_articles(documents, seed, start=0):
    """Yield complete shuffled epochs, resuming at a global article offset."""
    epoch, offset = divmod(start, len(documents))
    while True:
        order = list(documents)
        random.Random(f"{seed}:epoch:{epoch}").shuffle(order)
        yield from order[offset:]
        epoch += 1
        offset = 0


class DynamicTrainer:
    def __init__(self, backbone, writer, model_config, recipe, device):
        self.backbone, self.writer = backbone, writer
        self.model_config, self.recipe, self.device = model_config, recipe, device
        self.parameters = list(backbone.trainable_parameters()) + list(writer.parameters())
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=recipe.learning_rate, weight_decay=recipe.weight_decay
        )

    def step(self, episodes, tokenizer, seeds, allow_partial=False):
        batch_size = len(episodes)
        expected = self.recipe.gradient_accumulation_steps
        if (
            len(seeds) != batch_size
            or not 1 <= batch_size <= expected
            or (batch_size != expected and not allow_partial)
        ):
            raise ValueError(
                "one optimizer step requires gradient_accumulation_steps articles and seeds"
            )
        self.backbone.train()
        self.writer.train()
        self.optimizer.zero_grad(set_to_none=True)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        articles = [
            self._backward_episode(episodes[i], tokenizer, seeds[i], batch_size)
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
            for key in ("target_tokens", "input_tokens", "reads", "writes", "truncations")
        }
        return {
            "articles": batch_size,
            "article_metrics": articles,
            "loss": sum(a["loss"] for a in articles) / batch_size,
            "target_nll": sum(a["target_nll"] * a["target_tokens"] for a in articles)
            / totals["target_tokens"],
            **totals,
            "gradient_norm": float(grad_norm),
        }

    def _read_loss(self, memory, tokens):
        return self.backbone.read_batch([memory], [tokens])[0].mean_nll

    def _backward_episode(self, episode, tokenizer, seed, batch_size):
        schedule = read_schedule(episode, tokenizer, self.recipe, self.model_config, seed)
        count = sum(len(jobs) for jobs in schedule.values())
        state = None
        previous = segment_start = 0
        pending, loss_value, token_nll, target_tokens = [], 0.0, 0.0, 0
        truncations = segment_updates = 0
        segments = []
        for end in episode.write_ends:
            with precision_context(self.device):
                features = self.backbone.text_features(
                    [episode.input_ids[previous:end]], [previous]
                )[0]
                if state is None:
                    state = self.writer(
                        self.writer.initialize_state(features.dtype),
                        features,
                        first_slots=self.recipe.capacity,
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
            "writes": len(episode.write_ends),
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
    for end in episode.write_ends:
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
                    for a, b in zip((0, *donor.write_ends[:-1]), donor.write_ends)
                ):
                    raise ValueError("donor paragraph exceeds write context budget")
                wrong = encode_episode(backbone, writer, donor, recipe.capacity).values
            state, previous = None, 0
            for end in episode.write_ends:
                f = backbone.text_features([episode.input_ids[previous:end]], [previous])[0]
                state = (
                    writer(writer.initialize_state(f.dtype), f, first_slots=recipe.capacity)
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
                                "delay_writes": episode.write_ends.index(end)
                                - episode.write_ends.index(evidence_end),
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
        records = sorted(
            [row for rank_rows in gathered for row in rank_rows],
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
                metrics[f"{condition}/{kind}"] = {
                    "reads": len(rows),
                    "em": sum(r["em"] for r in rows) / len(rows),
                    "f1": sum(r["f1"] for r in rows) / len(rows),
                    "nll": sum(r["nll_sum"] for r in rows) / sum(r["target_tokens"] for r in rows),
                    "hit_limit_rate": sum(r["hit_limit"] for r in rows) / len(rows),
                }
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


def run_dynamic(
    checkpoint_path,
    index_path,
    output_dir,
    recipe,
    device,
    steps,
    save_every=100,
    eval_every=100,
    eval_articles=8,
    resume=False,
    swanlab_mode="disabled",
    swanlab_group=None,
    epochs=None,
):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    primary = rank == 0
    if min(steps, save_every, eval_every, eval_articles) <= 0:
        raise ValueError("run counts must be positive")
    if output_dir.exists() and not resume:
        raise FileExistsError("use a new output directory")
    checkpoint = load_model_checkpoint(checkpoint_path)
    if checkpoint.phase != ("dynamic" if resume else "pretrain"):
        raise ValueError("initialization needs pretrain; resume needs dynamic checkpoint")
    checkpoint = replace(
        checkpoint,
        config=replace(checkpoint.config, gradient_checkpointing=False),
    )
    data = SquadDataset(index_path)
    docs = data.select("train", recipe.min_tokens, recipe.max_tokens)
    dev_docs = data.select("dev", recipe.min_tokens, recipe.max_tokens)[:eval_articles]
    if not docs or len(dev_docs) < 2:
        raise ValueError("selection needs training articles and at least two dev articles")
    identity = {
        "recipe": asdict(recipe),
        "data_index": data.index,
        "eval_documents": dev_docs,
        "article_sampling": "shuffled_epochs",
        "world_size": world_size,
    }
    if resume and checkpoint.progress["identity"] != identity:
        raise ValueError("resume data index, sampling policy or dynamic configuration differs")
    random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    tokenizer, backbone, writer, value = load_components(checkpoint, device)
    if tokenizer.get_vocab() != data.tokenizer.get_vocab() or (
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    ) != (data.tokenizer.bos_token_id, data.tokenizer.eos_token_id):
        raise ValueError("SQuAD tokenizer differs from checkpoint tokenizer")
    trainer = DynamicTrainer(backbone, writer, checkpoint.config, recipe, device)
    next_step = checkpoint.progress["next_step"] if resume else 0
    articles_seen = checkpoint.progress["articles_seen"] if resume else 0
    if epochs is not None:
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        total_articles = epochs * len(docs)
        remaining = total_articles - articles_seen
        steps = (
            next_step
            + (remaining + recipe.gradient_accumulation_steps - 1)
            // recipe.gradient_accumulation_steps
        )
    else:
        total_articles = articles_seen + (steps - next_step) * recipe.gradient_accumulation_steps
    if steps <= next_step:
        raise ValueError("steps must exceed completed steps")
    if resume:
        trainer.optimizer.load_state_dict(checkpoint.optimizer_state)
        restore_rng_state(checkpoint.progress["rank_rng_states"][rank])
    if world_size > 1:
        dist.barrier()
    output_dir.mkdir(parents=True, exist_ok=True)
    origin = checkpoint.progress["initial_checkpoint"] if resume else str(checkpoint_path.resolve())
    if primary:
        (output_dir / "run.json").write_text(
            json.dumps(
                identity
                | {
                    "initial_checkpoint": origin,
                    "train_articles": len(docs),
                    "eval_documents": dev_docs,
                    "nll_includes_eos": True,
                    "target_articles": total_articles,
                    "target_epochs": epochs,
                    "target_steps": steps,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    dev = [data.episode(doc) for doc in dev_docs]
    final = None
    with (
        swanlab_run(
            output_dir,
            asdict(recipe)
            | {
                "target_epochs": epochs,
                "target_steps": steps,
                "target_articles": total_articles,
                "world_size": world_size,
                "train_articles": len(docs),
                "article_sampling": "shuffled_epochs",
                "initial_checkpoint": origin,
                "model_config": checkpoint.config.to_dict(),
            },
            mode=swanlab_mode if primary else "disabled",
            project="latent-working-memory-v1",
            group=swanlab_group,
            fixed_tags=("scope:main", "method:joint", "data:squad"),
        ) as tracking,
        (
            (output_dir / f"train-{uuid.uuid4().hex}.jsonl").open("x") if primary else nullcontext()
        ) as log,
    ):
        if next_step == 0:
            metrics, rows = evaluate_qa(
                backbone, writer, tokenizer, checkpoint.config, recipe, dev, device
            )
            if primary:
                (output_dir / "dev-000000.json").write_text(json.dumps(metrics, indent=2) + "\n")
                (output_dir / "dev-000000.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
            if tracking is not None:
                tracking.log(
                    {
                        f"evaluation/{group}/{key}": value
                        for group, values in metrics.items()
                        for key, value in values.items()
                    },
                    step=0,
                )
        article_stream = shuffled_articles(docs, recipe.seed, articles_seen)
        for step in range(next_step, steps):
            article_indices = range(
                articles_seen,
                min(articles_seen + recipe.gradient_accumulation_steps, total_articles),
            )
            episodes = [data.episode(next(article_stream)) for _ in article_indices]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            begin = time.perf_counter()
            result = trainer.step(
                episodes,
                tokenizer,
                [f"{recipe.seed}:read:{i}" for i in article_indices],
                allow_partial=True,
            )
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
            result["peak_memory_bytes"] = int(peak_memory)
            result["input_tokens_per_second"] = result["input_tokens"] / result["seconds"]
            articles_seen += len(episodes)
            result["articles_seen"] = articles_seen
            result["epochs_completed"], result["articles_into_epoch"] = divmod(
                result["articles_seen"], len(docs)
            )
            result["step"] = step + 1
            if primary:
                log.write(json.dumps(result) + "\n")
                log.flush()
                print(
                    json.dumps({k: v for k, v in result.items() if k != "article_metrics"}),
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
            if (step + 1) % eval_every == 0 or step + 1 == steps:
                metrics, rows = evaluate_qa(
                    backbone, writer, tokenizer, checkpoint.config, recipe, dev, device
                )
                if primary:
                    (output_dir / f"dev-{step + 1:06d}.json").write_text(
                        json.dumps(metrics, indent=2) + "\n"
                    )
                    (output_dir / f"dev-{step + 1:06d}.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in rows)
                    )
                if tracking is not None:
                    tracking.log(
                        {
                            f"evaluation/{group}/{key}": value
                            for group, values in metrics.items()
                            for key, value in values.items()
                        },
                        step=step + 1,
                    )
            if (step + 1) % save_every == 0 or step + 1 == steps:
                final = output_dir / f"dynamic-step-{step + 1:06d}.pt"
                rank_rng_states = [capture_rng_state()]
                if world_size > 1:
                    rank_rng_states = [None] * world_size
                    dist.all_gather_object(rank_rng_states, capture_rng_state())
                if primary:
                    save_model_checkpoint(
                        final,
                        "dynamic",
                        checkpoint.config,
                        trainable_model_state(backbone, writer, value),
                        trainer.optimizer.state_dict(),
                        {
                            "next_step": step + 1,
                            "articles_seen": articles_seen,
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
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--epochs", type=int, help="Train exactly this many epochs; overrides --steps"
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-articles", type=int, default=8)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--swanlab-group")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if args.mode != "train":
            raise ValueError("distributed launch is supported by the train entry point")
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
            args.eval_articles,
            args.resume,
            args.swanlab_mode,
            args.swanlab_group,
            args.epochs,
        )
    else:
        if args.resume or args.output_dir.exists() or args.eval_articles < 2:
            raise ValueError("evaluation needs a new directory, no resume and >=2 articles")
        checkpoint = load_model_checkpoint(args.checkpoint)
        data = SquadDataset(args.index)
        docs = data.select(args.split, recipe.min_tokens, recipe.max_tokens)[: args.eval_articles]
        tokenizer, backbone, writer, _ = load_components(checkpoint, device)
        if tokenizer.get_vocab() != data.tokenizer.get_vocab() or (
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ) != (data.tokenizer.bos_token_id, data.tokenizer.eos_token_id):
            raise ValueError("SQuAD tokenizer differs from checkpoint tokenizer")
        metrics, rows = evaluate_qa(
            backbone,
            writer,
            tokenizer,
            checkpoint.config,
            recipe,
            [data.episode(doc) for doc in docs],
            device,
        )
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (args.output_dir / "reads.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (args.output_dir / "evaluation.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint.resolve()),
                    "index": str(args.index.resolve()),
                    "split": args.split,
                    "documents": docs,
                    "recipe": asdict(recipe),
                },
                indent=2,
            )
            + "\n"
        )
        with swanlab_run(
            args.output_dir,
            asdict(recipe)
            | {
                "checkpoint": str(args.checkpoint),
                "training_phase": checkpoint.phase,
                "evaluation_split": args.split,
                "evaluation_articles": len(docs),
            },
            mode=args.swanlab_mode,
            project="latent-working-memory-v1",
            job_type="evaluate",
            group=args.swanlab_group,
            fixed_tags=("scope:main", "method:joint", "data:squad"),
        ) as tracking:
            if tracking is not None:
                tracking.log(
                    {
                        f"evaluation/{group}/{key}": value
                        for group, values in metrics.items()
                        for key, value in values.items()
                    }
                )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
