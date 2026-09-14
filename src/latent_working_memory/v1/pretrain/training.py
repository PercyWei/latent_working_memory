from __future__ import annotations

import json
import math
import random
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from latent_working_memory.v1.training import (
    precision_context,
    trainable_model_state,
    load_trainable_model_state,
)
from latent_working_memory.v1.backbone import LatentMemoryBackbone, load_backbone
from latent_working_memory.v1.checkpoint import (
    capture_rng_state,
    load_model_checkpoint,
    restore_rng_state,
    save_model_checkpoint,
)
from latent_working_memory.v1.config import ExperimentConfig, write_resolved_config
from latent_working_memory.v1.pretrain.data_selection import select_experiment, selection_metadata
from latent_working_memory.v1.distributed import synchronize_gradients
from latent_working_memory.v1.pretrain.evaluation import evaluate_pretraining
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.v1.objectives import ReaderOutput
from latent_working_memory.v1.pretrain.sampling import (
    PretrainExample,
    EpochSampler,
)
from latent_working_memory.v1.pretrain.tokenization import TokenizationPool
from latent_working_memory.v1.pretrain.tracking import (
    pretraining_tracking_config,
    log_evaluation,
    log_training,
)
from latent_working_memory.v1.pretrain.tracking import pretraining_run
from latent_working_memory.v1.tracking import DEFAULT_SWANLAB_PROJECT


def learning_rate_at(config: ExperimentConfig, step: int, total_steps: int | None = None) -> float:
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    decay_steps = total_steps if total_steps is not None else config.lr_decay_steps
    if not decay_steps:
        return config.learning_rate
    progress = min(
        max((step - config.warmup_steps) / max(decay_steps - config.warmup_steps, 1), 0), 1
    )
    return config.learning_rate * (
        config.min_lr_fraction
        + (1 - config.min_lr_fraction) * (1 + math.cos(math.pi * progress)) / 2
    )


@dataclass(frozen=True, slots=True)
class PretrainOutput:
    loss: Tensor
    ae: list[ReaderOutput | None]
    lm: list[ReaderOutput | None]


def pretrain_forward(
    config: ExperimentConfig,
    backbone: LatentMemoryBackbone,
    writer: JointMemoryWriter,
    examples: list[PretrainExample],
    sample_count: int | None = None,
) -> PretrainOutput:
    features = backbone.text_features([e.episode.input_ids for e in examples], [0] * len(examples))
    states = writer.update_batch(
        [writer.initialize_state(dtype=f.dtype) for f in features],
        features,
        [0] * len(examples),
        [e.capacity for e in examples],
    )
    memories = [state.values for state in states]
    ae, lm = [None] * len(examples), [None] * len(examples)
    count = sample_count if sample_count is not None else sum(e.loss_weight for e in examples)
    loss = memories[0].sum() * 0
    positions = [(i, "ae" if e.ae is not None else "lm") for i, e in enumerate(examples)]
    outputs = backbone.read_batch(memories, [getattr(examples[i], name) for i, name in positions])
    for (i, name), output in zip(positions, outputs, strict=True):
        destination = ae if name == "ae" else lm
        destination[i] = output
        loss = loss + examples[i].loss_weight * output.mean_nll / count
    return PretrainOutput(loss, ae, lm)


class PretrainTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        backbone: LatentMemoryBackbone,
        writer: JointMemoryWriter,
        device: torch.device,
    ) -> None:
        self.config, self.backbone, self.writer, self.device = config, backbone, writer, device
        self.parameters = list(backbone.trainable_parameters()) + list(writer.parameters())
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def step(self, examples: list[PretrainExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("a training step requires examples")
        self.backbone.train()
        self.writer.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Sorting changes only microbatch grouping, preserving the sampled document weights.
        ordered = sorted(
            examples,
            key=lambda e: max(e.input_length, len(e.lm.target_ids) if e.lm else 0) + e.capacity,
        )
        sample_count = round(sum(e.loss_weight for e in examples))
        distributed = dist.is_initialized()
        if distributed:
            ordered = ordered[dist.get_rank() :: dist.get_world_size()]
        records, loss_value = [], 0.0
        for start in range(0, len(ordered), self.config.batch_size):
            batch = ordered[start : start + self.config.batch_size]
            with precision_context(self.device):
                output = pretrain_forward(
                    self.config, self.backbone, self.writer, batch, sample_count
                )
                scaled_loss = output.loss
            scaled_loss.backward()
            loss_value += float(scaled_loss.detach())
            for example, ae, lm in zip(batch, output.ae, output.lm, strict=True):
                source = example.episode.sources[0]
                records.append(
                    {
                        "episode_id": example.episode.episode_id,
                        "document_id": source.document_id,
                        "boundary_variant": source.provenance["boundary_variant"],
                        "input_tokens": example.input_length,
                        "length_bucket": next(
                            (
                                b
                                for b in self.config.input_length_bounds
                                if example.input_length <= b
                            ),
                            self.config.max_input_tokens,
                        ),
                        "continuation_tokens": len(example.lm.target_ids) - 1 if example.lm else 0,
                        "capacity": example.capacity,
                        "effective_ratio": example.input_length / example.capacity,
                        "loss_weight": example.loss_weight,
                        "ae_nll": float(ae.mean_nll.detach()) if ae is not None else None,
                        "lm_nll": float(lm.mean_nll.detach()) if lm is not None else None,
                    }
                )
            del output, scaled_loss, ae, lm
        if distributed:
            synchronize_gradients(self.parameters)
            loss_tensor = torch.tensor(loss_value, device=self.device)
            dist.all_reduce(loss_tensor)
            loss_value = loss_tensor.item()
            rank_records = [None] * dist.get_world_size()
            dist.all_gather_object(rank_records, records)
            records = [row for rows in rank_records for row in rows]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters,
            self.config.gradient_clip,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        return {
            "loss": loss_value,
            "gradient_norm": float(grad_norm),
            "samples": records,
            "input_length_bounds": sorted(
                {*self.config.input_length_bounds, self.config.max_input_tokens}
            ),
            "input_tokens": sum(e.input_length for e in examples),
            "sample_visits": sample_count,
            "capacity_reads": len(examples),
            "target_tokens": sum(
                len(e.ae.target_ids if e.ae else e.lm.target_ids) for e in examples
            ),
        }


@dataclass(frozen=True, slots=True)
class PretrainRunResult:
    final_checkpoint: Path
    completed_steps: int
    dev_metrics: dict[str, Any]


def run_pretraining(
    config: ExperimentConfig,
    data_selection: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    save_every: int = 100,
    resume: Path | None = None,
    stop_after_steps: int | None = None,
    swanlab_mode: str = "disabled",
    swanlab_project: str = DEFAULT_SWANLAB_PROJECT,
    swanlab_group: str | None = None,
    swanlab_tags: tuple[str, ...] = (),
    max_samples_per_epoch: int | None = None,
    tokenizer_workers: int = 0,
    tokenization_batch_size: int = 256,
    prefetch_batches: int = 2,
) -> PretrainRunResult:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    primary = rank == 0
    if type(epochs) is not int or epochs <= 0 or save_every <= 0:
        raise ValueError("epochs and save_every must be positive")
    if max_samples_per_epoch is not None and (
        type(max_samples_per_epoch) is not int or max_samples_per_epoch <= 0
    ):
        raise ValueError("max_samples_per_epoch must be a positive integer or null")
    if stop_after_steps is not None and stop_after_steps <= 0:
        raise ValueError("stop_after_steps must be positive")
    if output_dir.exists() and resume is None:
        raise FileExistsError("use a new output directory or resume an existing run")
    random.seed(config.model_seed)
    torch.manual_seed(config.model_seed)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer, backbone = load_backbone(config, device, dtype)
    with TokenizationPool(tokenizer, config, tokenizer_workers) as tokenization:
        dev_indices, test_indices, evaluation_ids = {}, {}, {}
        spec = json.loads(data_selection.read_text())
        indices, selection_report = select_experiment(
            spec,
            config,
            tokenizer,
            tokenization=tokenization,
            tokenization_batch_size=tokenization_batch_size,
        )
        train_index = indices["train", "train"]
        metadata = selection_metadata(selection_report, "train")
        for name in spec["sources"]:
            dev_indices[name], test_indices[name] = indices[name, "dev"], indices[name, "test"]
            evaluation_ids[name] = selection_metadata(selection_report, name)["preparation_id"]
        for index in [*dev_indices.values(), *test_indices.values()]:
            if (
                train_index.source_ids & index.source_ids
                or train_index.cluster_ids & index.cluster_ids
                or train_index.groups.keys() & index.groups.keys()
            ):
                raise ValueError("train/evaluation source leakage")
        if (
            max(config.write_context_tokens, config.read_context_tokens)
            > backbone.max_position_embeddings
        ):
            raise ValueError("configured context budgets exceed the language model window")
        writer = JointMemoryWriter(
            config.d_mem, config.num_layers, config.num_heads, config.ffn_dim, config.k_limit
        ).to(device)
        value = GrowthValueNetwork(config.d_mem).to(device)
        value.requires_grad_(False)
        trainer = PretrainTrainer(config, backbone, writer, device)
        batch_size = config.batch_size * config.gradient_accumulation_steps * world_size
        sampler = EpochSampler(
            train_index,
            tokenizer,
            config,
            spec["training"],
            spec["seed"],
            batch_size,
            epochs,
            max_samples_per_epoch,
            tokenization,
            prefetch_batches,
        )
        max_steps = (
            min(sampler.total_steps, stop_after_steps) if stop_after_steps else sampler.total_steps
        )
        run_identity = {
            "world_size": world_size,
            "preparation_id": metadata["preparation_id"],
            "epochs": epochs,
            "max_samples_per_epoch": max_samples_per_epoch,
            "total_steps": sampler.total_steps,
            "evaluation_preparations": evaluation_ids,
        }
        next_step, input_tokens, target_tokens = 0, 0, 0
        seen_documents: set[str] = set()
        seen_samples: set[str] = set()
        checkpoint = None
        if resume:
            checkpoint = load_model_checkpoint(resume)
            if checkpoint.phase != "pretrain" or checkpoint.config != config:
                raise ValueError("checkpoint phase/config differs from this pretraining run")
            if checkpoint.progress["run_identity"] != run_identity:
                raise ValueError("checkpoint data preparation or sampling limit differs")
            load_trainable_model_state(checkpoint.model_state, backbone, writer, value)
            trainer.optimizer.load_state_dict(checkpoint.optimizer_state)
            sampler.load_state_dict(checkpoint.progress["sampler"])
            next_step = checkpoint.progress["next_step"]
            input_tokens = checkpoint.progress["input_tokens"]
            target_tokens = checkpoint.progress["target_tokens"]
            seen_documents = set(checkpoint.progress["seen_documents"])
            seen_samples = set(checkpoint.progress["seen_samples"])
            if max_steps <= next_step:
                raise ValueError("max_steps must exceed the resumed completed step count")
            restore_rng_state(checkpoint.progress["rank_rng_states"][rank])
        if world_size > 1:
            dist.barrier()
        output_dir.mkdir(parents=True, exist_ok=True)
        if primary:
            (output_dir / "epoch-plan.json").write_text(
                json.dumps([sampler.epoch_report(e) for e in range(1, epochs + 1)], indent=2) + "\n"
            )
            write_resolved_config(config, output_dir / "config.json")
            if data_selection is not None:
                (output_dir / "data-selection.json").write_text(json.dumps(spec, indent=2) + "\n")
            (output_dir / "provenance.json").write_text(
                json.dumps(
                    {
                        "preparation": metadata,
                        "torch": torch.__version__,
                        "device": str(device),
                        "model_dtype": str(dtype),
                        "run_identity": run_identity,
                        "tokenization": {
                            "workers": tokenizer_workers,
                            "batch_size": tokenization_batch_size,
                            "prefetch_batches": prefetch_batches,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        # A resume segment has its own log, so prior completed or interrupted logs remain reviewable.
        segment_id = f"{next_step:06d}-{uuid.uuid4().hex[:12]}"
        log_path = output_dir / f"train-from-{segment_id}.jsonl"
        dev_metrics: dict[str, Any] = {}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        final_checkpoint = checkpoint_dir / f"pretrain-step-{max_steps:06d}.pt"
        with (
            sampler,
            pretraining_run(
                output_dir,
                pretraining_tracking_config(config.to_dict(), run_identity, metadata),
                swanlab_mode if primary else "disabled",
                swanlab_project,
                group=swanlab_group,
                tags=swanlab_tags,
            ) as tracking,
            log_path.open("x", encoding="utf-8") if primary else nullcontext() as log,
        ):

            def evaluate_sets(step):
                # Each GPU evaluates one source; rank zero publishes the combined reports.
                results = {}
                if world_size > 1:
                    dist.barrier()
                for number, (name, index) in enumerate(dev_indices.items()):
                    if number % world_size != rank:
                        continue
                    destination = output_dir if name == "dev" else output_dir / name
                    with precision_context(device):
                        results[name] = evaluate_pretraining(
                            config,
                            tokenizer,
                            backbone,
                            writer,
                            index,
                            destination,
                            step,
                            input_tokens,
                        )
                if world_size > 1:
                    gathered = [None] * world_size
                    dist.all_gather_object(gathered, results)
                    results = {k: v for item in gathered for k, v in item.items()}
                if primary:
                    records_paths = {
                        name: (output_dir if name == "dev" else output_dir / name)
                        / f"dev-step-{step:06d}.jsonl"
                        for name in dev_indices
                    }
                    log_evaluation(tracking, results, records_paths, step)
                if world_size > 1:
                    dist.barrier()
                return results["dev"] if list(results) == ["dev"] else results

            if next_step == 0:
                evaluate_sets(0)
            for step in range(next_step, max_steps):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_begin = time.perf_counter()
                examples = sampler.sample_batch()
                for group in trainer.optimizer.param_groups:
                    group["lr"] = learning_rate_at(config, step, sampler.total_steps)
                result = trainer.step(examples)
                result["learning_rate"] = learning_rate_at(config, step, sampler.total_steps)
                result["epoch"] = sampler.epoch
                result["epoch_progress"] = sampler.cursor / len(sampler.order)
                result["epoch_samples"] = len(sampler.order)
                result["completed_epochs"] = sampler.epoch - (sampler.cursor < len(sampler.order))
                seen_samples.update(e.episode.episode_id for e in examples)
                result["distinct_samples"] = len(seen_samples)
                result["sample_visits"] = sampler.visits
                input_tokens += result["input_tokens"]
                target_tokens += result["target_tokens"]
                seen_documents.update(e.episode.sources[0].document_id for e in examples)
                result["distinct_documents"] = len(seen_documents)
                result["document_visits"] = sampler.visits
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - step_begin
                peak_memory = (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                )
                if world_size > 1:
                    resources = torch.tensor(
                        [elapsed, peak_memory], dtype=torch.float64, device=device
                    )
                    dist.all_reduce(resources, op=dist.ReduceOp.MAX)
                    elapsed, peak_memory = resources.tolist()
                result.update(
                    step=step + 1,
                    seconds=elapsed,
                    input_tokens_per_second=result["input_tokens"] / elapsed,
                    peak_memory_bytes=int(peak_memory),
                )
                if primary:
                    log.write(json.dumps(result, ensure_ascii=False) + "\n")
                    log.flush()
                    print(
                        json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True
                    )
                    log_training(tracking, result, input_tokens, target_tokens)
                if (
                    (step + 1) % save_every == 0
                    or step + 1 == max_steps
                    or sampler.cursor == len(sampler.order)
                ):
                    path = checkpoint_dir / f"pretrain-step-{step + 1:06d}.pt"
                    rank_rng_states = [capture_rng_state()]
                    if world_size > 1:
                        rank_rng_states = [None] * world_size
                        dist.all_gather_object(rank_rng_states, capture_rng_state())
                    if primary:
                        save_model_checkpoint(
                            path,
                            "pretrain",
                            config,
                            trainable_model_state(backbone, writer, value),
                            trainer.optimizer.state_dict(),
                            {
                                "next_step": step + 1,
                                "rank_rng_states": rank_rng_states,
                                "sampler": sampler.state_dict(),
                                "input_tokens": input_tokens,
                                "target_tokens": target_tokens,
                                "seen_documents": sorted(seen_documents),
                                "seen_samples": sorted(seen_samples),
                                "run_identity": run_identity,
                            },
                            capture_rng_state(),
                        )
                if (
                    (step + 1) % config.eval_every == 0
                    or step + 1 == max_steps
                    or sampler.cursor == len(sampler.order)
                ):
                    dev_metrics = evaluate_sets(step + 1)
        if primary:
            (output_dir / f"resources-from-{segment_id}.json").write_text(
                json.dumps(
                    {
                        "seconds": time.perf_counter() - begin,
                        "completed_steps": max_steps,
                        "cumulative_input_tokens": input_tokens,
                        "cumulative_target_tokens": target_tokens,
                        "distinct_documents": len(seen_documents),
                        "document_visits": sampler.visits,
                        "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0,
                    },
                    indent=2,
                )
                + "\n"
            )
        if primary:
            (output_dir / "training-result.json").write_text(
                json.dumps(
                    {
                        "final_checkpoint": str(final_checkpoint.resolve()),
                        "completed_steps": max_steps,
                        "completed_epochs": sampler.epoch - (sampler.cursor < len(sampler.order)),
                        "epochs": epochs,
                        "total_steps": sampler.total_steps,
                        "complete": max_steps == sampler.total_steps,
                    },
                    indent=2,
                )
                + "\n"
            )
        return PretrainRunResult(final_checkpoint, max_steps, dev_metrics)
