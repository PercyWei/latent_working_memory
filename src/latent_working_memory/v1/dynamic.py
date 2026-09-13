"""Run dynamic-memory training or independent QA evaluation."""

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

import torch
import torch.distributed as dist

from latent_working_memory.devices import validate_device
from latent_working_memory.v1.checkpoint import (
    capture_rng_state,
    load_model_checkpoint,
    restore_rng_state,
    save_model_checkpoint,
)
from latent_working_memory.v1.dynamic_config import load_dynamic_config
from latent_working_memory.v1.dynamic_data import DynamicTextSampler
from latent_working_memory.data_preparation.dynamic import load_evaluation_plan
from latent_working_memory.v1.dynamic_training import DynamicTrainer, load_components
from latent_working_memory.v1.dynamic_evaluation import evaluate_panel, write_evaluation
from latent_working_memory.v1.dynamic_reporting import log_qa
from latent_working_memory.v1.squad import SquadDataset
from latent_working_memory.v1.tracking import swanlab_run
from latent_working_memory.v1.training import trainable_model_state


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
    evaluation_plan,
    steps=None,
    resume=False,
    swanlab_mode="disabled",
    swanlab_group=None,
    swanlab_tags=(),
):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    primary = rank == 0
    epochs = recipe.epochs
    if recipe.global_batch_size % world_size:
        raise ValueError("global batch must be divisible by world size")
    total_steps = epochs * recipe.micro_epochs_per_epoch * recipe.steps_per_micro_epoch
    if steps is not None and (type(steps) is not int or not 0 < steps <= total_steps):
        raise ValueError("steps must be positive and within the configured epoch schedule")
    stop_step = total_steps if steps is None else steps
    if output_dir.exists() and not resume:
        raise FileExistsError("use a new output directory")
    if resume and output_dir.resolve() != checkpoint_path.resolve().parent.parent:
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
    shared_plan, panels = load_evaluation_plan(evaluation_plan, data, recipe)
    identity = {
        "recipe": asdict(recipe),
        "evaluation_plan": shared_plan,
        "world_size": world_size,
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
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    plans_dir = output_dir / "data_plans"
    plans_dir.mkdir(exist_ok=True)
    origin = checkpoint.progress["initial_checkpoint"] if resume else str(checkpoint_path.resolve())
    segment_id = f"{next_step:06d}-{uuid.uuid4().hex[:12]}"
    run_info = {
        "config": asdict(recipe),
        "initial_checkpoint": origin,
        "data_index": str(index_path.resolve()),
        "evaluation_plan": str(evaluation_plan.resolve()),
        "target_steps": total_steps,
        "global_batch_size": recipe.global_batch_size,
        "world_size": world_size,
        "microbatch_per_device": 1,
        "gradient_accumulation_steps": recipe.global_batch_size // world_size,
        "nll_includes_eos": True,
        "model_config": checkpoint.config.to_dict(),
    }
    if primary:
        if not resume:
            (output_dir / "config.json").write_text(json.dumps(asdict(recipe), indent=2) + "\n")
            (output_dir / "provenance.json").write_text(json.dumps(run_info, indent=2) + "\n")
        (output_dir / f"runtime-from-{segment_id}.json").write_text(
            json.dumps(
                {
                    "runtime": runtime_info(),
                    "start_step": next_step,
                    "stop_step": stop_step,
                    "checkpoint": str(checkpoint_path.resolve()),
                },
                indent=2,
            )
            + "\n"
        )
    cumulative = Counter(checkpoint.progress["cumulative"]) if resume else Counter()
    begin_run = time.perf_counter()
    final = None
    with (
        swanlab_run(
            output_dir,
            run_info,
            mode=swanlab_mode if primary else "disabled",
            project="latent-working-memory-v1",
            job_type="train",
            group=swanlab_group,
            tags=swanlab_tags,
            fixed_tags=("scope:main", "method:latent-working-memory", "data:squad"),
        ) as tracking,
        (
            (output_dir / f"train-from-{segment_id}.jsonl").open("x") if primary else nullcontext()
        ) as log,
    ):

        def evaluate(step, generate):
            metrics, rows = evaluate_panel(
                backbone,
                writer,
                tokenizer,
                checkpoint.config,
                recipe,
                data,
                panels["dev"],
                device,
                generate=generate,
                read_plans=shared_plan["reads"]["dev"],
            )
            if primary:
                write_evaluation(output_dir / "dev", f"dev-step-{step:06d}", metrics, rows)
            log_qa(
                tracking, metrics, rows, step, "dev", media=generate, history_dir=output_dir / "dev"
            )

        if next_step == 0:
            evaluate(0, True)
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
            offset = batch_index * recipe.global_batch_size
            batch = texts[offset : offset + recipe.global_batch_size]
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
                samples_seen=(step + 1) * recipe.global_batch_size,
            )
            result["learning_rate"] = trainer.optimizer.param_groups[0]["lr"]
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
            cumulative.update(
                {
                    key: result[key]
                    for key in (
                        "samples",
                        "input_tokens",
                        "target_tokens",
                        "reads",
                        "writes",
                        "updates",
                    )
                }
            )
            result["cumulative"] = dict(cumulative)
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
                    }
                    | {f"train/cumulative_{k}": v for k, v in cumulative.items()},
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
            if (step + 1) % recipe.save_every == 0 or step + 1 == stop_step:
                final = checkpoint_dir / f"dynamic-step-{step + 1:06d}.pt"
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
                            "cumulative": dict(cumulative),
                            "epoch": next_epoch,
                            "micro_epoch": next_micro,
                            "batch_in_micro_epoch": next_batch,
                            "samples_seen": (step + 1) * recipe.global_batch_size,
                            "micro_totals": dict(micro_totals) if next_batch else {},
                            "identity": identity,
                            "initial_checkpoint": origin,
                            "rank_rng_states": rank_rng_states,
                        },
                        capture_rng_state(),
                    )
                if world_size > 1:
                    dist.barrier()
            generate = (step + 1) % recipe.eval_generation_every == 0 or step + 1 == stop_step
            if (step + 1) % recipe.eval_every == 0 or generate:
                evaluate(step + 1, generate)
    if primary:
        (output_dir / f"resources-from-{segment_id}.json").write_text(
            json.dumps(
                {
                    "start_step": next_step,
                    "completed_steps": stop_step,
                    "wall_seconds": time.perf_counter() - begin_run,
                    "cumulative": dict(cumulative),
                },
                indent=2,
            )
            + "\n"
        )
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, help="Stop within the configured epoch schedule")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-tag", action="append", default=[])
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=device)
    else:
        device = torch.device(args.device)
    validate_device(device)
    recipe = load_dynamic_config(args.config)
    if args.mode == "train":
        run_dynamic(
            args.checkpoint,
            args.index,
            args.output_dir,
            recipe,
            device,
            args.evaluation_plan,
            steps=args.steps,
            resume=args.resume,
            swanlab_mode=args.swanlab_mode,
            swanlab_group=args.swanlab_group,
            swanlab_tags=tuple(args.swanlab_tag),
        )
    else:
        if args.resume or args.steps or args.output_dir.exists():
            raise ValueError("evaluation needs a new output directory")
        checkpoint = load_model_checkpoint(args.checkpoint)
        checkpoint = replace(
            checkpoint, config=replace(checkpoint.config, gradient_checkpointing=False)
        )
        data = SquadDataset(args.index)
        shared_plan, panels = load_evaluation_plan(args.evaluation_plan, data, recipe)
        tokenizer, backbone, writer, _ = load_components(checkpoint, device)
        if tokenizer.get_vocab() != data.tokenizer.get_vocab() or (
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ) != (data.tokenizer.bos_token_id, data.tokenizer.eos_token_id):
            raise ValueError("SQuAD tokenizer differs from checkpoint tokenizer")
        metrics, rows = evaluate_panel(
            backbone,
            writer,
            tokenizer,
            checkpoint.config,
            recipe,
            data,
            panels[args.split],
            device,
            read_plans=shared_plan["reads"][args.split],
        )
        primary = not dist.is_initialized() or dist.get_rank() == 0
        step = checkpoint.progress["next_step"]
        evaluation_info = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_step": step,
            "index": str(args.index.resolve()),
            "split": args.split,
            "config": asdict(recipe),
            "evaluation_plan": str(args.evaluation_plan.resolve()),
        }
        if primary:
            args.output_dir.mkdir(parents=True)
            write_evaluation(args.output_dir, f"{args.split}-step-{step:06d}", metrics, rows)
            (args.output_dir / "evaluation.json").write_text(
                json.dumps(evaluation_info | {"runtime": runtime_info()}, indent=2) + "\n"
            )
        with swanlab_run(
            args.output_dir,
            evaluation_info,
            mode=args.swanlab_mode if primary else "disabled",
            project="latent-working-memory-v1",
            job_type="evaluate",
            group=args.swanlab_group,
            tags=tuple(args.swanlab_tag),
            fixed_tags=("scope:main", "method:latent-working-memory", "data:squad"),
        ) as tracking:
            log_qa(tracking, metrics, rows, 0, args.split, media=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
