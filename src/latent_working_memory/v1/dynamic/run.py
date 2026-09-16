"""Run dynamic-memory training or independent QA evaluation."""

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from latent_working_memory.v1.engine import initialize_device
from latent_working_memory.v1.checkpoint import (
    capture_rng_state,
    capture_rank_rng_states,
    load_model_checkpoint,
    restore_rng_state,
    save_model_checkpoint,
)
from latent_working_memory.v1.dynamic.config import load_dynamic_config
from latent_working_memory.v1.dynamic.data import DynamicTextSampler
from latent_working_memory.v1.dynamic.prepare import load_evaluation_plan
from latent_working_memory.v1.dynamic.training import DynamicTrainer, load_components
from latent_working_memory.v1.dynamic.evaluation import evaluate_panel, write_evaluation
from latent_working_memory.v1.dynamic.reporting import (
    log_qa,
    training_metrics,
    configure_dynamic_panels,
    append_qa_reports,
)
from latent_working_memory.v1.dynamic.selection import load_selection, load_source
from latent_working_memory.v1.tracking import swanlab_run
from latent_working_memory.v1.training import trainable_model_state, training_resources


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
            name: version(name) for name in ("torch", "transformers", "peft", "swanlab", "pyarrow", "verl")
        },
    }


def load_evaluation_sources(spec, tokenizer, recipe, names):
    sources = {}
    for name in names:
        source = spec["sources"][name]
        data = load_source(source, tokenizer)
        plan, panels = load_evaluation_plan(Path(source["evaluation_plan"]), data, recipe)
        required = {split for split, selected in spec["evaluation"].items() if name in selected}
        if not required <= panels.keys():
            raise ValueError(f"evaluation plan is missing required splits for {name}")
        sources[name] = data, plan, panels
    return sources


def run_dynamic(
    checkpoint_path,
    output_dir,
    recipe,
    device,
    evaluation_sets,
    steps=None,
    resume=False,
    swanlab_mode="disabled",
    swanlab_group=None,
    swanlab_tags=(),
    swanlab_project="latent-working-memory-v1",
):
    device = initialize_device(device)
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
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.config.model_name_or_path,
        revision=checkpoint.config.model_revision,
        local_files_only=True,
    )
    spec = load_selection(evaluation_sets, prepared=True)
    sources = load_evaluation_sources(spec, tokenizer, recipe, list(spec["sources"]))
    dataset = spec["training"]
    data = sources[dataset][0]
    sampler = DynamicTextSampler(data, recipe, checkpoint.config.write_context_tokens)
    identity = {
        "recipe": asdict(recipe),
        "selection": spec,
        "evaluation_plans": {name: source[1] for name, source in sources.items()},
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
        raise ValueError("dataset tokenizer differs from checkpoint tokenizer")
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
        "data_index": spec["sources"][dataset]["dataset_dir"],
        "selection": spec,
        "evaluation_sets": str(evaluation_sets.resolve()),
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
            project=swanlab_project,
            job_type="train",
            group=swanlab_group,
            tags=swanlab_tags,
            fixed_tags=("scope:main", "method:latent-working-memory", f"data:{dataset}"),
        ) as tracking,
        (
            (output_dir / f"train-from-{segment_id}.jsonl").open("x") if primary else nullcontext()
        ) as log,
    ):
        configure_dynamic_panels(tracking, swanlab_mode if primary else "disabled",
                                 spec["evaluation"]["dev"])

        def evaluate(step, generate):
            for name in spec["evaluation"]["dev"]:
                eval_data, shared_plan, panels = sources[name]
                metrics, rows = evaluate_panel(
                    backbone, writer, tokenizer, checkpoint.config, recipe, eval_data,
                    panels["dev"], device, generate=generate,
                    read_plans=shared_plan["reads"]["dev"],
                )
                if primary:
                    destination = output_dir / "dev" / name
                    write_evaluation(destination, f"dev-step-{step:06d}", metrics, rows)
                    (destination / "evaluation.json").write_text(json.dumps({
                        "name": name, "dataset": spec["sources"][name]["dataset"],
                        "split": "dev", "source": spec["sources"][name],
                    }, indent=2) + "\n")
                    log_qa(tracking, metrics, rows, step, "dev", name, media=generate)

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
            result.update(training_resources(device, time.perf_counter() - begin))
            result.update(
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
                tracking.log(training_metrics(result), step=step + 1)
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
                rank_rng_states = capture_rank_rng_states()
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-sets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, help="Stop within the configured epoch schedule")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-project", default="latent-working-memory-v1")
    parser.add_argument("--swanlab-tag", action="append", default=[])
    args = parser.parse_args()
    device = initialize_device(args.device)
    recipe = load_dynamic_config(args.config)
    if args.mode == "train":
        run_dynamic(
            args.checkpoint,
            args.output_dir,
            recipe,
            device,
            args.evaluation_sets,
            steps=args.steps,
            resume=args.resume,
            swanlab_mode=args.swanlab_mode,
            swanlab_group=args.swanlab_group,
            swanlab_project=args.swanlab_project,
            swanlab_tags=tuple(args.swanlab_tag),
        )
    else:
        if args.resume or args.steps or args.output_dir.exists():
            raise ValueError("evaluation needs a new output directory")
        checkpoint = load_model_checkpoint(args.checkpoint)
        checkpoint = replace(
            checkpoint, config=replace(checkpoint.config, gradient_checkpointing=False)
        )
        data_tokenizer = AutoTokenizer.from_pretrained(
            checkpoint.config.model_name_or_path,
            revision=checkpoint.config.model_revision,
            local_files_only=True,
        )
        spec = load_selection(args.evaluation_sets, prepared=True)
        names = spec["evaluation"][args.split]
        if not names:
            raise ValueError(f"no {args.split} sources configured")
        sources = load_evaluation_sources(spec, data_tokenizer, recipe, names)
        tokenizer, backbone, writer, _ = load_components(checkpoint, device)
        primary = not dist.is_initialized() or dist.get_rank() == 0
        step = checkpoint.progress["next_step"]
        reports = {}
        for name in names:
            data, shared_plan, panels = sources[name]
            if tokenizer.get_vocab() != data.tokenizer.get_vocab() or (
                tokenizer.bos_token_id, tokenizer.eos_token_id,
            ) != (data.tokenizer.bos_token_id, data.tokenizer.eos_token_id):
                raise ValueError("dataset tokenizer differs from checkpoint tokenizer")
            metrics, rows = evaluate_panel(
                backbone, writer, tokenizer, checkpoint.config, recipe, data,
                panels[args.split], device, read_plans=shared_plan["reads"][args.split],
            )
            destination = args.output_dir / name
            report = destination / f"{args.split}-step-{step:06d}.json"
            evaluation_info = {
                "name": name, "dataset": spec["sources"][name]["dataset"],
                "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": step,
                "index": spec["sources"][name]["dataset_dir"], "split": args.split,
                "config": asdict(recipe),
                "evaluation_plan": spec["sources"][name]["evaluation_plan"],
            }
            if primary:
                write_evaluation(destination, report.stem, metrics, rows)
                (destination / "evaluation.json").write_text(
                    json.dumps(evaluation_info | {"runtime": runtime_info()}, indent=2) + "\n")
                reports[name] = report
        if primary:
            (args.output_dir / "reports.json").write_text(json.dumps(
                {name: str(path.resolve()) for name, path in reports.items()}, indent=2) + "\n")
            if args.split == "test":
                append_qa_reports(args.checkpoint.parent.parent, reports, args.swanlab_mode)
            else:
                with swanlab_run(
                    args.output_dir,
                    {"checkpoint": str(args.checkpoint.resolve()), "selection": spec},
                    mode=args.swanlab_mode, project=args.swanlab_project, job_type="evaluate",
                    group=args.swanlab_group, tags=tuple(args.swanlab_tag),
                    fixed_tags=("scope:main", "method:latent-working-memory"),
                ) as tracking:
                    configure_dynamic_panels(tracking, args.swanlab_mode, names)
                    for name, path in reports.items():
                        metrics = json.loads(path.read_text())
                        rows = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
                        log_qa(tracking, metrics, rows, step, "dev", name, media=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
