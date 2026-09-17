"""python -m latent_working_memory.v2.pretrain.train --help"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import time

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer, set_seed

from latent_working_memory.v2.memory_codec import CodecConfig, MemoryCodec
from latent_working_memory.v2.pretrain.checkpoint import (
    restore_codec,
    restore_rng,
    save_checkpoint,
    prune_checkpoints,
)
from latent_working_memory.v2.pretrain.config import SelectionConfig, TrainingConfig
from latent_working_memory.v2.pretrain.data import (
    load_datasets,
    dataset_statistics,
    epoch_batches,
)
from latent_working_memory.v2.pretrain.engine import EngineRegistry, initialize_device
from latent_working_memory.v2.pretrain.evaluation import evaluate
from latent_working_memory.v2.pretrain.objective import ReconstructionTask
from latent_working_memory.v2.pretrain.tracking import (
    reconstruction_run,
    log_training,
    log_development,
    log_final_evaluation,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_experiment(path):
    raw = json.loads(path.read_text())
    if set(raw) != {"model", "selection", "training"}:
        raise ValueError("experiment requires model, selection references and training settings")
    model = CodecConfig(**json.loads((path.parent / raw["model"]).read_text()))
    selection = SelectionConfig(**json.loads((path.parent / raw["selection"]).read_text()))
    training = TrainingConfig(**raw["training"])
    return model, selection, training


def make_engine(task, device):
    engine = EngineRegistry.new(
        model_type="lwm_v2_reconstruction",
        backend="replicated",
        model=task,
        device=device,
    )
    engine.initialize()
    return engine


def run_training(args):
    if args.swanlab_mode != "disabled" and not args.swanlab_group:
        raise ValueError("enabled SwanLab requires an explicit --swanlab-group")
    model_config, selection, config = read_experiment(args.experiment)
    device = initialize_device(args.device)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    primary = rank == 0
    output = args.output_dir.resolve()
    if args.resume:
        if args.resume.resolve().parent.parent != output:
            raise ValueError("resume checkpoint must belong to output-dir/checkpoints")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("new training requires an empty output directory")
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, revision=model_config.revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"rank {rank}: loading and filtering prepared datasets", flush=True)
    base_config = AutoConfig.from_pretrained(
        model_config.model_name_or_path, revision=model_config.revision
    )
    datasets, preparation, filtering = load_datasets(
        selection, tokenizer, base_config.max_position_embeddings, config
    )
    task = ReconstructionTask(
        MemoryCodec(model_config, torch.bfloat16 if device.type == "cuda" else torch.float32),
        tokenizer,
        config,
    ).to(device)
    task.validate_data(datasets)
    schedule = [("warmup", e) for e in range(config.warmup_epochs)]
    schedule += [("multiround", e) for e in range(config.multiround_epochs)]
    total_steps = sum(
        math.ceil(len(datasets[stage]["train"]) / config.global_batch_size) for stage, _ in schedule
    )
    stage_endpoints = {total_steps}
    if config.warmup_epochs:
        stage_endpoints.add(
            math.ceil(len(datasets["warmup"]["train"]) / config.global_batch_size)
            * config.warmup_epochs
        )
    run = json.loads(
        json.dumps(
            {
                "model": asdict(model_config),
                "selection": asdict(selection),
                "data_preparation": preparation,
                "training": asdict(config),
                "world_size": world,
                "precision": "bf16" if device.type == "cuda" else "fp32",
                "resolved_model_revision": task.codec.backbone.config._commit_hash,
                "resolved_architecture": {
                    "encoder_layers": len(task.codec.backbone.get_base_model().model.layers),
                    "decoder_layers": len(task.codec.backbone.get_base_model().model.layers),
                    "alignment_layers": len(task.codec.read_alignment.layers),
                    "hidden_size": task.codec.width,
                    "encoder_attention": "causal",
                    "shared_backbone": True,
                    "reader_adapter": None,
                },
            }
        )
    )
    cursor = {
        "epoch_index": 0,
        "batch_index": 0,
        "step": 0,
        "source_tokens": 0,
        "target_tokens": 0,
        "sample_visits": 0,
    }
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["run"] != run or json.loads((output / "run.json").read_text()) != run:
            raise ValueError(
                "resume configuration, model revision, world size or precision differs"
            )
        restore_codec(task.codec, checkpoint["codec"])
        cursor = checkpoint["cursor"]
        task.codec.set_stage(checkpoint["stage"])
    else:
        task.codec.set_stage(schedule[0][0])
    engine = make_engine(task, device)
    if checkpoint:
        engine.optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"][rank], device)
        del checkpoint
    if primary:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "run.json", run)
        write_json(output / "data-summary.json", dataset_statistics(datasets))
        write_json(output / "data-filtering.json", filtering)
        write_json(
            output / "epoch-plan.json",
            {
                "epochs": [
                    {
                        "stage": stage,
                        "epoch": epoch + 1,
                        "samples": len(datasets[stage]["train"]),
                        "steps": math.ceil(
                            len(datasets[stage]["train"]) / config.global_batch_size
                        ),
                    }
                    for stage, epoch in schedule
                ],
                "total_steps": total_steps,
            },
        )
        if not args.resume:
            tokenizer.save_pretrained(output / "tokenizer")
            write_json(
                output / "provenance.json",
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "execution": {
                        "framework": "verl",
                        "backend": "replicated",
                        "world_size": world,
                    },
                    "engine_origin": "codex/verl-v1@e0bfa37",
                    "packages": {
                        n: version(n)
                        for n in ("torch", "transformers", "peft", "verl", "tensordict")
                    },
                    "git_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "git_dirty": bool(
                        subprocess.check_output(["git", "status", "--porcelain"], text=True)
                    ),
                },
            )
    tracking = (
        reconstruction_run(
            output,
            run,
            mode=args.swanlab_mode,
            project=args.swanlab_project,
            group=args.swanlab_group,
            tags=tuple(args.swanlab_tag),
        )
        if primary
        else nullcontext(None)
    )
    stop = args.stop_after_steps
    final_checkpoint = args.resume
    with tracking as tracker:
        for epoch_index in range(cursor["epoch_index"], len(schedule)):
            stage, epoch = schedule[epoch_index]
            if task.codec.stage != stage:
                # Stage transfer keeps the codec, copies learned Ar into Aw, and resets AdamW.
                del engine
                task.codec.initialize_write_alignment()
                task.codec.set_stage(stage)
                engine = make_engine(task, device)
            rows = datasets[stage]["train"]
            batches = math.ceil(len(rows) / config.global_batch_size)
            start_batch = cursor["batch_index"] if epoch_index == cursor["epoch_index"] else 0
            for batch_index, batch in enumerate(
                epoch_batches(rows, config.global_batch_size, config.seed, stage, epoch)
            ):
                if batch_index < start_batch:
                    continue
                if stop is not None and cursor["step"] >= stop:
                    break
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                begin = time.perf_counter()
                metrics = engine.step(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - begin
                resources = torch.tensor(
                    [
                        seconds,
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                if world > 1:
                    dist.all_reduce(resources, op=dist.ReduceOp.MAX)
                cursor["step"] += 1
                cursor["source_tokens"] += metrics["source_tokens"]
                cursor["target_tokens"] += metrics["target_tokens"]
                cursor["sample_visits"] += metrics["samples"]
                epoch_end = batch_index + 1 == batches
                cursor["epoch_index"] = epoch_index + int(epoch_end)
                cursor["batch_index"] = 0 if epoch_end else batch_index + 1
                step = cursor["step"]
                record = {
                    "step": step,
                    "stage": stage,
                    "epoch": epoch + 1,
                    "global_epoch": epoch_index + 1,
                    **metrics,
                    "seconds": resources[0].item(),
                    "peak_memory_bytes": int(resources[1].item()),
                }
                if primary:
                    with (output / "train.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                    log_training(
                        tracker,
                        record,
                        cursor,
                        engine.optimizer.param_groups[0]["lr"],
                        min((batch_index + 1) * config.global_batch_size, len(rows)) / len(rows),
                    )
                if step % config.eval_every == 0 or epoch_end:
                    # All training conditions use the same fixed multi-write evaluation panel.
                    metrics, records = evaluate(task, datasets["multiround"]["dev"], tokenizer)
                    if primary:
                        destination = output / "dev"
                        destination.mkdir(exist_ok=True)
                        write_json(
                            destination / f"step-{step:06d}.json",
                            {"metrics": metrics, "samples": records},
                        )
                        log_development(tracker, metrics, step)
                if (
                    step % config.save_every == 0
                    or epoch_end
                    or (stop is not None and step >= stop)
                ):
                    final_checkpoint = output / "checkpoints" / f"step-{step:06d}.pt"
                    save_checkpoint(final_checkpoint, engine, run, dict(cursor))
                    if primary:
                        prune_checkpoints(
                            final_checkpoint.parent, config.checkpoint_limit, stage_endpoints
                        )
            if stop is not None and cursor["step"] >= stop:
                break
        complete = cursor["epoch_index"] == len(schedule)
        if complete:
            metrics, records = evaluate(
                task, datasets["multiround"]["test"], tokenizer, config.generation_samples
            )
            if primary:
                write_json(output / "test.json", {"metrics": metrics, "samples": records})
                log_final_evaluation(tracker, metrics, records, cursor["step"])
        result = {
            "complete": complete,
            "completed_steps": cursor["step"],
            "total_steps": total_steps,
            "completed_epochs": cursor["epoch_index"],
            "source_tokens": cursor["source_tokens"],
            "target_tokens": cursor["target_tokens"],
            "checkpoint": str(final_checkpoint),
        }
        if primary:
            write_json(output / "training-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="v2 固定容量重构，verl replicated/DDP")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--stop-after-steps", type=int, help="本次执行停止的全局 step；保持原 epoch 预算"
    )
    parser.add_argument(
        "--swanlab-mode", choices=("disabled", "offline", "online"), default="online"
    )
    parser.add_argument("--swanlab-project", default="latent-working-memory-v2")
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-tag", action="append", default=[])
    args = parser.parse_args()
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        parser.error("--stop-after-steps must be positive")
    run_training(args)


if __name__ == "__main__":
    main()
