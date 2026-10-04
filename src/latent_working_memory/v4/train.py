"""python -m latent_working_memory.v4.train --config experiment.json --device cpu"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch.distributed as dist
from transformers import AutoTokenizer, set_seed

from latent_working_memory.v4.checkpoint import load_checkpoint, save_checkpoint
from latent_working_memory.v4.config import load_experiment
from latent_working_memory.v4.data import (
    dataset_identity,
    episode_order,
    load_tokenized_episodes,
)
from latent_working_memory.v4.engine import EngineRegistry, initialize_device
from latent_working_memory.v4.model import StreamingMemoryLM


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def validate_cursor(cursor, config, samples):
    names = {"epoch", "sample_offset", "step", "sample_visits", "target_tokens"}
    if set(cursor) != names or any(
        type(value) is not int or value < 0 for value in cursor.values()
    ):
        raise ValueError("checkpoint has an invalid training cursor")
    epoch, offset = cursor["epoch"], cursor["sample_offset"]
    if (
        epoch > config.epochs
        or offset >= samples
        or offset % config.global_batch_size
        or (epoch == config.epochs and offset != 0)
        or cursor["step"]
        != epoch * math.ceil(samples / config.global_batch_size)
        + offset // config.global_batch_size
        or cursor["sample_visits"] != epoch * samples + offset
    ):
        raise ValueError("checkpoint cursor does not match the epoch/batch schedule")


def rewind_metrics(path, step):
    """崩溃后丢弃晚于恢复点的记录，避免把未恢复的更新混入曲线。"""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    retained = [record for record in records if record["step"] <= step]
    training_steps = [record["step"] for record in retained if "train/loss" in record]
    if training_steps != list(range(1, step + 1)):
        raise ValueError("training metrics do not contain exactly the checkpoint's completed steps")
    if len(retained) != len(records):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                for record in retained
            ),
            encoding="utf-8",
        )
        temporary.replace(path)


def evaluate(engine, episodes):
    totals = {"target_tokens": 0, "samples": 0, "source_tokens": 0, "write_events": 0}
    nll, write_loss = 0.0, 0.0
    for start in range(0, len(episodes), engine.config.global_batch_size):
        metrics = engine.eval_batch(episodes[start : start + engine.config.global_batch_size])
        nll += metrics["loss"] * metrics["target_tokens"]
        write_loss += metrics["write_loss"] * metrics["write_events"]
        for name in totals:
            totals[name] += metrics[name]
    return {
        **totals,
        "loss": nll / totals["target_tokens"],
        "write_loss": write_loss / totals["write_events"],
    }


def run_training(args):
    model_config, config = load_experiment(args.config)
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        raise ValueError("stop_after_steps must be positive")
    device = initialize_device(args.device)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    output = Path(config.output_dir)
    resume = Path(args.resume).resolve() if args.resume is not None else None
    if resume is not None:
        if resume.parent != output / "checkpoints":
            raise ValueError("resume checkpoint must belong to output_dir/checkpoints")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("new training requires an empty output directory")

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, revision=model_config.revision
    )
    min_length = model_config.pending_size + model_config.recent_size + 2
    train = load_tokenized_episodes(config.train_file, tokenizer, min_length, config.max_seq_length)
    dev = load_tokenized_episodes(config.dev_file, tokenizer, min_length, config.max_seq_length)
    model = StreamingMemoryLM(model_config).to(device)
    position_limit = model.backbone.config.max_position_embeddings
    if any(len(episode.input_ids) > position_limit for episode in (*train, *dev)):
        raise ValueError("training/dev episode exceeds the backbone max_position_embeddings")
    engine = EngineRegistry.new(
        model_type="lwm_v4_meta", backend="replicated", model=model, config=config, device=device
    )
    engine.initialize()
    run = {
        "model": asdict(model_config),
        "resolved_model_revision": getattr(model.backbone.config, "_commit_hash", None),
        "training": asdict(config),
        "data": {"train": dataset_identity(train), "dev": dataset_identity(dev)},
        "world_size": world_size,
        "device_type": device.type,
    }
    cursor = {"epoch": 0, "sample_offset": 0, "step": 0, "sample_visits": 0, "target_tokens": 0}
    metrics_path = output / "metrics.jsonl"
    if resume is not None:
        if json.loads((output / "run.json").read_text(encoding="utf-8")) != run:
            raise ValueError("saved run does not match the resume configuration or tokenized data")
        cursor = load_checkpoint(resume, engine, run)
        validate_cursor(cursor, config, len(train))
        if rank == 0:
            rewind_metrics(metrics_path, cursor["step"])
    elif rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "run.json", run)
        tokenizer.save_pretrained(output / "tokenizer")
        metrics_path.touch()
    if world_size > 1:
        dist.barrier()

    total_steps = config.epochs * math.ceil(len(train) / config.global_batch_size)
    checkpoint_path = resume
    stop = args.stop_after_steps
    for epoch in range(cursor["epoch"], config.epochs):
        ordered = episode_order(train, config.seed, epoch)
        for start in range(cursor["sample_offset"], len(ordered), config.global_batch_size):
            if stop is not None and cursor["step"] >= stop:
                break
            batch = ordered[start : start + config.global_batch_size]
            metrics = engine.step(batch)
            cursor["step"] += 1
            cursor["sample_visits"] += len(batch)
            cursor["target_tokens"] += metrics["target_tokens"]
            epoch_end = start + len(batch) == len(ordered)
            cursor["epoch"] = epoch + int(epoch_end)
            cursor["sample_offset"] = 0 if epoch_end else start + len(batch)
            step = cursor["step"]
            records = [
                {"step": step, "epoch": epoch + 1, **{f"train/{k}": v for k, v in metrics.items()}}
            ]
            if step % config.eval_every == 0 or epoch_end:
                development = evaluate(engine, dev)
                records.append({"step": step, **{f"dev/{k}": v for k, v in development.items()}})
            if rank == 0:
                with metrics_path.open("a", encoding="utf-8") as stream:
                    for record in records:
                        text = json.dumps(record, ensure_ascii=False, allow_nan=False)
                        stream.write(text + "\n")
                        print(text, flush=True)
            if step % config.save_every == 0 or epoch_end or (stop is not None and step >= stop):
                checkpoint_path = output / "checkpoints" / f"step-{step:06d}.pt"
                save_checkpoint(checkpoint_path, engine, run, cursor)
        if stop is not None and cursor["step"] >= stop:
            break
    result = {
        "complete": cursor["epoch"] == config.epochs,
        "completed_steps": cursor["step"],
        "total_steps": total_steps,
        "completed_epochs": cursor["epoch"],
        "sample_visits": cursor["sample_visits"],
        "target_tokens": cursor["target_tokens"],
        "checkpoint": str(checkpoint_path),
    }
    if rank == 0:
        write_json(output / "training-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="v4 局部 attention-matching meta-training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-steps", type=int, help="停止于指定的全局 optimizer step")
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
