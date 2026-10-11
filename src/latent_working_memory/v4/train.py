"""python -m latent_working_memory.v4.train --config experiment.json --device cpu"""

import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, set_seed

from latent_working_memory.runtime import initialize_device
from latent_working_memory.v4.checkpoint import MetaMemoryCheckpointHandler, read_checkpoint
from latent_working_memory.v4.config import load_experiment
from latent_working_memory.v4.data import (
    dataset_identity,
    load_tokenized_episodes,
)
from latent_working_memory.v4.engine import EngineRegistry
from latent_working_memory.v4.model import StreamingMemoryLM
from latent_working_memory.v4.tracking import configure_metrics, log_metrics, training_tracking
from latent_working_memory.v4.training_data import make_data_loader


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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
    loader = make_data_loader(
        episodes, engine.config.global_batch_size, engine.config.seed, shuffle=False
    )
    for batch in loader:
        metrics = engine.eval_batch(batch)
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
    else:
        error = None
        if rank == 0 and output.exists() and any(output.iterdir()):
            error = "new training requires an empty output directory"
        if world_size > 1:
            status = [error]
            dist.broadcast_object_list(status, src=0)
            error = status[0]
        if error is not None:
            raise ValueError(error)

    set_seed(config.seed)
    load_model_config = model_config
    if resume is not None:
        previous = read_checkpoint(resume)
        resolved = previous["run"]["resolved_model_revision"]
        if resolved is not None:
            load_model_config = replace(model_config, revision=resolved)
        tokenizer = AutoTokenizer.from_pretrained(output / "tokenizer")
        del previous
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path, revision=model_config.revision
        )
    min_length = model_config.pending_size + model_config.recent_size + 2
    train = load_tokenized_episodes(config.train_file, tokenizer, min_length, config.max_seq_length)
    dev = load_tokenized_episodes(config.dev_file, tokenizer, min_length, config.max_seq_length)
    model = StreamingMemoryLM(load_model_config).to(device)
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
        "training": {**asdict(config), "tags": list(config.tags)},
        "data": {"train": dataset_identity(train), "dev": dataset_identity(dev)},
        "world_size": world_size,
        "device_type": device.type,
    }
    cursor = {"epoch": 0, "sample_offset": 0, "step": 0, "sample_visits": 0, "target_tokens": 0}
    metrics_path = output / "metrics.jsonl"
    loader = make_data_loader(train, config.global_batch_size, config.seed)
    handler = MetaMemoryCheckpointHandler(
        engine, loader, output / "checkpoints", run, cursor, resume_from_path=resume
    )
    if resume is not None:
        if json.loads((output / "run.json").read_text(encoding="utf-8")) != run:
            raise ValueError("saved run does not match the resume configuration or tokenized data")
        restored_step = handler.load_checkpoint()
        validate_cursor(cursor, config, len(train))
        if restored_step != cursor["step"]:
            raise ValueError("checkpoint directory step differs from its saved cursor")
        if rank == 0:
            rewind_metrics(metrics_path, cursor["step"])
    elif rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "run.json", run)
        tokenizer.save_pretrained(output / "tokenizer")
        metrics_path.touch()
    if world_size > 1:
        dist.barrier()

    with ExitStack() as stack:
        context = (
            training_tracking(config, run, device, resume=resume is not None)
            if rank == 0
            else nullcontext(None)
        )
        tracking, error = None, None
        try:
            tracking = stack.enter_context(context)
            if tracking is not None:
                configure_metrics(tracking, device)
        except Exception as exception:
            error = f"tracking initialization failed: {exception}"
        if world_size > 1:
            status = [error]
            dist.broadcast_object_list(status, src=0)
            error = status[0]
        if error is not None:
            raise RuntimeError(error)
        return train_loop(
            engine, config, train, dev, loader, handler, cursor, resume,
            args.stop_after_steps, tracking,
        )


def train_loop(engine, config, train, dev, loader, handler, cursor, checkpoint_path, stop, tracking):
    output = Path(config.output_dir)
    metrics_path = output / "metrics.jsonl"
    total_steps = config.epochs * math.ceil(len(train) / config.global_batch_size)
    for epoch in range(cursor["epoch"], config.epochs):
        if stop is not None and cursor["step"] >= stop:
            break
        loader.sampler.set_epoch(epoch)
        for batch in loader:
            if engine.device.type == "cuda":
                torch.cuda.synchronize(engine.device)
                torch.cuda.reset_peak_memory_stats(engine.device)
            started = time.perf_counter()
            metrics = engine.step(batch)
            if engine.device.type == "cuda":
                torch.cuda.synchronize(engine.device)
            resources = {"optimizer_step_seconds": time.perf_counter() - started}
            if engine.device.type == "cuda":
                resources["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
                    engine.device
                )
            if engine.world_size > 1:
                maximum = torch.tensor(
                    list(resources.values()), device=engine.device, dtype=torch.float64
                )
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                resources = dict(zip(resources, maximum.tolist(), strict=True))
            cursor["step"] += 1
            cursor["sample_visits"] += len(batch)
            cursor["target_tokens"] += metrics["target_tokens"]
            consumed = cursor["sample_offset"] + len(batch)
            epoch_end = consumed == len(train)
            cursor["epoch"] = epoch + int(epoch_end)
            cursor["sample_offset"] = 0 if epoch_end else consumed
            step = cursor["step"]
            stopping = stop is not None and step >= stop
            record = {
                "step": step,
                "epoch": epoch + 1,
                "train/epoch": cursor["sample_visits"] / len(train),
                **{f"train/{k}": v for k, v in metrics.items()},
                **{f"resources/{k}": v for k, v in resources.items()},
            }
            if step % config.eval_every == 0 or epoch_end or stopping:
                development = evaluate(engine, dev)
                record.update({f"dev/{k}": v for k, v in development.items()})
            if engine.rank == 0:
                with metrics_path.open("a", encoding="utf-8") as stream:
                    text = json.dumps(record, ensure_ascii=False, allow_nan=False)
                    stream.write(text + "\n")
                    print(text, flush=True)
                if tracking is not None:
                    log_metrics(tracking, record)
            if step % config.save_every == 0 or epoch_end or stopping:
                handler.save_checkpoint(step)
                checkpoint_path = output / "checkpoints" / f"global_step_{step}"
            if stopping:
                break
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
    if engine.rank == 0:
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
