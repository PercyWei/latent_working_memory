"""训练 worker：读取永久计划，完成训练后提交 summary。"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .. import method
from ..execution.plans import inspect_resume, load_plan, resolve_training
from ..execution.runtime import capture_rng, initialize_device, restore_rng, seed_all
from ..execution.state import checkpoint_file, rewind_metrics, save_checkpoint, write_json
from .cursor import SampleCursor
from .engine import TrainingEngine, evaluate_dev
from .tracking import tracking_run


def _rank_zero(action, rank):
    """写盘或记录失败时，让所有 rank 在相同位置退出。"""
    failure = None
    if rank == 0:
        try:
            action()
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
    status = [failure]
    if dist.is_initialized():
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])


def _save(output_dir, step, config, model, engine, cursor, generator, binding, device, rank, world):
    local_rng = (capture_rng(device), generator.get_state())
    rank_rng = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local_rng, rank_rng, dst=0)
    else:
        rank_rng[0] = local_rng

    def commit():
        save_checkpoint(
            output_dir,
            {
                "config": config,
                "model": method.trainable_state_dict(model),
                "optimizer": engine.optimizer.state_dict(),
                "scheduler": engine.scheduler.state_dict(),
                "cursor": {"step": step, "samples": cursor.state_dict()},
                "rng": [rng for rng, _ in rank_rng],
                "method_rng": [rng for _, rng in rank_rng],
                "binding": binding,
            },
            config["training"]["save_total_limit"],
        )

    _rank_zero(commit, rank)
    return output_dir / "checkpoints" / f"global_step_{step}" / "state.pt"


def train(plan_dir, device="cpu", resume=None):
    device = initialize_device(device)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    requested = load_plan(Path(plan_dir).resolve().parent)
    checkpoint = None
    if resume:
        original, checkpoint = inspect_resume(resume, world)
        if original != requested:
            raise ValueError("resume checkpoint and requested plan differ")
        if any(
            (state["cuda"] is not None) != (device.type == "cuda") for state in checkpoint["rng"]
        ):
            raise ValueError("resume requires the original CPU/CUDA device type")
    seed_all(requested["seed"], device)
    train_samples, train_identity, train_stats = method.load_samples(requested, "train")
    dev_samples, dev_identity, dev_stats = method.load_samples(requested, "dev")
    if not train_samples or not dev_samples:
        raise ValueError("train and dev selections must both contain samples")
    if set(train_stats["raw_document_ids"]) & set(dev_stats["raw_document_ids"]):
        raise ValueError("train and dev documents overlap")
    config = resolve_training(requested, len(train_samples), world)
    training = config["training"]
    output_dir = Path(training["output_dir"])
    binding = {"train": train_identity, "dev": dev_identity}
    if checkpoint is not None and (
        checkpoint["config"] != config or checkpoint["binding"] != binding
    ):
        raise ValueError("resume configuration or selected data differs from the checkpoint")
    model = method.build_model(config, device)
    engine = TrainingEngine(model, config)
    cursor = SampleCursor(len(train_samples), config["seed"])
    generator = torch.Generator().manual_seed(config["seed"] + 1 + rank)
    seed_all(config["seed"] + rank, device)
    global_batch = world * training["micro_batch_size"] * training["gradient_accumulation_steps"]
    step = 0
    if checkpoint is not None:
        step = checkpoint["cursor"]["step"]
        cursor.load_state_dict(checkpoint["cursor"]["samples"], step, global_batch)
        method.load_trainable_state_dict(model, checkpoint["model"])
        engine.optimizer.load_state_dict(checkpoint["optimizer"])
        engine.scheduler.load_state_dict(checkpoint["scheduler"])
        generator.set_state(checkpoint["method_rng"][rank])
        restore_rng(checkpoint["rng"][rank], device)
    del checkpoint
    metrics_path = output_dir / "metrics.jsonl"
    last_checkpoint = checkpoint_file(resume) if resume else None
    with ExitStack() as stack:
        stream = tracking = None

        def setup():
            nonlocal stream, tracking
            output_dir.mkdir(parents=True, exist_ok=True)
            if resume:
                if json.loads((output_dir / "config.json").read_text(encoding="utf-8")) != config:
                    raise ValueError("saved actual configuration differs from the checkpoint")
                rewind_metrics(metrics_path, step)
            elif any(
                (output_dir / name).exists()
                for name in (
                    "config.json",
                    "data.json",
                    "metrics.jsonl",
                    "summary.json",
                    "checkpoints",
                )
            ):
                raise ValueError("training outputs already exist; use resume")
            write_json(output_dir / "config.json", config)
            write_json(
                output_dir / "data.json",
                {
                    "binding": binding,
                    "train": train_stats,
                    "dev": dev_stats,
                    "world_size": world,
                    "effective_batch_size": global_batch,
                },
            )
            tracking = stack.enter_context(
                tracking_run(config, output_dir, bool(resume), device, world)
            )
            stream = stack.enter_context(metrics_path.open("a", encoding="utf-8"))
            print(
                f"train output_dir={output_dir} samples={len(train_samples)} "
                f"world_size={world} global_batch={global_batch} total_steps={training['max_steps']}",
                flush=True,
            )

        _rank_zero(setup, rank)
        while step < training["max_steps"]:
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            indices = cursor.next_batch(global_batch)
            metrics = engine.train_batch(
                [train_samples[index] for index in indices[rank::world]], generator
            )
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            resources = torch.tensor(
                [
                    time.perf_counter() - started,
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0,
                ],
                device=device,
                dtype=torch.float64,
            )
            if world > 1:
                dist.all_reduce(resources, op=dist.ReduceOp.MAX)
            record = {
                "step": step,
                "epoch": cursor.epoch,
                "epoch_end": cursor.epoch_end,
                **{f"train/{name}": value for name, value in metrics.items()},
                "resources/step_seconds": resources[0].item(),
            }
            if device.type == "cuda":
                record["resources/peak_memory_gib"] = resources[1].item()
            if (
                step % training["dev_every"] == 0
                or cursor.epoch_end
                or step == training["max_steps"]
            ):
                rng, method_rng = capture_rng(device), generator.get_state()
                try:
                    dev = evaluate_dev(
                        model, dev_samples[rank::world], training["micro_batch_size"]
                    )
                finally:
                    restore_rng(rng, device)
                    generator.set_state(method_rng)
                record.update({f"dev/{name}": value for name, value in dev.items()})

            def log_step():
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if tracking is not None:
                    rng = capture_rng(device)
                    try:
                        tracking.log(
                            {name: value for name, value in record.items() if "/" in name},
                            step=step,
                        )
                    finally:
                        restore_rng(rng, device)
                print(
                    f"step {step}/{training['max_steps']} loss={metrics['loss']:.6f} "
                    f"samples={metrics['samples']}"
                    + (f" dev={record['dev/loss']:.6f}" if "dev/loss" in record else ""),
                    flush=True,
                )

            _rank_zero(log_step, rank)
            interval = training["save_every"]
            if (
                (interval != "after_each_epoch" and step % interval == 0)
                or cursor.epoch_end
                or step == training["max_steps"]
            ):
                last_checkpoint = _save(
                    output_dir,
                    step,
                    config,
                    model,
                    engine,
                    cursor,
                    generator,
                    binding,
                    device,
                    rank,
                    world,
                )
        summary = {"step": step, "checkpoint": str(last_checkpoint), "output_dir": str(output_dir)}
        _rank_zero(lambda: write_json(output_dir / "summary.json", summary), rank)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        train(args.plan_dir, args.device, args.resume)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
