"""Measure real-article dynamic steps without experiment tracking or checkpoints."""

from dataclasses import replace
from datetime import timedelta
import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from latent_working_memory.devices import validate_device
from latent_working_memory.v1.checkpoint import load_model_checkpoint
from latent_working_memory.v1.dynamic import DynamicConfig, DynamicTrainer, load_components
from latent_working_memory.v1.squad import SquadDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    validate_device(device)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    recipe = DynamicConfig(**json.loads(args.recipe.read_text()))
    if args.gradient_checkpointing is not None:
        recipe = replace(recipe, gradient_checkpointing=args.gradient_checkpointing)
    checkpoint = load_model_checkpoint(args.checkpoint)
    checkpoint = replace(
        checkpoint,
        config=replace(
            checkpoint.config,
            gradient_checkpointing=recipe.gradient_checkpointing,
        ),
    )
    tokenizer, backbone, writer, _ = load_components(checkpoint, device)
    data = SquadDataset(args.index)
    docs = data.select("train", recipe.min_tokens, recipe.max_tokens)
    # Many paragraphs stress full BPTT; long paragraphs stress individual writes.
    cases = {
        "most_updates": sorted(
            docs, key=lambda d: len(data.records[d]["paragraph_tokens"]), reverse=True
        ),
        "largest_paragraph": sorted(
            docs, key=lambda d: max(data.records[d]["paragraph_tokens"]), reverse=True
        ),
    }
    trainer = DynamicTrainer(backbone, writer, checkpoint.config, recipe, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label, ordered in cases.items():
        episodes = [data.episode(d) for d in ordered[: recipe.gradient_accumulation_steps]]
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = trainer.step(episodes, tokenizer, [f"profile:{i}" for i in range(len(episodes))])
        torch.cuda.synchronize(device)
        resources = torch.tensor(
            [
                time.perf_counter() - start,
                torch.cuda.max_memory_allocated(device),
                torch.cuda.max_memory_reserved(device),
            ],
            dtype=torch.float64,
            device=device,
        )
        if world > 1:
            dist.all_reduce(resources, op=dist.ReduceOp.MAX)
        result.update(case=label, gradient_checkpointing=recipe.gradient_checkpointing)
        result["seconds"], result["peak_allocated_bytes"], result["peak_reserved_bytes"] = (
            resources.tolist()
        )
        if rank == 0:
            with args.output.open("a") as log:
                log.write(json.dumps(result) + "\n")
            print(
                json.dumps({k: v for k, v in result.items() if k != "article_metrics"}), flush=True
            )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
