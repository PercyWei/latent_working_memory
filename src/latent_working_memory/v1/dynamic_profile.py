"""Measure dynamic text steps using the project environment and local logs."""

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
from latent_working_memory.v1.dynamic import (
    DynamicConfig,
    DynamicTrainer,
    load_components,
    runtime_info,
)
from latent_working_memory.v1.squad import SquadDataset
from latent_working_memory.v1.dynamic_data import DynamicTextSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    runtime = runtime_info()
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
            gradient_checkpointing=False,
        ),
    )
    tokenizer, backbone, writer, _ = load_components(checkpoint, device)
    data = SquadDataset(args.index)
    if args.capacity not in recipe.capacities:
        raise ValueError("profile capacity must be in the recipe")
    sampler = DynamicTextSampler(data, recipe, checkpoint.config.write_context_tokens)
    texts = [
        text for ratio in recipe.ratios for text in sampler.pool("train", args.capacity, ratio)
    ]
    cases = {
        "most_updates": sorted(texts, key=lambda text: text.updates, reverse=True),
        "largest_initial": sorted(texts, key=lambda text: text.initial_tokens, reverse=True),
    }
    trainer = DynamicTrainer(backbone, writer, checkpoint.config, recipe, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label, ordered in cases.items():
        episodes = [text.episode(data) for text in ordered[: recipe.batch_size]]
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = trainer.step(
            episodes, tokenizer, [f"profile:{i}" for i in range(len(episodes))], args.capacity
        )
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
        result.update(
            case=label,
            gradient_checkpointing=recipe.gradient_checkpointing,
            layer_checkpointing=False,
            runtime=runtime,
        )
        result["seconds"], result["peak_allocated_bytes"], result["peak_reserved_bytes"] = (
            resources.tolist()
        )
        if rank == 0:
            with args.output.open("a") as log:
                log.write(json.dumps(result) + "\n")
            print(
                json.dumps({k: v for k, v in result.items() if k != "sample_metrics"}), flush=True
            )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
