from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from latent_working_memory.v1.config import load_config
from latent_working_memory.v1.training import run_p0_training


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent working-memory v1")
    parser.add_argument("--phase", choices=("p0",), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--episode-limit", type=int, default=16)
    parser.add_argument("--dev-episode-limit", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    _validate_device(device)
    config = load_config(args.config)
    result = run_p0_training(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        max_steps=args.max_steps,
        episode_limit=args.episode_limit,
        dev_episode_limit=args.dev_episode_limit,
        save_every=args.save_every,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "final_checkpoint": str(result.final_checkpoint),
                "completed_steps": result.completed_steps,
                "teacher_cache_entries": result.teacher_cache_entries,
                "dev_metrics": result.dev_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select physical GPU 0 or 1")
    physical_devices = tuple(part.strip() for part in visible.split(",") if part.strip())
    if not physical_devices or any(value not in {"0", "1"} for value in physical_devices):
        raise RuntimeError("this project only permits physical GPU 0 and 1")
    logical_index = 0 if device.index is None else device.index
    if logical_index < 0 or logical_index >= len(physical_devices):
        raise RuntimeError("the requested logical CUDA device is not visible")


if __name__ == "__main__":
    main()
