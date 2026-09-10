from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from latent_working_memory.devices import validate_device
from latent_working_memory.v1.config import load_config
from latent_working_memory.v1.training import run_pretraining


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent working-memory v1")
    parser.add_argument("--phase", choices=("pretrain",), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--train-example-limit", type=int)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--swanlab-project", default="latent-working-memory")
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-tag", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    validate_device(device)
    config = load_config(args.config)
    result = run_pretraining(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        max_steps=args.max_steps,
        train_example_limit=args.train_example_limit,
        save_every=args.save_every,
        resume=args.resume,
        swanlab_mode=args.swanlab_mode,
        swanlab_project=args.swanlab_project,
        swanlab_group=args.swanlab_group,
        swanlab_tags=tuple(args.swanlab_tag),
    )
    print(
        json.dumps(
            {
                "final_checkpoint": str(result.final_checkpoint),
                "completed_steps": result.completed_steps,
                "dev_metrics": result.dev_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
