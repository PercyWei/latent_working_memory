from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
from accelerate.state import PartialState

from latent_working_memory.v1.config import load_config
from latent_working_memory.v1.pretrain.training import run_pretraining
from latent_working_memory.v1.tracking import DEFAULT_SWANLAB_PROJECT


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent working-memory v1")
    parser.add_argument("--phase", choices=("pretrain",), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer-workers", type=int, default=4)
    parser.add_argument("--tokenization-batch-size", type=int, default=256)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max-samples-per-epoch", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--swanlab-project", default=DEFAULT_SWANLAB_PROJECT)
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-tag", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    config = load_config(args.config)
    result = run_pretraining(
        config=config,
        data_selection=args.data_selection,
        output_dir=args.output_dir,
        device=device,
        epochs=args.epochs,
        tokenizer_workers=args.tokenizer_workers,
        tokenization_batch_size=args.tokenization_batch_size,
        prefetch_batches=args.prefetch_batches,
        max_samples_per_epoch=args.max_samples_per_epoch,
        stop_after_steps=args.stop_after_steps,
        save_every=args.save_every,
        resume=args.resume,
        swanlab_mode=args.swanlab_mode,
        swanlab_project=args.swanlab_project,
        swanlab_group=args.swanlab_group,
        swanlab_tags=tuple(args.swanlab_tag),
    )
    if not PartialState().is_main_process:
        PartialState().destroy_process_group()
        return
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

    PartialState().destroy_process_group()


if __name__ == "__main__":
    main()
