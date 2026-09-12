from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

from latent_working_memory.v1.backbone import load_backbone
from latent_working_memory.v1.checkpoint import load_model_checkpoint
from latent_working_memory.v1.data import EpisodeIndex
from latent_working_memory.v1.evaluation import evaluate_pretraining
from latent_working_memory.data_preparation.fineweb import data_contract
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.devices import validate_device
from latent_working_memory.v1.training import load_trainable_model_state, precision_context
from latent_working_memory.v1.tracking import swanlab_run
from latent_working_memory.v1.reporting import build_evaluation_charts, reconstruction_media


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate FineWeb memory across capacities")
    parser.add_argument("--checkpoint", type=Path, required=True)
    datasets = parser.add_mutually_exclusive_group(required=True)
    datasets.add_argument("--data-dir", type=Path)
    datasets.add_argument("--evaluation-dirs", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run directory and name, e.g. pretrain-random-157k-eval-20260912",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--swanlab-project", default="latent-working-memory")
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-tag", action="append", default=[])
    parser.add_argument("--examples", type=int)
    parser.add_argument("--generation-examples", type=int)
    args = parser.parse_args(argv)
    if args.swanlab_mode != "disabled" and (args.output_dir / "swanlab.json").exists():
        raise ValueError("static report already published; use a new output directory")
    device = torch.device(args.device)
    validate_device(device)
    checkpoint = load_model_checkpoint(args.checkpoint)
    config = checkpoint.config
    evaluation_options = {"eval_generation_every": 1}
    if args.examples is not None:
        evaluation_options["eval_examples"] = args.examples
    if args.generation_examples is not None:
        evaluation_options["eval_generation_examples"] = args.generation_examples
    config = replace(config, **evaluation_options)
    directories = (
        {k: Path(v) for k, v in json.loads(args.evaluation_dirs.read_text()).items()}
        if args.evaluation_dirs
        else {args.split: args.data_dir}
    )
    metadata = {}
    for name, directory in directories.items():
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("evaluation names must be simple directory names")
        metadata[name] = json.loads((directory / "preparation.json").read_text())
        if metadata[name]["contract"] != data_contract(config):
            raise ValueError("evaluation data contract differs from checkpoint")
    tokenizer, backbone = load_backbone(
        config, device, torch.bfloat16 if device.type == "cuda" else torch.float32
    )
    writer = JointMemoryWriter(
        config.d_mem, config.num_layers, config.num_heads, config.ffn_dim, config.k_limit
    ).to(device)
    value = GrowthValueNetwork(config.d_mem).to(device)
    load_trainable_model_state(checkpoint.model_state, backbone, writer, value)
    results = {}
    for name, directory in directories.items():
        destination = args.output_dir / name if args.evaluation_dirs else args.output_dir
        with precision_context(device):
            result = evaluate_pretraining(
                config,
                tokenizer,
                backbone,
                writer,
                EpisodeIndex(directory / f"{args.split}.jsonl"),
                destination,
                checkpoint.progress["next_step"],
                checkpoint.progress["input_tokens"],
                args.split,
            )
        results[name] = result
    with swanlab_run(
        args.output_dir,
        config.to_dict()
        | {
            "evaluation_checkpoint": str(args.checkpoint.resolve()),
            "evaluation_split": args.split,
            "checkpoint_step": checkpoint.progress["next_step"],
            "training_input_tokens": checkpoint.progress["input_tokens"],
            "evaluation_preparations": metadata,
        },
        args.swanlab_mode,
        args.swanlab_project,
        job_type="evaluate",
        group=args.swanlab_group,
        tags=tuple(args.swanlab_tag),
    ) as tracking:
        if tracking is not None:
            charts = build_evaluation_charts(list(results.items()))
            step = checkpoint.progress["next_step"]
            for name in directories:
                destination = args.output_dir / name if args.evaluation_dirs else args.output_dir
                charts.update(
                    reconstruction_media(
                        destination / f"{args.split}-step-{step:06d}.jsonl", f"examples/{name}"
                    )
                )
            tracking.log(charts, step=0)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
