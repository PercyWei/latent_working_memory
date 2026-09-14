from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

from latent_working_memory.v1.backbone import load_backbone
from latent_working_memory.v1.checkpoint import load_model_checkpoint
from latent_working_memory.v1.pretrain.data_selection import select_experiment, selection_metadata
from latent_working_memory.v1.pretrain.prepared_data import pretraining_index, validate_preparation
from latent_working_memory.v1.pretrain.evaluation import evaluate_pretraining
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.devices import validate_device
from latent_working_memory.v1.training import load_trainable_model_state, precision_context
from latent_working_memory.v1.pretrain.tracking import pretraining_run
from latent_working_memory.v1.pretrain.tracking import append_evaluation_reports
from latent_working_memory.v1.tracking import DEFAULT_SWANLAB_PROJECT
from latent_working_memory.v1.pretrain.reporting import (
    build_evaluation_charts,
    reconstruction_media,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate FineWeb memory across capacities")
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--checkpoint", type=Path)
    checkpoints.add_argument("--training-result", type=Path)
    datasets = parser.add_mutually_exclusive_group(required=True)
    datasets.add_argument("--data-dir", type=Path)
    datasets.add_argument("--evaluation-dirs", type=Path)
    datasets.add_argument("--data-selection", type=Path)
    parser.add_argument("--evaluation-source", help="Evaluate one source from the selection configuration")
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
    parser.add_argument("--swanlab-project", default=DEFAULT_SWANLAB_PROJECT)
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-tag", action="append", default=[])
    parser.add_argument("--training-run", type=Path, help="Append to this training directory’s SwanLab run")
    parser.add_argument("--examples", type=int)
    parser.add_argument("--generation-examples", type=int)
    parser.add_argument("--prefix-tokens", type=int, nargs="+", default=())
    args = parser.parse_args(argv)
    if args.training_result:
        result = json.loads(args.training_result.read_text())
        if not result["complete"]:
            raise ValueError("final evaluation requires completed epoch training")
        args.checkpoint = Path(result["final_checkpoint"])
    if args.evaluation_source and not args.data_selection:
        raise ValueError("evaluation-source requires data-selection")
    if args.training_run:
        if args.swanlab_mode != "online":
            raise ValueError("training-run append requires online mode")
        identity = json.loads((args.training_run / "swanlab.json").read_text())
        if identity["job_type"] != "train" or identity["mode"] != "online":
            raise ValueError("evaluation append requires an online training run")
        if args.checkpoint.resolve().parent.parent != args.training_run.resolve():
            raise ValueError("checkpoint must belong to the target training directory")
    if not args.training_run and args.swanlab_mode != "disabled" and (args.output_dir / "swanlab.json").exists():
        raise ValueError("static report already published; use a new output directory")
    device = torch.device(args.device)
    validate_device(device)
    checkpoint = load_model_checkpoint(args.checkpoint)
    if args.training_run and (args.training_run / "evaluation-publications" /
                              f"{args.split}-step-{checkpoint.progress['next_step']:06d}.json").exists():
        raise ValueError("evaluation already appended")
    config = checkpoint.config
    evaluation_options = {"eval_generation_every": 1}
    if args.examples is not None:
        evaluation_options["eval_examples"] = args.examples
    if args.generation_examples is not None:
        evaluation_options["eval_generation_examples"] = args.generation_examples
    config = replace(config, **evaluation_options)
    tokenizer, backbone = load_backbone(
        config, device, torch.bfloat16 if device.type == "cuda" else torch.float32
    )
    indices, metadata = {}, {}
    multiple = args.evaluation_dirs is not None or (args.data_selection is not None and args.evaluation_source is None)
    if args.data_selection:
        spec = json.loads(args.data_selection.read_text())
        selected, report = select_experiment(spec, config, tokenizer, splits=(args.split,))
        names = [args.evaluation_source] if args.evaluation_source else list(spec["sources"])
        if any(name not in spec["sources"] for name in names):
            raise ValueError("unknown evaluation source")
        for name in names:
            indices[name] = selected[name, args.split]
            metadata[name] = selection_metadata(report, name)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "data-selection.json").write_text(json.dumps(spec, indent=2) + "\n")
    else:
        directories = ({k: Path(v) for k, v in json.loads(args.evaluation_dirs.read_text()).items()}
                       if args.evaluation_dirs else {args.split: args.data_dir})
        for name, directory in directories.items():
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError("evaluation names must be simple directory names")
            metadata[name] = json.loads((directory / "preparation.json").read_text())
            validate_preparation(metadata[name], config)
            indices[name] = pretraining_index(directory / f"{args.split}.jsonl", tokenizer, config)
    writer = JointMemoryWriter(
        config.d_mem, config.num_layers, config.num_heads, config.ffn_dim, config.k_limit
    ).to(device)
    value = GrowthValueNetwork(config.d_mem).to(device)
    load_trainable_model_state(checkpoint.model_state, backbone, writer, value)
    results = {}
    for name, index in indices.items():
        destination = args.output_dir / name if multiple else args.output_dir
        with precision_context(device):
            result = evaluate_pretraining(
                config,
                tokenizer,
                backbone,
                writer,
                index,
                destination,
                checkpoint.progress["next_step"],
                checkpoint.progress["input_tokens"],
                args.split,
                tuple(args.prefix_tokens),
            )
        results[name] = result
    if args.training_run:
        step = checkpoint.progress["next_step"]
        entries = [
            {"evaluation_source": name,
             "report": str(((args.output_dir / name if multiple else args.output_dir)
                            / f"{args.split}-step-{step:06d}.json").resolve())}
            for name in indices
        ]
        append_evaluation_reports(args.training_run, entries)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    with pretraining_run(
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
            for name in indices:
                destination = args.output_dir / name if multiple else args.output_dir
                charts.update(
                    reconstruction_media(
                        destination / f"{args.split}-step-{step:06d}.jsonl", f"examples/{name}"
                    )
                )
            tracking.log(charts, step=0)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
