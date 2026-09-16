"""python -m latent_working_memory.v2.static.train --help"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from latent_working_memory.devices import validate_device
from latent_working_memory.v2.gmsa_checkpoint import checkpoint_stage, load_weights, save_model
from latent_working_memory.v2.gmsa_config import GMSAConfig
from latent_working_memory.v2.gmsa import GMSA
from latent_working_memory.v2.static.data import StaticCollator, StaticDataset


class StaticTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        directory = output_dir or self.args.output_dir
        save_model(self.model, directory, state_dict)
        self.processing_class.save_pretrained(directory)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = self.model if model is None else model
        if checkpoint_stage(resume_from_checkpoint) != target.stage:
            raise ValueError("resume must use a checkpoint from the same stage")
        load_weights(target, resume_from_checkpoint)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return super().prediction_step(
            model, dict(inputs, ratio=self.evaluation_ratio), prediction_loss_only, ignore_keys
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = {}
        for ratio in self.model.model_config.compression_ratios:
            self.evaluation_ratio = ratio
            metrics.update(
                super().evaluate(eval_dataset, ignore_keys, f"{metric_key_prefix}_r{ratio}")
            )
        return metrics


def main():
    parser = argparse.ArgumentParser(description="GMSA 静态 AE／QA 训练，使用根目录环境")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument(
        "--training-config",
        required=True,
        type=Path,
        help="JSON: stage, token limits, and Hugging Face training arguments",
    )
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--eval-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--initialize-from", type=Path, help="AE checkpoint for a new QA run")
    group.add_argument("--resume", type=Path, help="same-run checkpoint-N directory")
    args = parser.parse_args()
    config = GMSAConfig(**json.loads(args.model_config.read_text()))
    settings = json.loads(args.training_config.read_text())
    if set(settings) != {"stage", "max_context_tokens", "max_target_tokens", "trainer"}:
        raise ValueError("training config requires stage, token limits, trainer")
    stage = settings["stage"]
    if stage not in {"autoencoding", "finetune"}:
        raise ValueError("static training supports autoencoding and finetune")
    if stage == "finetune" and not (args.initialize_from or args.resume):
        raise ValueError("QA training requires --initialize-from AE checkpoint or --resume")
    if args.initialize_from and (
        stage != "finetune" or checkpoint_stage(args.initialize_from) != "autoencoding"
    ):
        raise ValueError("stage initialization must be autoencoding -> finetune")
    reserved = {
        "output_dir",
        "report_to",
        "remove_unused_columns",
        "label_names",
        "prediction_loss_only",
        "save_safetensors",
        "load_best_model_at_end",
        "gradient_checkpointing_kwargs",
        "overwrite_output_dir",
        "ddp_find_unused_parameters",
    }
    if reserved.intersection(settings["trainer"]):
        raise ValueError(f"trainer fields managed by v2: {sorted(reserved)}")
    hf_args = TrainingArguments(
        output_dir=str(args.output_dir),
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        prediction_loss_only=True,
        save_safetensors=True,
        load_best_model_at_end=False,
        overwrite_output_dir=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        **settings["trainer"],
    )
    validate_device(hf_args.device)
    if hf_args.fp16:
        raise ValueError("v2 uses BF16 on CUDA or FP32; FP16 is not supported")
    if hf_args.deepspeed or hf_args.fsdp:
        raise ValueError("initial v2 runner supports single device and DDP only")
    run = {
        "model": json.loads(args.model_config.read_text()),
        "training": settings,
        "train_file": str(args.train_file.resolve()),
        "eval_file": str(args.eval_file.resolve()),
        "initialize_from": str(args.initialize_from.resolve()) if args.initialize_from else None,
        "world_size": hf_args.world_size,
    }
    if args.resume:
        if args.resume.resolve().parent != args.output_dir.resolve():
            raise ValueError("resume checkpoint must belong to output-dir")
        previous = json.loads((args.output_dir / "run.json").read_text())
        run["initialize_from"] = previous["initialize_from"]
        if previous != run:
            raise ValueError("resume configuration or data paths differ from the original run")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("new run requires an empty output directory")
    set_seed(hf_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, revision=config.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets = [
        StaticDataset(
            path, tokenizer, stage, settings["max_context_tokens"], settings["max_target_tokens"]
        )
        for path in (args.train_file, args.eval_file)
    ]
    model = GMSA(config, torch.bfloat16 if hf_args.bf16 else torch.float32)
    model.set_stage(stage)
    # Fail on budget problems before performing any optimizer updates.
    minimum_ratio = min(config.compression_ratios)
    for dataset in datasets:
        for row in dataset.rows:
            length = len(row["context_ids"])
            read_length = (length + minimum_ratio - 1) // minimum_ratio
            read_length += len(row["prompt_ids"]) + len(row["labels"])
            if max(length, read_length) > model.max_positions:
                raise ValueError("sample exceeds encoder/decoder positional budget")
    if args.initialize_from:
        load_weights(model, args.initialize_from)
    trainer = StaticTrainer(
        model=model,
        args=hf_args,
        train_dataset=datasets[0],
        eval_dataset=datasets[1],
        data_collator=StaticCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    if trainer.is_world_process_zero():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        if not args.resume:
            provenance = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "upstream": "Twilightaaa/GMSA@2da109e7da39805430e1efebe620f2c5cc6e94c9",
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "git_dirty": bool(
                    subprocess.check_output(["git", "status", "--porcelain"], text=True)
                ),
                "packages": {
                    name: version(name)
                    for name in ("torch", "transformers", "peft", "accelerate", "safetensors")
                },
                "device": str(hf_args.device),
                "resolved_model_revision": model.decoder.config._commit_hash,
            }
            (args.output_dir / "provenance.json").write_text(
                json.dumps(provenance, indent=2) + "\n"
            )
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(args.output_dir / "final"))
    metrics = trainer.evaluate()
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
