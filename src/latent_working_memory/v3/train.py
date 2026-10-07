"""启动器使用的阶段训练 worker；实验入口为 v3.gpu_job。"""

import argparse
from dataclasses import replace
from pathlib import Path

from transformers import set_seed

from latent_working_memory.v3.config import load_experiment
from latent_working_memory.v3.engine import TokenMemoryEngine, initialize_device
from latent_working_memory.v3.objective import TokenMemoryTask
from latent_working_memory.v3.runtime import (
    load_codec,
    load_initialization,
    load_splits,
    make_run,
    read_checkpoint,
    train_loop,
)


def run_training(args):
    config = load_experiment(args.config)
    training = {
        key: str(value)
        for key, value in {
            "dataset_dir": args.dataset_dir,
            "output_dir": args.output_dir,
            "init_checkpoint": args.init_checkpoint,
        }.items()
        if value is not None
    }
    training.update(
        {
            key: value
            for key, value in {
                "max_train_samples": args.max_train_samples,
                "max_dev_samples": args.max_dev_samples,
                "micro_batch_size_per_gpu": args.micro_batch_size_per_gpu,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            }.items()
            if value is not None
        }
    )
    config = replace(
        config,
        training=replace(config.training, **training),
    )
    if args.resume is None and config.objective.stage in {"warmup", "qa", "policy"}:
        if config.training.init_checkpoint is None:
            raise ValueError(
                f"{config.objective.stage} requires --init-checkpoint or training.init_checkpoint"
            )
    device = initialize_device(args.device)
    set_seed(config.training.seed)
    initial_path = args.resume if args.resume is not None else config.training.init_checkpoint
    previous = read_checkpoint(initial_path) if initial_path is not None else None
    model_config = config.model
    if previous is not None and model_config.revision is None:
        resolved = previous["run"]["resolved_model_revision"]
        if resolved is not None:
            model_config = replace(model_config, revision=resolved)
    codec, tokenizer = load_codec(model_config, device)
    initialization = None
    if args.resume is not None:
        initialization = previous["run"]["initialization"]
    elif config.training.init_checkpoint is not None:
        initialization = load_initialization(config.training.init_checkpoint, codec, config)
    splits, statistics = load_splits(config, tokenizer)
    model = TokenMemoryTask(codec, tokenizer, config.objective)
    engine = TokenMemoryEngine(model, config.training, device)
    engine.initialize()
    run = make_run(
        config,
        splits,
        statistics,
        device,
        engine.world_size,
        resolved_model_revision=getattr(
            codec.language_model.get_base_model().config, "_commit_hash", None
        ),
        initialization=initialization,
    )
    return train_loop(
        config, engine, splits, run, resume=args.resume, stop_after_steps=args.stop_after_steps
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--init-checkpoint", "--init", type=Path)
    parser.add_argument("--stop-after-steps", type=int, help="本阶段 optimizer step 上限")
    parser.add_argument("--max-train-samples", type=int, help="内存选取的训练样本上限")
    parser.add_argument("--max-dev-samples", type=int, help="内存选取的开发集样本上限")
    parser.add_argument(
        "--micro-batch-size-per-gpu", type=int, help="每卡一次并行处理的样本/轨迹数"
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, help="每次参数更新累积的 microbatch 数"
    )
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
