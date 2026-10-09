"""同一方法的连续阶段训练 worker；实验入口为 v3.gpu_job。"""

import argparse
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import torch.distributed as dist
from transformers import set_seed

from latent_working_memory.v3.config import load_experiment, validate_stage_sequence
from latent_working_memory.v3.engine import TokenMemoryEngine, initialize_device
from latent_working_memory.v3.objective import TokenMemoryTask
from latent_working_memory.v3.runtime import (
    _tracking_context,
    initialization_record,
    load_codec,
    load_initialization,
    load_splits,
    make_run,
    prepare_training,
    read_checkpoint,
    train_loop,
)
from latent_working_memory.v3.tracking import experiment_directory, update_method_tracking


def training_configs(args):
    configs = [load_experiment(path) for path in args.config]
    if len(configs) > 1 and args.max_train_samples is not None:
        raise ValueError(
            "multi-stage training requires a separate max_train_samples in each config"
        )
    if len(configs) > 1 and any(
        value is not None for value in (args.dataset_dir, args.output_dir, args.init_checkpoint)
    ):
        raise ValueError("multi-stage training takes data, outputs and initialization from configs")
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
    configs = [replace(config, training=replace(config.training, **training)) for config in configs]
    if any(config.training.output_dir is None for config in configs):
        raise ValueError(
            "training requires a resolved output_dir; start experiments with v3.gpu_job "
            "or resume using the saved stage config.json"
        )
    first = configs[0]
    validate_stage_sequence(first.objective.method, [config.objective.stage for config in configs])
    if len(configs) > 1:
        for config in configs[1:]:
            if (
                config.model != first.model
                or config.objective.method != first.objective.method
                or (
                    first.objective.method == "icae_multi"
                    and (
                        config.objective.icae_min_segments != first.objective.icae_min_segments
                        or config.objective.icae_max_segments != first.objective.icae_max_segments
                    )
                )
                or experiment_directory(config.training) != experiment_directory(first.training)
                or any(
                    getattr(config.training, name) != getattr(first.training, name)
                    for name in (
                        "experiment_id",
                        "experiment_name",
                        "swanlab_project",
                        "group",
                        "tags",
                    )
                )
            ):
                raise ValueError("continuous stages must share their model and method identity")
            if config.training.init_checkpoint is not None:
                raise ValueError("later stages inherit the preceding stage automatically")
    return configs


def run_training(args):
    configs = training_configs(args)
    config = configs[0]
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
    codec, tokenizer = load_codec(model_config, device, config.objective)
    initialization = None
    if args.resume is not None:
        initialization = previous["run"]["initialization"]
    elif config.training.init_checkpoint is not None:
        initialization = load_initialization(config.training.init_checkpoint, codec, config)
    del previous
    model = TokenMemoryTask(codec, tokenizer, config.objective)
    engine = TokenMemoryEngine(model, config.training, device)
    engine.initialize()
    result, run = None, None
    with ExitStack() as stack:
        for index, config in enumerate(configs):
            if index:
                config = replace(
                    config,
                    training=replace(config.training, init_checkpoint=result["checkpoint"]),
                )
                initialization = initialization_record(
                    result["checkpoint"], run, result["completed_steps"], config
                )
                # 各阶段使用自身 seed；不再通过重建模型间接消耗随机数。
                set_seed(config.training.seed)
                model.cfg = config.objective
                engine.reset_optimizer(config.training)
            splits, statistics = load_splits(config, tokenizer, codec.max_positions)
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
            stage = prepare_training(
                config,
                engine,
                splits,
                run,
                resume=args.resume if index == 0 else None,
                stop_after_steps=args.stop_after_steps,
            )
            if index == 0:
                tracking = stack.enter_context(_tracking_context(config, engine, run))
            elif engine.rank == 0:
                update_method_tracking(config, run, tracking)
            if engine.rank == 0:
                print(f"Starting {config.objective.method}/{config.objective.stage}", flush=True)
            result = train_loop(
                config,
                engine,
                splits,
                run,
                stage,
                stop_after_steps=args.stop_after_steps,
                tracking=tracking,
            )
            # 下一阶段读取来源记录前，确保 rank 0 已写完结果与阶段指标。
            if engine.world_size > 1:
                dist.barrier()
            del splits, statistics
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, nargs="+", required=True)
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
    return parser.parse_args(argv)


def main():
    run_training(parse_args())


if __name__ == "__main__":
    main()
