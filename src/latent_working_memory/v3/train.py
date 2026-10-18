"""同一方法的连续阶段训练 worker；实验入口为 v3.gpu_job。"""

import argparse
from contextlib import ExitStack
from dataclasses import replace
import gc
from pathlib import Path

import torch.distributed as dist
from transformers import set_seed

from latent_working_memory.v3.config import (
    ExperimentConfig,
    load_common_config,
    load_training_plan,
    load_training_stages,
    parse_optional_positive_integer,
    validate_stage_sequence,
)
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
from latent_working_memory.v3.tracking import (
    configure_posttraining_panel,
    experiment_directory,
    update_method_tracking,
)
from latent_working_memory.tracking_credentials import swanlab_api_key


def training_configs(args):
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
                "micro_batch_size_per_gpu": args.micro_batch_size_per_gpu,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            }.items()
            if value is not None
        }
    )
    if "max_dev_samples" in vars(args):
        training["max_dev_samples"] = args.max_dev_samples
    sample_limits = {
        name: getattr(args, name)
        for name in ("max_pretrain_train_samples", "max_qa_train_samples")
        if name in vars(args)
    }
    if args.resume is not None and (training or sample_limits):
        options = sorted(name.replace("_", "-") for name in {*training, *sample_limits})
        raise ValueError(
            "--resume does not accept configuration overrides: "
            + ", ".join(f"--{name}" for name in options)
        )
    if args.resume is None:
        configs = load_training_plan(args.config, args.train_config, args.stages)
    else:
        current = ExperimentConfig.from_dict(read_checkpoint(args.resume)["run"]["config"])
        if load_common_config(args.config)["seed"] != current.training.seed:
            raise ValueError("common seed differs from the saved training seed")
        if args.stages[0] != current.objective.stage:
            raise ValueError("resume checkpoint must belong to the first selected stage")
        later = (
            load_training_stages(
                args.train_config, current.model, current.training.seed, args.stages[1:]
            )
            if len(args.stages) > 1
            else ()
        )
        configs = (current, *later)
    if len(configs) > 1 and any(
        name in training for name in ("dataset_dir", "output_dir", "init_checkpoint")
    ):
        raise ValueError("multi-stage training takes data, outputs and initialization from configs")
    resolved = []
    for config in configs:
        sample_limit_name = (
            "max_pretrain_train_samples"
            if config.objective.stage in {"pretrain", "lm"}
            else "max_qa_train_samples"
        )
        stage_training = dict(training)
        if sample_limit_name in sample_limits:
            stage_training[sample_limit_name] = sample_limits[sample_limit_name]
        resolved.append(replace(config, training=replace(config.training, **stage_training)))
    configs = resolved
    if any(config.training.output_dir is None for config in configs):
        raise ValueError(
            "training requires a resolved output_dir; start experiments with v3.gpu_job "
            "or supply a resolved training plan"
        )
    first = configs[0]
    validate_stage_sequence(first.objective.method, [config.objective.stage for config in configs])
    if len(configs) > 1:
        for previous, config in zip(configs[:-1], configs[1:], strict=True):
            if (
                config.model != first.model
                or config.objective.method != first.objective.method
                or (
                    previous.objective.stage != "pretrain"
                    and config.objective.writer_mode != previous.objective.writer_mode
                )
                or (
                    previous.objective.stage != "pretrain"
                    and previous.objective.writer_mode == "tag"
                    and config.objective.tag_tokens != previous.objective.tag_tokens
                )
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
    codec, tokenizer = load_codec(config.model, device, config.objective)
    initialization = None
    if args.resume is not None:
        initialization = read_checkpoint(args.resume)["run"]["initialization"]
    elif config.training.init_checkpoint is not None:
        initialization = load_initialization(config.training.init_checkpoint, codec, config)
    model = TokenMemoryTask(codec, tokenizer, config.objective, config.training.seed)
    engine = TokenMemoryEngine(model, config.training, device)
    engine.initialize()
    result, run, tracking = None, None, None
    with ExitStack() as stack:
        for index, config in enumerate(configs):
            if index:
                config = replace(
                    config,
                    training=replace(config.training, init_checkpoint=result["checkpoint"]),
                )
                set_seed(config.training.seed)
                if config.objective.writer_mode != model.cfg.writer_mode:
                    # 写入器会改变参数和 attention 结构，按已有初始化契约重建一次。
                    del engine, model, codec, tokenizer
                    gc.collect()
                    codec, tokenizer = load_codec(config.model, device, config.objective)
                    initialization = load_initialization(result["checkpoint"], codec, config)
                    model = TokenMemoryTask(codec, tokenizer, config.objective, config.training.seed)
                    engine = TokenMemoryEngine(model, config.training, device)
                    engine.initialize()
                else:
                    initialization = initialization_record(
                        result["checkpoint"], run, result["completed_steps"], config
                    )
                    model.cfg = config.objective
                    model.seed = config.training.seed
                    engine.reset_optimizer(config.training)
            splits, statistics = load_splits(config, tokenizer)
            run = make_run(
                config,
                splits,
                statistics,
                device,
                engine.world_size,
                initialization=initialization,
            )
            if (
                index
                and engine.rank == 0
                and tracking is not None
                and initialization["stage"] == "pretrain"
            ):
                configure_posttraining_panel(config, tracking, device, swanlab_api_key())
            stage = prepare_training(
                config,
                engine,
                splits,
                run,
                resume=args.resume if index == 0 else None,
                max_steps=args.max_steps,
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
                max_steps=args.max_steps,
                tracking=tracking,
            )
            # 下一阶段读取来源记录前，确保 rank 0 已写完结果与阶段指标。
            if engine.world_size > 1:
                dist.barrier()
            del splits, statistics, stage
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--init-checkpoint", "--init", type=Path)
    parser.add_argument(
        "--max-steps",
        type=parse_optional_positive_integer,
        help="本阶段 optimizer step 上限：正整数或 null（完整训练），默认完整训练",
    )
    for category, stages in (("pretrain", "pretrain/lm"), ("qa", "qa/warmup/policy")):
        parser.add_argument(
            f"--max-{category}-train-samples",
            type=parse_optional_positive_integer,
            default=argparse.SUPPRESS,
            help=f"{stages} 训练样本上限：正整数或 null（不限）；未传时沿用配置",
        )
    parser.add_argument(
        "--max-dev-samples",
        type=parse_optional_positive_integer,
        default=argparse.SUPPRESS,
        help="开发集样本上限：正整数或 null（不限）；未传时沿用配置",
    )
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
