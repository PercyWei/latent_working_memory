"""训练参数解析与显式配置覆盖。"""

import argparse

from ..cli import (
    boolean, ignore_resume_options, nonnegative_int, optional_positive_int, positive_int,
    save_frequency, unit_interval,
)


def parse_run_args(argv=None):
    parser = argparse.ArgumentParser(
        description="v3.1.1 预训练、后训练与恢复。",
        argument_default=argparse.SUPPRESS,
    )
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true", help="恢复已有训练计划")
    restart.add_argument("--overwrite", action="store_true", help="清理目标 run 后重建")
    parser.add_argument("--config", help="新训练必填，从 JSON 配置读取 stage；恢复从 checkpoint 读取")
    parser.add_argument("--output-root", help="实验产物父目录")
    parser.add_argument("--run-id", help=(
        "运行目录名，显式值原样使用；预训练默认 pretrain_k{memory_slots}_{YYYYMMDD-HHMM}，"
        "压缩器后训练默认 compressor-post_k{memory_slots}_{YYYYMMDD-HHMM}，"
        "边界／联合后训练默认 policy-post_k{memory_slots}_{YYYYMMDD-HHMM}，时间采用 Asia/Shanghai"
    ))
    parser.add_argument("--checkpoint", help="训练恢复用的单个 checkpoint 路径")
    parser.add_argument("--init-checkpoint", help="后训练的压缩器来源 checkpoint，可来自预训练或 compressor_only 后训练")
    parser.add_argument("--model-path")
    parser.add_argument("--seed", type=nonnegative_int)
    parser.add_argument("--train-data")
    parser.add_argument("--length-curriculum", action="store_true",
                        help="按离线长度档位从短到长逐阶段训练，并补充历史数据，默认关闭")
    for name in ("max-train-data-tokens", "max-train-samples", "max-dev-samples"):
        parser.add_argument(f"--{name}", type=optional_positive_int, help="null 取消限制")
    parser.add_argument("--epochs", type=positive_int, help="普通训练轮数，默认 1；长度课程由每阶段轮数控制")
    parser.add_argument("--curriculum-stage-epochs", type=positive_int,
                        help="长度课程每阶段完整遍历本档数据的轮数，默认 1")
    parser.add_argument("--curriculum-replay-ratio", type=unit_interval,
                        help="长度课程历史补充数量相对本档样本数的比例，默认 0.2")
    parser.add_argument("--micro-batch-size-per-gpu", type=positive_int)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"),
                        help="基座参数精度，省略时沿用 JSON")
    parser.add_argument("--attention-implementation", choices=("sdpa", "flash_attention_2"),
                        help="attention 实现，省略时沿用 JSON")
    for name, description in (
        ("gradient-checkpointing", "Transformer 层梯度检查点"),
        ("mixed-precision", "FP32 参数配合标准 BF16 autocast"),
        ("save-on-cpu", "将反向所需的保存张量暂存到 CPU"),
    ):
        parser.add_argument(
            f"--{name}", type=boolean, nargs="?", const=True, metavar="{true,false}",
            help=f"{description}；仅传参数名表示开启，false 关闭，省略时沿用 JSON",
        )
    parser.add_argument("--dev-every", type=save_frequency)
    parser.add_argument("--save-every", type=save_frequency)
    parser.add_argument("--save-total-limit", type=positive_int)
    parser.add_argument("--tracking", choices=("online", "offline"),
                        help="online 开启 SwanLab，offline 仅保留本地产物")
    args = parser.parse_args(argv)
    if getattr(args, "resume", False):
        ignore_resume_options(args, {"resume", "overwrite", "checkpoint"})
    supplied = set(vars(args))
    args.resume = getattr(args, "resume", False)
    args.checkpoint = getattr(args, "checkpoint", None)
    args.overwrite = getattr(args, "overwrite", False)
    if args.resume:
        if args.checkpoint is None:
            parser.error("--resume requires --checkpoint")
    else:
        if "config" not in supplied:
            parser.error("new training requires --config")
        if args.checkpoint is not None:
            parser.error("--checkpoint requires --resume")
    return args


def config_overrides(args):
    """仅收集实际提供的三层覆盖项，保留 nullable 字段的显式 null。"""
    values = vars(args)
    overrides = {}
    if "seed" in values:
        overrides["seed"] = values["seed"]
    if "model_path" in values:
        overrides["model_name_or_path"] = values["model_path"]
    compute = {name: values[name] for name in (
        "dtype", "attention_implementation", "gradient_checkpointing", "mixed_precision", "save_on_cpu",
    ) if name in values}
    if compute:
        overrides["compute"] = compute
    training = {}
    if "output_root" in values:
        training["run"] = {"output_root": values["output_root"], "output_dir": None}
    for option, group, name in (
        ("train_data", "data", "dataset_dir"),
        ("length_curriculum", "data", "length_curriculum"),
        ("max_train_data_tokens", "data", "max_train_data_tokens"),
        ("max_train_samples", "data", "max_train_samples"),
        ("max_dev_samples", "data", "max_dev_samples"),
        ("init_checkpoint", "run", "init_checkpoint"), ("run_id", "run", "run_id"),
        ("epochs", "loop", "epochs"), ("micro_batch_size_per_gpu", "loop", "micro_batch_size"),
        ("curriculum_stage_epochs", "loop", "curriculum_stage_epochs"),
        ("curriculum_replay_ratio", "loop", "curriculum_replay_ratio"),
        ("gradient_accumulation_steps", "loop", "gradient_accumulation_steps"),
        ("dev_every", "loop", "dev_every"), ("save_every", "loop", "save_every"),
        ("save_total_limit", "loop", "save_total_limit"),
    ):
        if option in values:
            training.setdefault(group, {})[name] = values[option]
    if "tracking" in values:
        training.setdefault("swanlab", {})["tracking"] = {
            "online": True, "offline": False,
        }[values["tracking"]]
    if training:
        overrides["training"] = training
    return overrides
