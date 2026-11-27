"""使用官方 ICAE v2 模型和 Trainer 进行预训练或指令微调。"""

import argparse
import os
from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import set_seed

from latent_working_memory.prompts import qa_input_format
from reproductions.icae.source.icae.code.icae_v2.modeling_icae_multi_span import (
    ICAE, ModelArguments, TrainingArguments, adapter_config,
)
from reproductions.icae.source.icae.code.icae_v2.training_utils import ICAETrainingInputs, train_model


def run_training(stage, argv=None):
    parser = argparse.ArgumentParser(description=f"ICAE v2 {stage}", allow_abbrev=False)
    parser.add_argument("--model-path", type=Path, required=True, help="本地基座模型与 tokenizer 目录")
    parser.add_argument("--train-file", type=Path, required=True, help="本地训练集 JSON/JSONL")
    parser.add_argument("--validation-file", type=Path, help="可选，本地验证集 JSON/JSONL；提供时在训练结束后计算验证 loss")
    parser.add_argument("--output-dir", type=Path, required=True, help="当前训练阶段的输出目录")
    parser.add_argument("--fixed-mem-size", type=int, default=128)
    parser.add_argument("--mean-compression-rate", type=int, default=4)
    parser.add_argument("--model-max-length", type=int, default=5120, help="每条样本的 Writer 正文 token 上限")
    parser.add_argument("--num-train-epochs", type=float, default=1, help="仅在 --max-steps=-1 时生效")
    parser.add_argument("--max-steps", type=int, default=200000 if stage == "pretrained" else 30000,
                        help="优化器更新总步数；设为 -1 时按训练轮数执行")
    parser.add_argument("--learning-rate", type=float, default=1e-4 if stage == "pretrained" else 5e-5)
    parser.add_argument("--warmup-steps", type=int, default=300, help="学习率预热的优化器更新步数")
    parser.add_argument("--max-grad-norm", type=float, default=2.0, help="梯度裁剪范数，0 表示关闭")
    parser.add_argument("--gradient-accumulation-steps", type=int,
                        help="默认按 256 / 分布式进程数计算，使全局有效批量为 256")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    if stage == "pretrained":
        parser.add_argument("--lm-ratio", type=float, default=0, help="预训练尝试采样后文续写的概率")
    else:
        parser.add_argument("--checkpoint", type=Path, required=True, help="预训练 ICAE 的 .safetensors 文件")
    args = parser.parse_args(argv)
    if args.gradient_accumulation_steps is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if 256 % world_size:
            parser.error("默认全局批量 256 不能被分布式进程数整除，请显式指定 --gradient-accumulation-steps")
        args.gradient_accumulation_steps = 256 // world_size
    if args.fixed_mem_size <= 0 or args.fixed_mem_size & (args.fixed_mem_size - 1):
        parser.error("--fixed-mem-size must be a positive power of two")
    if min(args.model_max_length, args.mean_compression_rate, args.gradient_accumulation_steps, args.save_steps) < 1:
        parser.error("length, compression rate, gradient accumulation and save interval must be positive")
    if args.learning_rate <= 0 or args.num_train_epochs <= 0 or (args.max_steps != -1 and args.max_steps <= 0):
        parser.error("learning rate and epochs must be positive; --max-steps must be positive or -1")
    if args.warmup_steps < 0 or args.max_grad_norm < 0:
        parser.error("warmup steps and maximum gradient norm must be nonnegative")
    if stage == "pretrained" and not 0 <= args.lm_ratio <= 1:
        parser.error("--lm-ratio must be between 0 and 1")

    base = args.model_path.expanduser().resolve(strict=True)
    checkpoint = str(args.checkpoint.expanduser().resolve(strict=True)) if stage == "finetuned" else ""
    files = {"train": str(args.train_file.expanduser().resolve(strict=True))}
    if args.validation_file is not None:
        files["validation"] = str(args.validation_file.expanduser().resolve(strict=True))
    splits = load_dataset("json", data_files=files)
    columns = {"text"} if stage == "pretrained" else {"input", "prompt", "answer"}
    for name, dataset in splits.items():
        if not len(dataset) or not columns.issubset(dataset.column_names):
            parser.error(f"{name} requires nonempty data with columns {sorted(columns)}")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir.expanduser().resolve()), restore_from=checkpoint,
        fixed_mem_size=args.fixed_mem_size, mean_compression_rate=args.mean_compression_rate,
        lm_ratio=args.lm_ratio if stage == "pretrained" else 0.0,
        model_max_length=args.model_max_length, num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps, learning_rate=args.learning_rate, seed=args.seed,
        warmup_steps=args.warmup_steps, max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps, save_steps=args.save_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, ddp_find_unused_parameters=False,
        remove_unused_columns=False, prediction_loss_only=True, bf16=True, report_to=[],
    )
    set_seed(args.seed)
    lora = adapter_config()
    model = ICAE(ModelArguments(model_name_or_path=str(base), lora_r=lora.r, train=True),
                 training_args, lora, mean_resizing=False)
    qa_input_format(model.tokenizer)
    inputs = ICAETrainingInputs(model, stage)
    if stage == "pretrained":
        train_transform = partial(inputs.pretrain, lm_ratio=training_args.lm_ratio)
        validation_transform = partial(inputs.pretrain, validation=True)
    else:
        train_transform = validation_transform = inputs.finetune
    eval_dataset = None
    if args.validation_file is not None:
        eval_dataset = splits["validation"].with_transform(validation_transform)
    train_model(model, splits["train"].with_transform(train_transform), eval_dataset, training_args)
