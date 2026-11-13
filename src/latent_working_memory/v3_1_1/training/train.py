"""单机多卡两阶段训练；执行组织沿用 v4.1。"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, set_seed
from transformers.utils.logging import disable_progress_bar

from latent_working_memory.runtime import initialize_device
from ..config import config_from_dict, config_to_dict
from ..data import load_documents, load_pretraining, tokenization_identity
from ..evaluation.records import write_json
from ..modeling.compressor import MemoryModel
from ..plans import load_training_plan, validate_training_plan
from .checkpoint import checkpoint_file, read_checkpoint, restore_training, save_checkpoint
from .engine import SampleCursor, TrainingEngine, evaluate_dev, training_steps
from .tracking import log_metrics, preserve_training_rng, tracking_run


def _training_output_dir(config, checkpoint):
    if checkpoint is not None:
        return Path(checkpoint["config"]["training"]["run"]["output_dir"]).expanduser().resolve()
    run_id = config.training.run.run_id
    if run_id is None:
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M")
        prefix = "policy-post" if config.stage == "post" else "pretrain"
        run_id = f"{prefix}_k{config.model.structure.memory_slots}_{timestamp}"
    return Path(config.training.run.output_root).expanduser().resolve() / run_id / "train"


def _restore_metrics(path, checkpoint_step):
    kept = []
    for raw_line in path.read_bytes().split(b"\n")[:-1]:
        line = raw_line.decode("utf-8")
        row = json.loads(line)
        if type(row["step"]) is not int or row["step"] < 1:
            raise ValueError("training metrics contain an invalid optimizer step")
        if row["step"] <= checkpoint_step:
            if row["step"] != len(kept) + 1 or "train/loss" not in row:
                raise ValueError("training metrics must contain each completed checkpoint step once")
            kept.append(line + "\n")
    if len(kept) != checkpoint_step:
        raise ValueError("training metrics do not cover all completed checkpoint steps")
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(kept), encoding="utf-8")
    temporary.replace(path)


def _load_data(config, tokenizer):
    loader = load_pretraining if config.stage == "pretrain" else load_documents
    train, train_statistics = loader(config, tokenizer, "train")
    dev, dev_statistics = loader(config, tokenizer, "dev")
    if not train:
        raise ValueError("training selection must contain at least one document")
    return train, dev, train_statistics, dev_statistics


def _initialize_compressor(model, config):
    if config.stage == "pretrain":
        return None
    source_path = checkpoint_file(config.training.run.init_checkpoint)
    source = read_checkpoint(source_path)
    source_config = config_from_dict(source["config"])
    # 初始化仅继承压缩器；精度、attention 和保存张量策略由后训练独立选择。
    source_compute = replace(
        source_config.compute, dtype=config.compute.dtype,
        attention_implementation=config.compute.attention_implementation,
        mixed_precision=config.compute.mixed_precision, save_on_cpu=config.compute.save_on_cpu,
    )
    if (
        source_config.stage != "pretrain"
        or source_config.model_name_or_path != config.model_name_or_path
        or source_config.model.structure.memory_slots != config.model.structure.memory_slots
        or source_config.model.lora != config.model.lora
        or source_compute != config.compute
    ):
        raise ValueError("post initialization requires a matching pretrain compressor")
    model.load_compressor_state_dict(source["model"])
    return {
        "checkpoint": str(source_path), "step": source["cursor"]["step"],
        "binding": source["binding"],
    }


def _summary(output_dir, step, checkpoint):
    value = {"step": step, "checkpoint": checkpoint, "output_dir": str(output_dir)}
    write_json(output_dir / "summary.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def train(plan_dir, resume=None):
    config = load_training_plan(plan_dir)
    checkpoint = read_checkpoint(resume) if resume else None
    if checkpoint is not None:
        validate_training_plan(config, checkpoint)
    device = initialize_device("cuda")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    if rank != 0:
        disable_progress_bar()
    output_dir = Path(config.training.run.output_dir).expanduser().resolve()
    training = config.training
    if checkpoint is not None:
        if len(checkpoint["rng"]) != world or len(checkpoint["boundary_rng"]) != world:
            raise ValueError("resume requires the original number of training processes")
        selected = checkpoint_file(resume)
        step = checkpoint["cursor"]["step"]
        if selected != output_dir / "checkpoints" / f"global_step_{step}" / "state.pt":
            raise ValueError("checkpoint path differs from its saved training directory")
        complete = [
            int(path.parent.name.rsplit("_", 1)[1])
            for path in (output_dir / "checkpoints").glob("global_step_*/state.pt")
        ]
        if max(complete) != step:
            raise ValueError("resume the latest complete checkpoint in this run")
    existing = [checkpoint is None and rank == 0 and any(
        (output_dir / name).exists() for name in (
            "config.json", "data.json", "metrics.jsonl", "summary.json", "checkpoints", "swanlab.json",
        )
    )]
    if world > 1:
        dist.broadcast_object_list(existing, src=0)
    if existing[0]:
        raise ValueError(f"training results already exist: {output_dir}; use --resume")
    global_batch = world * training.loop.micro_batch_size * training.loop.gradient_accumulation_steps
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_documents, dev_documents, train_statistics, dev_statistics = _load_data(config, tokenizer)
    bin_sizes = ([bucket["samples"] for bucket in train_statistics["length_bins"]]
                 if training.data.length_curriculum else None)
    training = replace(training, loop=replace(
        training.loop, max_steps=training_steps(
            training.loop, len(train_documents), global_batch, bin_sizes,
        ),
    ))
    config = replace(config, training=training)
    if checkpoint is not None and checkpoint["config"] != config_to_dict(config):
        raise ValueError("resume requires the original run configuration and output directory")
    model = MemoryModel(
        config.model_name_or_path, asdict(config.model), asdict(config.compute), device,
    )
    # 每个 rank 都在参数广播之前加载预训练压缩器，包括没有本地样本的空 rank。
    pretraining = (_initialize_compressor(model, config) if checkpoint is None
                   else checkpoint["binding"]["pretraining"])
    binding = {
        "train": train_statistics, "dev": dev_statistics,
        "tokenization": tokenization_identity(tokenizer),
        "model": model.writer.get_base_model().config.to_dict(), "pretraining": pretraining,
    }
    if checkpoint is not None and checkpoint["binding"] != binding:
        raise ValueError("training data, model or tokenization differs from the resumed checkpoint")
    engine = TrainingEngine(model, config)
    cursor = SampleCursor(
        len(train_documents), config.seed, bin_sizes=bin_sizes,
        stage_epochs=training.loop.curriculum_stage_epochs,
        replay_ratio=training.loop.curriculum_replay_ratio,
    )
    boundary_generator = (
        torch.Generator().manual_seed((config.seed + rank) % 2**32)
        if config.stage == "post" else None
    )
    set_seed((config.seed + rank) % 2**32)
    step = restore_training(checkpoint, model, engine, cursor, boundary_generator, device) if checkpoint else 0
    del checkpoint, tokenizer
    last_checkpoint = str(checkpoint_file(resume)) if resume else None
    metrics_path = output_dir / "metrics.jsonl"
    setup_error = None
    if rank == 0:
        try:
            if resume:
                _restore_metrics(metrics_path, step)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(output_dir / "config.json", config_to_dict(config))
            write_json(output_dir / "data.json", {
                "binding": binding, "train": train_statistics, "dev": dev_statistics,
            })
            print(
                f"train stage={config.stage} output_dir={output_dir} world_size={world} "
                f"microbatch_per_gpu={training.loop.micro_batch_size} "
                f"gradient_accumulation_steps={training.loop.gradient_accumulation_steps} "
                f"effective_batch_size={global_batch} documents={len(train_documents)} "
                f"total_steps={training.loop.max_steps}", flush=True,
            )
        except Exception as error:
            setup_error = f"training output setup failed: {error}"
    status = [setup_error]
    if world > 1:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])
    if step == training.loop.max_steps:
        summary = {"step": step, "checkpoint": last_checkpoint, "output_dir": str(output_dir)}
        if rank == 0:
            _summary(output_dir, step, last_checkpoint)
        return summary
    train_documents, dev_documents = list(train_documents.values()), list(dev_documents.values())
    with ExitStack() as stack:
        remote = stream = None
        registered, setup_error = set(), None
        if rank == 0:
            try:
                remote = stack.enter_context(tracking_run(
                    config, output_dir, bool(resume), device, boundary_generator,
                ))
                stream = stack.enter_context(metrics_path.open("a", encoding="utf-8"))
            except Exception as error:
                setup_error = f"training tracking setup failed: {error}"
        status = [setup_error]
        if world > 1:
            dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(status[0])
        while step < training.loop.max_steps:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            curriculum = ({
                "curriculum_bin": cursor.stage + 1,
                "curriculum_epoch": cursor.stage_epoch + 1,
            } if training.data.length_curriculum else {})
            indices = cursor.next_batch(global_batch)
            metrics = engine.train_batch(
                [train_documents[index] for index in indices[rank::world]], boundary_generator,
            )
            step += 1
            torch.cuda.synchronize(device)
            resources = torch.tensor([
                time.perf_counter() - started,
                torch.cuda.max_memory_allocated(device) / 1024**3,
            ], device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(resources, op=dist.ReduceOp.MAX)
            seconds, peak_memory = resources.tolist()
            record = {
                "step": step, "epoch": cursor.epoch, "epoch_end": cursor.epoch_end,
                **curriculum,
                **{f"train/{key}": value for key, value in metrics.items()},
                "resources/step_seconds": seconds,
                "resources/peak_memory_gib": peak_memory,
            }
            dev_interval = training.loop.dev_every != "after_each_epoch" and step % training.loop.dev_every == 0
            if dev_documents and (dev_interval or cursor.epoch_end or step == training.loop.max_steps):
                with preserve_training_rng(device, boundary_generator):
                    dev = evaluate_dev(model, dev_documents[rank::world], config)
                record.update({f"dev/{key}": value for key, value in dev.items()})
            if rank == 0:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if remote is not None:
                    with preserve_training_rng(device, boundary_generator):
                        log_metrics(remote, record, step, registered)
                print(
                    f"step {step}/{training.loop.max_steps} loss={metrics['loss']:.5f} "
                    f"grad={metrics['grad_norm']:.5f} samples={metrics['samples']} seconds={seconds:.1f}"
                    + (f" bin={curriculum['curriculum_bin']} "
                       f"stage_epoch={curriculum['curriculum_epoch']}" if curriculum else "")
                    + (f" dev={record['dev/loss']:.5f}" if "dev/loss" in record else ""),
                    flush=True,
                )
            save_interval = training.loop.save_every != "after_each_epoch" and step % training.loop.save_every == 0
            if save_interval or cursor.epoch_end or step == training.loop.max_steps:
                last_checkpoint = str(save_checkpoint(
                    output_dir, step, config, model, engine, cursor, boundary_generator, binding, device,
                ))
    summary = {"step": step, "checkpoint": last_checkpoint, "output_dir": str(output_dir)}
    if rank == 0:
        _summary(output_dir, step, last_checkpoint)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    if args.resume != (args.checkpoint is not None):
        parser.error("--resume and --checkpoint must be supplied together")
    try:
        train(args.plan_dir, resume=args.checkpoint)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
