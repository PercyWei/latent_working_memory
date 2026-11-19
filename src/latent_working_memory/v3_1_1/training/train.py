"""单机多卡预训练与后训练；执行组织沿用 v4.1。"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, set_seed
from transformers.utils.logging import disable_progress_bar

from latent_working_memory.runtime import initialize_device
from ..config import config_from_dict, config_to_dict
from ..data import load_documents, load_pretraining, tokenization_identity
from ..io import write_json
from ..modeling.compressor import MemoryModel
from .plans import load_training_plan, validate_training_plan
from .checkpoint import checkpoint_file, read_checkpoint, restore_training, save_checkpoint
from .engine import SampleCursor, TrainingEngine, evaluate_dev, training_steps, uses_condition_dev
from .tracking import log_metrics, preserve_training_rng, tracking_run


def _restore_metrics(path, checkpoint_step, updates_per_batch):
    kept = []
    for raw_line in path.read_bytes().split(b"\n")[:-1]:
        line = raw_line.decode("utf-8")
        row = json.loads(line)
        if type(row["step"]) is not int or row["step"] < 1:
            raise ValueError("training metrics contain an invalid optimizer step")
        if row["step"] <= checkpoint_step:
            expected = min((len(kept) + 1) * updates_per_batch, checkpoint_step)
            if row["step"] != expected or "train/loss" not in row:
                raise ValueError("training metrics must contain each completed rollout batch once")
            kept.append(line + "\n")
    if not kept or json.loads(kept[-1])["step"] != checkpoint_step:
        raise ValueError("training metrics do not cover all completed checkpoint steps")
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(kept), encoding="utf-8")
    temporary.replace(path)


def _load_data(config, tokenizer):
    loader = load_pretraining if config.training.objective.type == "ae_lm" else load_documents
    train, train_statistics = loader(config, tokenizer, "train")
    dev, dev_statistics = loader(config, tokenizer, "dev")
    if not train:
        raise ValueError("training selection must contain at least one document")
    return train, dev, train_statistics, dev_statistics


def _initialize_compressor(model, config):
    if config.training.run.init_checkpoint is None:
        return None
    source_path = checkpoint_file(config.training.run.init_checkpoint)
    source = read_checkpoint(source_path)
    source_config = config_from_dict(source["config"])
    # 分段窗口和边界模块由后训练独立设置；压缩器结构与 LoRA 须和来源一致。
    source_model = replace(source_config.model, structure=replace(
        source_config.model.structure,
        pending_stride=config.model.structure.pending_stride,
        pending_size=config.model.structure.pending_size,
        recent_size=config.model.structure.recent_size,
        boundary_hidden_size=config.model.structure.boundary_hidden_size,
    ))
    if (
        source_config.uses_boundary_policy
        or source_config.model_name_or_path != config.model_name_or_path
        or source_model != config.model
    ):
        raise ValueError("initialization requires a matching pretrain or compressor_only checkpoint")
    model.load_compressor_state_dict(source["model"])
    return {
        "checkpoint": str(source_path), "step": source["cursor"]["step"],
        "binding": source["binding"],
    }


def _validate_context_lengths(model, documents, config, training=False):
    """边界策略可一直等待，需容纳除固定 recent 外的完整前缀。"""
    if not config.uses_boundary_policy:
        return
    for document in documents:
        model._check_length(
            len(document.input_ids) - config.recent_size(training) + model.memory_slots,
        )


def _run_dev(model, documents, config, device, boundary_generator, output_dir, step,
             baseline_cache, temperature=1.0):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    records = []
    with preserve_training_rng(device, boundary_generator):
        metrics = evaluate_dev(
            model, documents[rank::world], config, temperature=temperature,
            baseline_cache=baseline_cache, records=records,
        )
    if "segment_tokens" in metrics:
        metrics["compression_ratio"] = metrics["segment_tokens"] / model.memory_slots
    if uses_condition_dev(config):
        folder = output_dir / "dev" / f"global_step_{step}"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"rank_{rank}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if rank == 0:
            write_json(folder / "summary.json", {"step": step, **metrics})
    return metrics


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
            training.updates_per_batch,
        ),
    ))
    config = replace(config, training=training)
    if (config.uses_boundary_policy and training.objective.policy_transition is not None
        and training.loop.max_steps < 2):
        raise ValueError("temperature annealing requires at least two optimizer steps")
    if checkpoint is not None and checkpoint["config"] != config_to_dict(config):
        raise ValueError("resume requires the original run configuration and output directory")
    model = MemoryModel(
        config.model_name_or_path, asdict(config.model), asdict(config.compute), device,
        with_value_head=training.objective.strategy == "ppo",
    )
    _validate_context_lengths(model, train_documents.values(), config, training=True)
    _validate_context_lengths(model, dev_documents.values(), config)
    # 每个 rank 都在参数广播之前加载来源压缩器，包括没有本地样本的空 rank。
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
        if config.uses_boundary_policy else None
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
                _restore_metrics(metrics_path, step, training.updates_per_batch)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(output_dir / "config.json", config_to_dict(config))
            write_json(output_dir / "data.json", {
                "binding": binding, "train": train_statistics, "dev": dev_statistics,
                "compression_windows": {
                    "train_recent_size": config.recent_size(training=True),
                    "dev_recent_size": config.recent_size(),
                },
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
        baseline_cache = {}
        if not resume and dev_documents and uses_condition_dev(config):
            dev = _run_dev(
                model, dev_documents, config, device, boundary_generator, output_dir, 0,
                baseline_cache,
            )
            if rank == 0:
                baseline = {"step": 0, **{f"dev/{key}": value for key, value in dev.items()}}
                write_json(output_dir / "dev_baseline.json", baseline)
                if remote is not None:
                    with preserve_training_rng(device, boundary_generator):
                        log_metrics(
                            remote, baseline, 0, registered, config.stage, output_dir,
                            training.objective,
                        )
        while step < training.loop.max_steps:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            curriculum = ({
                "curriculum_bin": cursor.stage + 1,
                "curriculum_epoch": cursor.stage_epoch + 1,
            } if training.data.length_curriculum else {})
            indices = cursor.next_batch(global_batch)
            previous_step = step
            metrics = engine.train_batch(
                [train_documents[index] for index in indices[rank::world]], boundary_generator,
                step + 1,
            )
            step = engine.completed_steps
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
                **{f"train/{key}": value for key, value in metrics.items()
                   if config.stage != "post" or key != "samples"},
                "resources/step_seconds": seconds / (step - previous_step),
                "resources/peak_memory_gib": peak_memory,
            }
            if "segment_tokens" in metrics:
                record["train/compression_ratio"] = (
                    metrics["segment_tokens"] / config.model.structure.memory_slots
                )
            dev_interval = (training.loop.dev_every != "after_each_epoch"
                            and step // training.loop.dev_every > previous_step // training.loop.dev_every)
            if dev_documents and (dev_interval or cursor.epoch_end or step == training.loop.max_steps):
                dev = _run_dev(
                    model, dev_documents, config, device, boundary_generator, output_dir, step,
                    baseline_cache, metrics.get("policy_temperature", 1.0),
                )
                record.update({f"dev/{key}": value for key, value in dev.items()})
            if rank == 0:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if remote is not None:
                    with preserve_training_rng(device, boundary_generator):
                        log_metrics(
                            remote, record, step, registered, config.stage, output_dir,
                            training.objective,
                        )
                print(
                    f"step {step}/{training.loop.max_steps} loss={metrics['loss']:.5f} "
                    f"grad={metrics['grad_norm']:.5f} samples={metrics['samples']} seconds={seconds:.1f}"
                    + (f" bin={curriculum['curriculum_bin']} "
                       f"stage_epoch={curriculum['curriculum_epoch']}" if curriculum else "")
                    + (f" dev={record['dev/loss']:.5f}" if "dev/loss" in record else ""),
                    flush=True,
                )
            save_interval = (training.loop.save_every != "after_each_epoch"
                             and step // training.loop.save_every > previous_step // training.loop.save_every)
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
