"""v3 模型加载、阶段衔接、可恢复训练与本地运行记录。"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_working_memory.v3.checkpoint import TokenMemoryCheckpointHandler, read_checkpoint
from latent_working_memory.v3.config import DYNAMIC_METHODS, ModelConfig
from latent_working_memory.v3.data import filter_factqa, load_factqa
from latent_working_memory.v3.model import GistMemoryModel
from latent_working_memory.v3.pretrain_data import load_pretraining
from latent_working_memory.v3.tracking import (
    configure_training_metrics,
    experiment_directory,
    method_tracking_run,
    training_metrics,
)
from latent_working_memory.v3.tracking_credentials import swanlab_api_key
from latent_working_memory.v3.training_data import TrainingDataLoader, make_data_loader


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_codec(model_config, device, objective_config):
    """同源独立编码器／解码器；编码训练 LoRA，冻结读取保留记忆梯度。"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, revision=model_config.revision
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("v3 training and generation require tokenizer.eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.revision,
        torch_dtype=getattr(torch, model_config.dtype),
        attn_implementation=model_config.attention_implementation,
    ).to(device)
    write_divisor = (
        objective_config.icae_min_segments
        if objective_config.method == "icae_multi"
        else objective_config.ac_num_segments
        if objective_config.method == "autocompressors"
        else 1
    )
    codec = GistMemoryModel(
        base,
        memory_slots=model_config.memory_slots,
        write_slots=((model_config.memory_slots + write_divisor - 1) // write_divisor),
        lora_rank=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
        lora_target_modules=model_config.lora_target_modules,
        gradient_checkpointing=model_config.gradient_checkpointing,
    )
    return codec, tokenizer


def load_splits(config, tokenizer, model_window):
    training = config.training
    pretraining = config.objective.stage in {"pretrain", "lm"}
    if pretraining:
        splits, statistics = load_pretraining(config, tokenizer, model_window)
    else:
        splits, statistics = filter_factqa(
            load_factqa(training.dataset_dir, tokenizer),
            training.max_qa_input_tokens,
            model_window,
        )
    statistics["source_data"] = {name: dataset_identity(rows) for name, rows in splits.items()}
    statistics["selection"] = {
        "seed": training.seed,
        "max_train_samples": training.max_train_samples,
        "max_dev_samples": training.max_dev_samples,
    }
    limits = {"train": training.max_train_samples, "dev": training.max_dev_samples, "test": None}
    for split, rows in splits.items():
        selected = select_examples(rows, limits[split], training.seed, split, pretraining)
        splits[split] = selected
        counts = statistics["splits"][split]
        counts["selected"] = len(selected)
        if pretraining:
            counts["selected_by_task"] = {
                task: sum(row.task == task for row in selected) for task in ("ae", "continuation")
            }
            counts["selected_input_tokens"] = sum(len(row.input_ids) for row in selected)
            counts["selected_target_tokens"] = sum(len(row.target_ids) for row in selected)
        else:
            counts["selected_source_tokens"] = sum(len(row.full_input_ids) for row in selected)
            counts["selected_questions"] = sum(len(row.qas) for row in selected)
    return splits, statistics


def select_examples(rows, limit, seed, split, pretraining):
    """在内存选择完整样本；AE/LM 按比例分层，QA 保留整条更新轨迹。"""
    if limit is None or limit >= len(rows):
        return rows

    def priority(example):
        identity = example.sample_id if pretraining else example.trajectory_id
        return hashlib.blake2b(f"{seed}:{split}:{identity}".encode(), digest_size=16).digest()

    if not pretraining:
        return tuple(sorted(rows, key=priority)[:limit])
    groups = {
        task: sorted((row for row in rows if row.task == task), key=priority)
        for task in sorted({row.task for row in rows})
    }
    quotas = {task: limit * len(group) / len(rows) for task, group in groups.items()}
    counts = {task: math.floor(quota) for task, quota in quotas.items()}
    while sum(counts.values()) < limit:
        task = max(
            (task for task in groups if counts[task] < len(groups[task])),
            key=lambda task: (quotas[task] - counts[task], task),
        )
        counts[task] += 1
    return tuple(
        sorted(
            (row for task, group in groups.items() for row in group[: counts[task]]), key=priority
        )
    )


def dataset_identity(rows):
    """绑定实际分词结果、问题与使用安排，防止续训时替换数据。"""
    digest = hashlib.blake2b(digest_size=32)
    for example in rows:
        digest.update(
            json.dumps(
                asdict(example), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {"examples": len(rows), "fingerprint": digest.hexdigest()}


def _validate_pretraining_identity(config, pretraining):
    if (
        config.objective.method in DYNAMIC_METHODS
        and config.objective.stage != "pretrain"
        and config.training.experiment_id is not None
    ):
        if pretraining is None or pretraining["experiment_id"] is None:
            raise ValueError("dynamic experiment requires an explicit pretraining experiment_id")
        if pretraining["experiment_id"] != config.training.experiment_id:
            raise ValueError("dynamic experiment_id differs from its original pretraining source")


def make_run(
    config,
    splits,
    statistics,
    device,
    world_size,
    resolved_model_revision=None,
    initialization=None,
):
    inherited = (
        initialization["pretraining_sources"]
        if initialization is not None
        else {"document_ids": [], "dedup_clusters": []}
    )
    documents = {example.document_id for rows in splits.values() for example in rows}
    clusters = {example.dedup_cluster for rows in splits.values() for example in rows}
    if config.objective.stage in {"pretrain", "lm"}:
        pretraining_sources = {
            "document_ids": sorted(documents | set(inherited["document_ids"])),
            "dedup_clusters": sorted(clusters | set(inherited["dedup_clusters"])),
        }
    else:
        if documents.intersection(inherited["document_ids"]) or clusters.intersection(
            inherited["dedup_clusters"]
        ):
            raise ValueError("QA sources overlap pretraining document IDs or dedup clusters")
        pretraining_sources = inherited
    pretraining = initialization["pretraining"] if initialization is not None else None
    _validate_pretraining_identity(config, pretraining)
    step_offset = (
        initialization["global_step"]
        if initialization is not None
        and initialization["experiment_dir"] == str(experiment_directory(config.training))
        else 0
    )
    return {
        "config": config.to_dict(),
        "resolved_model_revision": resolved_model_revision,
        "data": {name: dataset_identity(rows) for name, rows in splits.items()},
        "source_data": statistics["source_data"],
        "data_statistics": {
            key: value for key, value in statistics.items() if key != "source_data"
        },
        "world_size": world_size,
        "global_batch_size": config.training.global_batch_size(world_size),
        "device_type": torch.device(device).type,
        "autocast_dtype": "bfloat16" if torch.device(device).type == "cuda" else None,
        "initialization": initialization,
        "step_offset": step_offset,
        "pretraining": pretraining,
        "pretraining_sources": pretraining_sources,
    }


def load_initialization(path, model, config):
    """阶段切换只载入新增权重，不继承优化器、训练游标或随机状态。"""
    checkpoint = read_checkpoint(path)
    initialization = initialization_record(
        path, checkpoint["run"], checkpoint["cursor"]["step"], config
    )
    model.load_trainable_state_dict(checkpoint["trainable"])
    return initialization


def initialization_record(path, previous_run, step, config):
    """校验并记录阶段来源；连续训练直接使用内存记录，无须重读 checkpoint。"""
    previous = previous_run["config"]
    previous_model = ModelConfig(**previous["model"])
    # 激活重算是执行设置，阶段初始化可复用启用该设置前的训练权重。
    if (
        replace(previous_model, gradient_checkpointing=config.model.gradient_checkpointing)
        != config.model
    ):
        raise ValueError("initialization model configuration differs from the current model")
    required_stages = {
        "warmup": {"pretrain"},
        "qa": {"pretrain"},
        "policy": {"pretrain", "warmup"},
    }
    stage = config.objective.stage
    if stage in required_stages and previous["objective"]["stage"] not in required_stages[stage]:
        raise ValueError(f"{stage} initialization requires {sorted(required_stages[stage])}")
    if config.objective.method == "icae_multi" and (
        previous["objective"]["method"] != "icae_multi"
        or any(
            previous["objective"].get(name) != getattr(config.objective, name)
            for name in ("icae_min_segments", "icae_max_segments")
        )
    ):
        raise ValueError("ICAE-multi initialization requires the same segment range")
    # 初始化只读取来源身份，保留原 checkpoint 的训练配置与数据协议。
    previous_training = previous["training"]
    root = Path(previous_training["experiment_dir"] or previous_training["output_dir"]).resolve()
    if previous["objective"]["stage"] == "pretrain":
        identity = (
            json.loads((root / "swanlab.json").read_text(encoding="utf-8"))
            if previous_training["swanlab_project"] is not None
            else None
        )
        pretraining = {
            "experiment_id": previous_training["experiment_id"],
            "run_name": previous_training.get("experiment_name") or root.name,
            "run_dir": str(root),
            "run_id": identity["id"] if identity is not None else None,
            "run_url": identity["url"] if identity is not None else None,
            "checkpoint": str(Path(path).resolve()),
            "step": step,
        }
    else:
        pretraining = previous_run["pretraining"]
    _validate_pretraining_identity(config, pretraining)
    return {
        "checkpoint": str(Path(path).resolve()),
        "method": previous["objective"]["method"],
        "stage": previous["objective"]["stage"],
        "step": step,
        "global_step": previous_run["step_offset"] + step,
        "experiment_dir": str(root),
        "experiment_id": previous_training["experiment_id"],
        "resolved_model_revision": previous_run["resolved_model_revision"],
        "pretraining_sources": previous_run["pretraining_sources"],
        "pretraining": pretraining,
    }


def validate_cursor(cursor, config, samples, world_size):
    batch_size = config.global_batch_size(world_size)
    names = {"epoch", "sample_offset", "step", "sample_visits"}
    if set(cursor) != names or any(
        type(value) is not int or value < 0 for value in cursor.values()
    ):
        raise ValueError("checkpoint has an invalid training cursor")
    epoch, offset = cursor["epoch"], cursor["sample_offset"]
    if (
        epoch > config.epochs
        or offset >= samples
        or offset % batch_size
        or (epoch == config.epochs and offset != 0)
        or cursor["step"] != epoch * math.ceil(samples / batch_size) + offset // batch_size
        or cursor["sample_visits"] != epoch * samples + offset
    ):
        raise ValueError("checkpoint cursor does not match the epoch/batch schedule")


def evaluate_split(engine, examples, epoch=0):
    totals, samples = {}, 0
    loader = make_data_loader(examples, engine.global_batch_size, engine.config.seed, shuffle=False)
    for batch in loader:
        metrics = engine.eval_batch(batch, epoch=epoch)
        weight = metrics["samples"]
        for name, value in metrics.items():
            if name != "samples" and value is not None:
                totals[name] = totals.get(name, 0.0) + value * weight
        samples += weight
    if not samples:
        raise ValueError("evaluation split must contain at least one example")
    return {"samples": samples, **{name: value / samples for name, value in totals.items()}}


def _rewind_metrics(path, step):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    retained = [record for record in records if record["step"] <= step]
    if [record["step"] for record in retained] != list(range(1, step + 1)):
        raise ValueError("training metrics do not match the checkpoint's completed steps")
    if len(retained) != len(records):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in retained
            ),
            encoding="utf-8",
        )
        temporary.replace(path)


def _tracking_context(config, engine, run):
    if engine.rank != 0:
        return nullcontext(None)
    return method_tracking_run(
        config,
        run,
        engine.device,
        api_key=swanlab_api_key() if config.training.swanlab_project is not None else None,
    )


@dataclass
class TrainingStage:
    cursor: dict
    data_loader: TrainingDataLoader
    checkpoint_handler: TokenMemoryCheckpointHandler
    checkpoint_path: Path | None = None


def prepare_training(config, engine, splits, run, resume=None, stop_after_steps=None):
    """先校验并准备阶段目录或恢复进度，再开启／更新方法级跟踪会话。"""
    settings = config.training
    train, dev = splits["train"], splits["dev"]
    if not train or not dev:
        raise ValueError("training requires nonempty train and dev splits")
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps < 1):
        raise ValueError("stop_after_steps must be a positive optimizer-step limit")
    if resume is None and config.objective.stage in {"warmup", "qa", "policy"}:
        if settings.init_checkpoint is None or run["initialization"] is None:
            raise ValueError(f"{config.objective.stage} requires an initialization checkpoint")
    output = Path(settings.output_dir)
    metrics_path = output / "metrics.jsonl"
    cursor = {"epoch": 0, "sample_offset": 0, "step": 0, "sample_visits": 0}
    checkpoint_path = Path(resume) if resume is not None else None
    if resume is not None:
        if Path(resume).resolve().parent != (output / "checkpoints").resolve():
            raise ValueError("resume checkpoint must belong to output_dir/checkpoints")
        if json.loads((output / "run.json").read_text(encoding="utf-8")) != run:
            raise ValueError("saved run differs from the resume configuration or data")
    else:
        error = None
        if engine.rank == 0 and output.exists() and any(output.iterdir()):
            error = "new training requires an empty output directory"
        if engine.world_size > 1:
            status = [error]
            dist.broadcast_object_list(status, src=0)
            error = status[0]
        if error is not None:
            raise ValueError(error)
        if engine.rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "config.json", config.to_dict())
            write_json(output / "run.json", run)
            metrics_path.touch()
    if engine.world_size > 1:
        dist.barrier()
    loader = make_data_loader(train, engine.global_batch_size, settings.seed)
    handler = TokenMemoryCheckpointHandler(
        engine, loader, output / "checkpoints", run, cursor, resume_from_path=resume
    )
    if resume is not None:
        step = handler.load_checkpoint()
        validate_cursor(cursor, settings, len(train), engine.world_size)
        if step != cursor["step"]:
            raise ValueError("checkpoint directory step differs from its saved cursor")
        if engine.rank == 0:
            _rewind_metrics(metrics_path, step)
    return TrainingStage(cursor, loader, handler, checkpoint_path)


def train_loop(config, engine, splits, run, stage, stop_after_steps=None, tracking=None):
    """执行已准备阶段的全局 batch 日程，跟踪会话由方法训练入口管理。"""
    settings = config.training
    train, dev = splits["train"], splits["dev"]
    output = Path(settings.output_dir)
    metrics_path = output / "metrics.jsonl"
    cursor = stage.cursor
    total_steps = settings.epochs * math.ceil(len(train) / engine.global_batch_size)
    planned_steps = (
        min(total_steps, stop_after_steps) if stop_after_steps is not None else total_steps
    )
    if tracking is not None:
        configure_training_metrics(tracking, config.objective.method, config.objective.stage)
    for epoch in range(cursor["epoch"], settings.epochs):
        if stop_after_steps is not None and cursor["step"] >= stop_after_steps:
            break
        stage.data_loader.sampler.set_epoch(epoch)
        for batch in stage.data_loader:
            if engine.device.type == "cuda":
                torch.cuda.synchronize(engine.device)
                torch.cuda.reset_peak_memory_stats(engine.device)
            started = time.perf_counter()
            metrics = engine.step(batch, epoch=epoch)
            if engine.device.type == "cuda":
                torch.cuda.synchronize(engine.device)
            resources = {"optimizer_step_seconds": time.perf_counter() - started}
            if engine.device.type == "cuda":
                resources["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
                    engine.device
                )
            if engine.world_size > 1:
                maximum = torch.tensor(
                    list(resources.values()), device=engine.device, dtype=torch.float64
                )
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                resources = dict(zip(resources, maximum.tolist(), strict=True))
            cursor["step"] += 1
            cursor["sample_visits"] += len(batch)
            consumed = cursor["sample_offset"] + len(batch)
            epoch_end = consumed == len(train)
            cursor["epoch"] = epoch + int(epoch_end)
            cursor["sample_offset"] = 0 if epoch_end else consumed
            step = cursor["step"]
            stopping = stop_after_steps is not None and step >= stop_after_steps
            record = {
                "step": step,
                "global_step": run["step_offset"] + step,
                "stage": config.objective.stage,
                "epoch": epoch + 1,
                "train/epoch": cursor["sample_visits"] / len(train),
                **{f"train/{key}": value for key, value in metrics.items() if value is not None},
                **{f"resources/{key}": value for key, value in resources.items()},
            }
            if step % settings.eval_every == 0 or epoch_end or stopping:
                development = evaluate_split(engine, dev, epoch=epoch)
                record.update({f"dev/{key}": value for key, value in development.items()})
            if engine.rank == 0:
                with metrics_path.open("a", encoding="utf-8") as stream:
                    text = json.dumps(record, ensure_ascii=False, allow_nan=False)
                    stream.write(text + "\n")
                progress = [
                    f"{config.objective.method}/{config.objective.stage}",
                    f"epoch={epoch + 1}",
                    f"step={step}/{planned_steps}",
                    f"loss={metrics['loss']:.4f}",
                ]
                if "train/slots_final" in record:
                    progress.append(f"slots={record['train/slots_final']:.1f}")
                if "dev/loss" in record:
                    progress.append(f"dev_loss={record['dev/loss']:.4f}")
                progress.append(f"{resources['optimizer_step_seconds']:.2f}s/step")
                if "peak_memory_allocated_bytes" in resources:
                    progress.append(
                        f"peak={resources['peak_memory_allocated_bytes'] / 1024**3:.1f}GiB"
                    )
                print(" | ".join(progress), flush=True)
                if tracking is not None:
                    tracking.log(
                        training_metrics(record, config.objective.method),
                        step=record["global_step"],
                    )
            if step % settings.save_every == 0 or epoch_end or stopping:
                stage.checkpoint_handler.save_checkpoint(step)
                stage.checkpoint_path = output / "checkpoints" / f"global_step_{step}"
            if stopping:
                break
        if stop_after_steps is not None and cursor["step"] >= stop_after_steps:
            break
    result = {
        "complete": cursor["epoch"] == settings.epochs,
        "completed_steps": cursor["step"],
        "global_step": run["step_offset"] + cursor["step"],
        "total_steps": total_steps,
        "completed_epochs": cursor["epoch"],
        "sample_visits": cursor["sample_visits"],
        "stop_after_steps": stop_after_steps,
        "checkpoint": str(stage.checkpoint_path),
    }
    if engine.rank == 0:
        write_json(output / "training-result.json", result)
    return result
