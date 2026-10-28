"""单卡 v4.1 元训练入口。"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import swanlab
import torch
from transformers import AutoConfig, AutoTokenizer, set_seed

from latent_working_memory.runtime import capture_rng, initialize_device, restore_rng
from latent_working_memory.tracking_credentials import swanlab_api_key
from latent_working_memory.v4_1.config import load_config
from latent_working_memory.v4_1.data import dataset_identity, load_episodes
from latent_working_memory.v4_1.modeling.model import OnlineMemoryLM
from latent_working_memory.v4_1.teacher.cache import TeacherCache, build_cache_identity
from latent_working_memory.v4_1.teacher.replay import load_teacher
from latent_working_memory.v4_1.teacher.runtime import RuntimeTeacher
from latent_working_memory.v4_1.training.checkpoint import (
    checkpoint_file, read_checkpoint, restore_training, save_checkpoint,
)
from latent_working_memory.v4_1.training.engine import SampleCursor, TrainingEngine, evaluate_dev


@contextmanager
def tracking_run(config, output_dir, resume, device):
    training = config.training
    if training.tracking == "disabled":
        yield None
        return
    path = output_dir / "swanlab.json"
    identity = json.loads(path.read_text(encoding="utf-8")) if resume else None
    rng = capture_rng(device)
    try:
        run = swanlab.init(
            project=training.swanlab_project, name=output_dir.name,
            config=asdict(config), group=training.group, job_type="train",
            tags=sorted({"scope:main", "method:online-ttt-memory", *training.tags}),
            mode="online", public=False, log_dir=str(output_dir / "swanlab"),
            id=identity["id"] if identity else None,
            resume="must" if identity else "never",
            settings=swanlab.Settings(
                api_key=swanlab_api_key(), interactive=False,
                terminal={"proxy_type": "none"}, probe={"git": False, "monitor": False},
            ),
        )
    finally:
        restore_rng(rng, device)
    with run:
        if identity is None:
            path.write_text(
                json.dumps({"id": run.id, "url": run.url}, indent=2) + "\n", encoding="utf-8"
            )
        for section in ("train", "dev"):
            for name in ("loss", "qa_loss", "outer_am", "inner_before", "inner_after"):
                run.define_metric(f"{section}/{name}", x_axis="_step", section_name=section)
        for name in ("train/grad_norm", "resources/step_seconds", "resources/peak_memory_gib"):
            run.define_metric(name, x_axis="_step", section_name=name.split("/")[0])
        yield run


def _restore_metrics(path, checkpoint_step):
    """保留已提交完整行，只丢弃 checkpoint 后被中断的末行。"""
    lines = path.read_bytes().split(b"\n")
    kept, last_complete_step = [], 0
    for raw_line in lines[:-1]:
        line = raw_line.decode("utf-8")
        row = json.loads(line)
        last_complete_step = row["step"]
        if row["step"] <= checkpoint_step:
            kept.append(line + "\n")
    if lines[-1] and last_complete_step < checkpoint_step:
        raise ValueError("incomplete metrics history reaches into the committed checkpoint steps")
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(kept), encoding="utf-8")
    temporary.replace(path)


def train(config_path, output_dir=None, device="cuda", resume=None):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("v4.1 currently supports one process; do not use torchrun")
    config = load_config(config_path)
    output_dir = output_dir or config.training.output_dir
    if output_dir is None:
        raise ValueError("specify --output-dir or training.output_dir")
    output_dir = Path(output_dir).expanduser().resolve()
    config = replace(config, training=replace(config.training, output_dir=str(output_dir)))
    training = config.training
    checkpoint = read_checkpoint(resume) if resume else None
    if checkpoint is not None and checkpoint["config"] != asdict(config):
        raise ValueError("resume requires the original run configuration and output directory")
    if checkpoint is not None:
        completed_steps = [
            int(path.parent.name.rsplit("_", 1)[1])
            for path in (output_dir / "checkpoints").glob("global_step_*/state.pt")
        ]
        if completed_steps and max(completed_steps) > checkpoint["cursor"]["step"]:
            raise ValueError("resume the latest complete checkpoint in this run")
    if checkpoint is None and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}; use --resume")
    device = initialize_device(device)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    model_window = AutoConfig.from_pretrained(config.model.model_name_or_path).max_position_embeddings
    train_episodes, _, train_statistics = load_episodes(
        training.dataset_dir, tokenizer, split="train",
        min_text_tokens=config.model.pending_size + config.model.recent_size,
        max_text_tokens=training.max_train_data_tokens, model_window=model_window,
        max_samples=training.max_train_samples, seed=config.seed,
    )
    dev_episodes, _, dev_statistics = load_episodes(
        training.dataset_dir, tokenizer, split="dev",
        min_text_tokens=config.model.pending_size + config.model.recent_size,
        model_window=model_window, max_samples=training.max_dev_samples, seed=config.seed,
    )
    if not train_episodes or not dev_episodes:
        raise ValueError("train and dev selections must both contain at least one QA episode")
    if set(train_statistics["raw_document_ids"]) & set(dev_statistics["raw_document_ids"]):
        raise ValueError("train and dev documents overlap")
    teacher_cache = TeacherCache(
        training.teacher_cache_dir, expected_identity=build_cache_identity(config, tokenizer),
    )
    binding = {
        "train": dataset_identity(train_episodes), "dev": dataset_identity(dev_episodes),
        "teacher": teacher_cache.identity,
        "train_documents": train_statistics["raw_document_ids"],
        "dev_documents": dev_statistics["raw_document_ids"],
    }
    if checkpoint is not None and checkpoint["binding"] != binding:
        raise ValueError("training data or teacher cache differs from the resumed checkpoint")
    model = OnlineMemoryLM(config.model).to(device)
    teacher = RuntimeTeacher(load_teacher(config.model, device), config, teacher_cache)
    engine = TrainingEngine(model, training)
    cursor = SampleCursor(len(train_episodes), config.seed)
    query_generator = torch.Generator().manual_seed(config.seed + 1)
    step = 0
    if checkpoint is not None:
        step = restore_training(checkpoint, model, engine, cursor, query_generator, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "data.json").write_text(
        json.dumps({"binding": binding, "train": train_statistics, "dev": dev_statistics},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.jsonl"
    if resume and metrics_path.exists():
        _restore_metrics(metrics_path, step)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last_checkpoint = str(checkpoint_file(resume)) if resume else None
    with tracking_run(config, output_dir, bool(resume), device) as tracking:
        with metrics_path.open("a", encoding="utf-8") as stream:
            while step < training.max_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                indices = cursor.next_batch(training.gradient_accumulation_steps)
                metrics = engine.train_batch(
                    [train_episodes[index] for index in indices], teacher, query_generator,
                )
                step += 1
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                record = {
                    "step": step, "epoch": cursor.epoch,
                    **{f"train/{key}": value for key, value in metrics.items()},
                    "resources/step_seconds": time.perf_counter() - started,
                }
                if device.type == "cuda":
                    record["resources/peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
                if step % training.dev_every == 0 or step == training.max_steps:
                    rng = capture_rng(device)
                    try:
                        dev = evaluate_dev(
                            model, dev_episodes, teacher, training.lambda_am,
                            torch.Generator().manual_seed(config.seed + 2),
                        )
                    finally:
                        restore_rng(rng, device)
                    record.update({f"dev/{key}": value for key, value in dev.items()})
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if tracking is not None:
                    tracking.log({key: value for key, value in record.items() if "/" in key}, step=step)
                print(
                    f"step {step}/{training.max_steps} loss={metrics['loss']:.5f} "
                    f"QA={metrics['qa_loss']:.5f} AM={metrics['outer_am']:.5f} "
                    f"inner={metrics['inner_before']:.5f}->{metrics['inner_after']:.5f} "
                    f"samples={len(indices)} seconds={record['resources/step_seconds']:.1f}"
                    + (f" dev={record['dev/loss']:.5f}" if "dev/loss" in record else ""),
                    flush=True,
                )
                if step % training.save_every == 0 or step == training.max_steps:
                    last_checkpoint = str(save_checkpoint(
                        output_dir, step, config, model, engine, cursor,
                        query_generator, binding, device,
                    ))
    summary = {"step": step, "checkpoint": last_checkpoint, "output_dir": str(output_dir)}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", help="同一运行的 checkpoint 目录或 state.pt")
    args = parser.parse_args(argv)
    train(args.config, args.output_dir, args.device, args.resume)


if __name__ == "__main__":
    main()
