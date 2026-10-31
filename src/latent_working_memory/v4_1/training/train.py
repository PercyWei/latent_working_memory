"""单机多卡 v4.1 元训练入口。"""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from datetime import datetime
import fcntl
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import swanlab
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer, set_seed
from transformers.utils.logging import disable_progress_bar

from latent_working_memory.runtime import capture_rng, initialize_device, restore_rng
from latent_working_memory.tracking_credentials import swanlab_api_key
from latent_working_memory.v4_1.config import config_from_dict
from latent_working_memory.v4_1.data import dataset_identity, load_episodes
from latent_working_memory.v4_1.modeling.model import OnlineMemoryLM
from latent_working_memory.v4_1.plans import load_training_plan, validate_training_plan
from latent_working_memory.v4_1.teacher.cache import TeacherCache, build_cache_identity
from latent_working_memory.v4_1.teacher.replay import load_teacher
from latent_working_memory.v4_1.teacher.runtime import RuntimeTeacher
from latent_working_memory.v4_1.training.checkpoint import (
    checkpoint_file, read_checkpoint, restore_training, save_checkpoint,
)
from latent_working_memory.v4_1.training.engine import (
    SampleCursor, TrainingEngine, evaluate_dev, training_steps,
)
from latent_working_memory.v4_1.training.tracking import configure_inner_am_panels


@contextmanager
def tracking_run(config, output_dir, resume, device):
    training = config.training
    if training.tracking == "disabled":
        yield None
        return
    with (output_dir / ".swanlab-publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output_dir / "swanlab.json"
        identity = json.loads(path.read_text(encoding="utf-8")) if resume else None
        workspace = None
        if identity is not None:
            project_path, run_id = identity["url"].split("/@", 1)[1].split("/runs/", 1)
            workspace, project = project_path.split("/")
            if project != training.swanlab_project or run_id != identity["id"]:
                raise ValueError("SwanLab URL differs from the saved project or run id")
        rng = capture_rng(device)
        try:
            api_key = swanlab_api_key()
            run = swanlab.init(
                project=training.swanlab_project, workspace=workspace,
                name=None if resume else output_dir.parent.name,
                config=None if resume else {
                    **asdict(config),
                    "world_size": dist.get_world_size() if dist.is_initialized() else 1,
                    "effective_batch_size": training.micro_batch_size * training.gradient_accumulation_steps
                    * (dist.get_world_size() if dist.is_initialized() else 1),
                },
                group=training.group, job_type="train",
                tags=sorted({"scope:main", "method:online-ttt-memory", *training.tags}),
                mode="online", public=False, log_dir=str(output_dir / "swanlab"),
                id=identity["id"] if identity else None,
                resume="must" if identity else "never",
                settings=swanlab.Settings(
                    api_key=api_key, interactive=False,
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
            project_path = run.url.split("/@", 1)[1].split("/runs/", 1)[0]
            with (output_dir.parents[1] / ".swanlab-panels.lock").open("a") as panel_lock:
                fcntl.flock(panel_lock, fcntl.LOCK_EX)
                api = swanlab.Api(api_key=api_key)
                configure_inner_am_panels(api, api.run(f"{project_path}/{run.id}"))
            for section in ("train", "dev"):
                for name in (
                    "loss", "qa_loss", "outer_am", "outer_am_output", "outer_am_mass",
                    "inner_before", "inner_after",
                ):
                    run.define_metric(f"{section}/{name}", x_axis="_step", section_name=section)
            for name in ("train/grad_norm", "resources/step_seconds", "resources/peak_memory_gib"):
                run.define_metric(name, x_axis="_step", section_name=name.split("/")[0])
            yield run


def _restore_metrics(path, checkpoint_step):
    """已提交步骤须完整连续；丢弃 checkpoint 后记录和未提交的末行。"""
    lines = path.read_bytes().split(b"\n")
    kept = []
    for raw_line in lines[:-1]:
        line = raw_line.decode("utf-8")
        row = json.loads(line)
        if row["step"] <= checkpoint_step:
            if type(row["step"]) is not int or row["step"] != len(kept) + 1 or "train/loss" not in row:
                raise ValueError("training metrics must contain every completed checkpoint step exactly once")
            kept.append(line + "\n")
    if len(kept) != checkpoint_step:
        raise ValueError("training metrics do not cover all completed checkpoint steps")
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(kept), encoding="utf-8")
    temporary.replace(path)


def _training_output_dir(config, checkpoint):
    if checkpoint is not None:
        return Path(checkpoint["config"]["training"]["output_dir"]).expanduser().resolve()
    model = config.model
    run_id = config.training.run_id or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    name = (
        f"am-recent_m{model.num_slots}-d{model.memory_dim}"
        f"_p{model.pending_size}-r{model.recent_size}"
        f"_inner{model.inner_steps}_{run_id}"
    )
    return Path(config.training.output_root) / name / "train"


def train(plan_dir, device="cuda", resume=None):
    config = load_training_plan(plan_dir)
    checkpoint = read_checkpoint(resume) if resume else None
    if checkpoint is not None:
        validate_training_plan(config, checkpoint)
    device = initialize_device(device)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if rank != 0:
        disable_progress_bar()
    output_dir = Path(config.training.output_dir).expanduser().resolve()
    training = config.training
    if checkpoint is not None:
        if len(checkpoint["rng"]) != world_size or len(checkpoint["query_rng"]) != world_size:
            raise ValueError("resume requires the original number of training processes")
        completed_steps = [
            int(path.parent.name.rsplit("_", 1)[1])
            for path in (output_dir / "checkpoints").glob("global_step_*/state.pt")
        ]
        if completed_steps and max(completed_steps) > checkpoint["cursor"]["step"]:
            raise ValueError("resume the latest complete checkpoint in this run")
    existing_results = [checkpoint is None and rank == 0 and any(
        (output_dir / name).exists() for name in (
            "config.json", "data.json", "metrics.jsonl", "summary.json", "checkpoints", "swanlab.json",
        )
    )]
    if world_size > 1:
        dist.broadcast_object_list(existing_results, src=0)
    if existing_results[0]:
        raise ValueError(f"training results already exist: {output_dir}; use --resume")
    global_batch_size = world_size * training.micro_batch_size * training.gradient_accumulation_steps
    if rank == 0:
        print(f"train output_dir={output_dir}", flush=True)
        print(
            f"train world_size={world_size} microbatch_per_gpu={training.micro_batch_size} "
            f"gradient_accumulation_steps={training.gradient_accumulation_steps} "
            f"effective_batch_size={global_batch_size} teacher_batch_size={config.teacher.batch_size}",
            flush=True,
        )
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
    training = replace(training, max_steps=training_steps(training, len(train_episodes), global_batch_size))
    config = replace(config, training=training)
    if checkpoint is not None and asdict(config_from_dict(checkpoint["config"])) != asdict(config):
        raise ValueError("resume requires the original run configuration and output directory")
    if rank == 0:
        steps_per_epoch = (len(train_episodes) + global_batch_size - 1) // global_batch_size
        print(
            f"train samples={len(train_episodes)} steps_per_epoch={steps_per_epoch} "
            f"epochs={training.epochs} total_steps={training.max_steps}", flush=True,
        )
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
    query_generator = torch.Generator().manual_seed(config.seed + 1 + rank)
    # 所有副本以相同 seed 初始化参数，此后各 rank 使用独立随机流。
    set_seed(config.seed + rank)
    step = 0
    if checkpoint is not None:
        step = restore_training(checkpoint, model, engine, cursor, query_generator, device)
    del checkpoint
    metrics_path = output_dir / "metrics.jsonl"
    last_checkpoint = str(checkpoint_file(resume)) if resume else None
    with ExitStack() as stack:
        tracking = stream = None
        setup_error = None
        if rank == 0:
            try:
                if resume:
                    _restore_metrics(metrics_path, step)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "config.json").write_text(
                    json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                (output_dir / "data.json").write_text(
                    json.dumps({"binding": binding, "train": train_statistics, "dev": dev_statistics},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                )
                tracking = stack.enter_context(tracking_run(config, output_dir, bool(resume), device))
                stream = stack.enter_context(metrics_path.open("a", encoding="utf-8"))
            except Exception as error:
                setup_error = f"training output setup failed: {error}"
        status = [setup_error]
        if world_size > 1:
            dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(status[0])

        while step < training.max_steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            # 各 rank 推进相同的全局游标；尾批不复制样本，空 rank 仍参与归约。
            indices = cursor.next_batch(global_batch_size)
            metrics = engine.train_batch(
                [train_episodes[index] for index in indices[rank::world_size]],
                teacher, query_generator,
            )
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            resources = torch.tensor(
                [time.perf_counter() - started,
                 torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0],
                device=device, dtype=torch.float64,
            )
            if world_size > 1:
                dist.all_reduce(resources, op=dist.ReduceOp.MAX)
            seconds, peak_memory = resources.tolist()
            record = {
                "step": step, "epoch": cursor.epoch, "epoch_end": cursor.epoch_end,
                **{f"train/{key}": value for key, value in metrics.items()},
                "resources/step_seconds": seconds,
            }
            if device.type == "cuda":
                record["resources/peak_memory_gib"] = peak_memory
            if step % training.dev_every == 0 or cursor.epoch_end or step == training.max_steps:
                rng = capture_rng(device)
                try:
                    dev = evaluate_dev(
                        model, dev_episodes[rank::world_size], teacher, training.lambda_am,
                        torch.Generator().manual_seed(config.seed + 2 + rank),
                    )
                finally:
                    restore_rng(rng, device)
                record.update({f"dev/{key}": value for key, value in dev.items()})
            if rank == 0:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if tracking is not None:
                    tracking.log({key: value for key, value in record.items() if "/" in key}, step=step)
                print(
                    f"step {step}/{training.max_steps} loss={metrics['loss']:.5f} "
                    f"QA={metrics['qa_loss']:.5f} AM={metrics['outer_am']:.5f} "
                    f"AM-output={metrics['outer_am_output']:.5f} AM-mass={metrics['outer_am_mass']:.5f} "
                    f"inner={metrics['inner_before']:.5f}->{metrics['inner_after']:.5f} "
                    f"samples={metrics['samples']} seconds={seconds:.1f}"
                    + (f" dev={record['dev/loss']:.5f}" if "dev/loss" in record else ""),
                    flush=True,
                )
            save_interval = training.save_every != "after_each_epoch" and step % training.save_every == 0
            if save_interval or cursor.epoch_end or step == training.max_steps:
                last_checkpoint = str(save_checkpoint(
                    output_dir, step, config, model, engine, cursor,
                    query_generator, binding, device,
                ))
    summary = {"step": step, "checkpoint": last_checkpoint, "output_dir": str(output_dir)}
    if rank == 0:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, help="控制进程提前保存的计划目录")
    parser.add_argument("--resume", action="store_true", help="沿用 checkpoint 配置恢复训练")
    parser.add_argument("--checkpoint", help="恢复训练的单个 checkpoint 目录")
    args = parser.parse_args(argv)
    if args.resume:
        if args.checkpoint is None:
            parser.error("--resume requires --checkpoint")
    elif args.checkpoint is not None:
        parser.error("--checkpoint requires --resume")
    try:
        train(args.plan_dir, resume=args.checkpoint if args.resume else None)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
