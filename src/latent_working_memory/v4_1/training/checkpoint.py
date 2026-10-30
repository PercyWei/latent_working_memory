"""单文件 checkpoint；仅在 outer optimizer step 完成后保存。"""

from dataclasses import asdict
from pathlib import Path
import shutil

import torch
import torch.distributed as dist

from latent_working_memory.runtime import capture_rng, restore_rng


def checkpoint_file(path):
    """CLI 可接收目录或文件；持久化引用统一为绝对 state.pt 路径。"""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "state.pt"
    return path


def read_checkpoint(path):
    path = checkpoint_file(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"config", "model", "optimizer", "scheduler", "cursor", "rng", "query_rng", "binding"}
    if set(state) != expected:
        raise ValueError(f"{path}: not a v4.1 training checkpoint")
    # 已有单卡 checkpoint 在读取边界归一化，新写出统一按 rank 保存。
    if isinstance(state["rng"], dict):
        state["rng"] = [state["rng"]]
        state["query_rng"] = [state["query_rng"]]
    return state


def save_checkpoint(output_dir, step, config, model, engine, cursor, query_generator, binding, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # capture_rng 与 Generator.get_state 均返回 CPU 随机状态，不传输模型参数。
    local_rng = (capture_rng(device), query_generator.get_state())
    rank_rng = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(local_rng, rank_rng, dst=0)
    else:
        rank_rng[0] = local_rng

    root = Path(output_dir).expanduser().resolve() / "checkpoints"
    destination = root / f"global_step_{step}"
    destination_file = destination / "state.pt"
    write_error = None
    if rank == 0:
        try:
            destination.mkdir(parents=True, exist_ok=True)
            state = {
                "config": asdict(config),
                "model": model.trainable_state_dict(),
                "optimizer": engine.optimizer.state_dict(),
                "scheduler": engine.scheduler.state_dict(),
                "cursor": {"step": step, "samples": cursor.state_dict()},
                "rng": [rng for rng, _ in rank_rng],
                "query_rng": [query_rng for _, query_rng in rank_rng],
                "binding": binding,
            }
            temporary = destination / "state.tmp"
            torch.save(state, temporary)
            temporary.replace(destination_file)
            checkpoints = sorted(
                (path for path in root.glob("global_step_*") if (path / "state.pt").exists()),
                key=lambda path: int(path.name.rsplit("_", 1)[1]),
            )
            for path in checkpoints[:-config.training.save_total_limit]:
                shutil.rmtree(path)
        except Exception as error:
            write_error = error

    # rank 0 写盘失败也通知其余 rank，避免它们停在保存后的同步点。
    status = [None if write_error is None else f"{type(write_error).__name__}: {write_error}"]
    if world_size > 1:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        if write_error is not None:
            raise write_error
        raise RuntimeError(f"rank 0 failed to save {destination_file}: {status[0]}")
    return destination_file


def restore_training(state, model, engine, cursor, query_generator, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if any(
        not isinstance(states, list) or len(states) != world_size
        for states in (state["rng"], state["query_rng"])
    ):
        raise ValueError(f"checkpoint RNG state must contain one entry per rank (world_size={world_size})")
    step = state["cursor"]["step"]
    if type(step) is not int or not 0 <= step <= engine.config.max_steps:
        raise ValueError("checkpoint optimizer step is outside the training budget")
    if state["scheduler"]["last_epoch"] != step:
        raise ValueError("checkpoint scheduler differs from its optimizer step")
    batch_size = world_size * engine.config.micro_batch_size * engine.config.gradient_accumulation_steps
    cursor.load_state_dict(state["cursor"]["samples"], step, batch_size)
    model.load_trainable_state_dict(state["model"])
    engine.optimizer.load_state_dict(state["optimizer"])
    engine.scheduler.load_state_dict(state["scheduler"])
    query_generator.set_state(state["query_rng"][rank])
    restore_rng(state["rng"][rank], device)
    return step
