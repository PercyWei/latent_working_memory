"""完整 optimizer step 的单文件 checkpoint 与各 rank 随机流。"""

import os
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist

from latent_working_memory.runtime import capture_rng, restore_rng
from ..config import config_from_dict, config_to_dict


def checkpoint_file(path):
    path = Path(path).expanduser().resolve()
    return path / "state.pt" if path.is_dir() else path


def read_checkpoint(path):
    path = checkpoint_file(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"config", "model", "optimizer", "scheduler", "cursor", "rng", "boundary_rng", "binding"}
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError(f"{path}: not a v3.1.1 training checkpoint")
    config = config_from_dict(state["config"])
    step = state["cursor"]["step"]
    if type(step) is not int or step < 1 or path.parent.name != f"global_step_{step}":
        raise ValueError("checkpoint directory and completed optimizer step disagree")
    if (
        not isinstance(state["rng"], list) or not state["rng"]
        or not isinstance(state["boundary_rng"], list)
        or len(state["rng"]) != len(state["boundary_rng"])
        or any((not isinstance(value, torch.Tensor) if config.stage == "post" else value is not None)
               for value in state["boundary_rng"])
    ):
        raise ValueError("checkpoint random states must contain one stage-appropriate entry per rank")
    return state


def _cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def save_checkpoint(output_dir, step, config, model, engine, cursor, boundary_generator, binding, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    local = (capture_rng(device), None if boundary_generator is None else boundary_generator.get_state())
    states = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local, states, dst=0)
    else:
        states[0] = local
    root = Path(output_dir).expanduser().resolve() / "checkpoints"
    destination = root / f"global_step_{step}" / "state.pt"
    error = None
    if rank == 0:
        temporary = None
        try:
            if destination.exists():
                raise FileExistsError(f"checkpoint already committed: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "config": config_to_dict(config), "model": model.trainable_state_dict(),
                "optimizer": _cpu_state(engine.optimizer.state_dict()),
                "scheduler": engine.scheduler.state_dict(),
                "cursor": {"step": step, "samples": cursor.state_dict()},
                "rng": [rng for rng, _ in states],
                "boundary_rng": [boundary for _, boundary in states], "binding": binding,
            }
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".state.", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            checkpoints = sorted(
                (path for path in root.glob("global_step_*") if (path / "state.pt").is_file()),
                key=lambda path: int(path.name.rsplit("_", 1)[1]),
            )
            for old in checkpoints[:-config.training.loop.save_total_limit]:
                shutil.rmtree(old)
        except Exception as exc:
            error = exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    status = [None if error is None else f"{type(error).__name__}: {error}"]
    if world > 1:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        if error is not None:
            raise error
        raise RuntimeError(f"rank 0 failed to save {destination}: {status[0]}")
    return destination


def restore_training(state, model, engine, cursor, boundary_generator, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    if any(len(values) != world for values in (state["rng"], state["boundary_rng"])):
        raise ValueError("resume requires the original number of training processes")
    step = state["cursor"]["step"]
    if not 0 <= step <= engine.config.loop.max_steps or state["scheduler"]["last_epoch"] != step:
        raise ValueError("checkpoint step or scheduler differs from the training budget")
    batch = world * engine.config.loop.micro_batch_size * engine.config.loop.gradient_accumulation_steps
    cursor.load_state_dict(state["cursor"]["samples"], step, batch)
    model.load_trainable_state_dict(state["model"])
    engine.optimizer.load_state_dict(state["optimizer"])
    engine.scheduler.load_state_dict(state["scheduler"])
    boundary_state = state["boundary_rng"][rank]
    if boundary_generator is not None:
        boundary_generator.set_state(boundary_state)
    elif boundary_state is not None:
        raise ValueError("pretraining cannot restore a boundary sampling stream")
    restore_rng(state["rng"][rank], device)
    return step
