"""单文件 checkpoint；仅在 outer optimizer step 完成后保存。"""

from dataclasses import asdict
from pathlib import Path
import shutil

import torch

from latent_working_memory.runtime import capture_rng, restore_rng


def read_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"config", "model", "optimizer", "scheduler", "cursor", "rng", "query_rng", "binding"}
    if set(state) != expected:
        raise ValueError(f"{path}: not a v4.1 training checkpoint")
    return state


def save_checkpoint(output_dir, step, config, model, engine, cursor, query_generator, binding, device):
    root = Path(output_dir) / "checkpoints"
    destination = root / f"global_step_{step}"
    destination.mkdir(parents=True, exist_ok=True)
    state = {
        "config": asdict(config),
        "model": model.trainable_state_dict(),
        "optimizer": engine.optimizer.state_dict(),
        "scheduler": engine.scheduler.state_dict(),
        "cursor": {"step": step, "samples": cursor.state_dict()},
        "rng": capture_rng(device),
        "query_rng": query_generator.get_state(),
        "binding": binding,
    }
    temporary = destination / "state.tmp"
    torch.save(state, temporary)
    temporary.replace(destination / "state.pt")
    checkpoints = sorted(
        (path for path in root.glob("global_step_*") if (path / "state.pt").exists()),
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )
    for path in checkpoints[:-config.training.save_total_limit]:
        shutil.rmtree(path)
    return destination


def restore_training(state, model, engine, cursor, query_generator, device):
    model.load_trainable_state_dict(state["model"])
    engine.optimizer.load_state_dict(state["optimizer"])
    engine.scheduler.load_state_dict(state["scheduler"])
    cursor.load_state_dict(state["cursor"]["samples"])
    query_generator.set_state(state["query_rng"])
    restore_rng(state["rng"], device)
    return state["cursor"]["step"]
