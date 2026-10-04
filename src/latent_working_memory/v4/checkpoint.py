"""只保存新增权重、optimizer 和精确续训所需的游标与各 rank RNG。"""

from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist


def capture_rng(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(path, engine, run, cursor):
    local_rng = capture_rng(engine.device)
    rng = [local_rng]
    if engine.world_size > 1:
        rng = [None] * engine.world_size
        dist.all_gather_object(rng, local_rng)
    if engine.rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = engine.model.trainable_state_dict()
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "run": run,
                "memory": state,
                "optimizer": engine.optimizer.state_dict(),
                "cursor": dict(cursor),
                "rng": rng,
            },
            temporary,
        )
        temporary.replace(path)
    if engine.world_size > 1:
        dist.barrier()


def load_checkpoint(path, engine, run):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if set(checkpoint) != {"run", "memory", "optimizer", "cursor", "rng"}:
        raise ValueError("checkpoint must contain run, memory, optimizer, cursor and rng")
    if checkpoint["run"] != run:
        raise ValueError("resume configuration, data, device or world size differs")
    if len(checkpoint["rng"]) != engine.world_size:
        raise ValueError("checkpoint RNG states do not match world size")
    engine.model.load_trainable_state_dict(checkpoint["memory"])
    engine.optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint["rng"][engine.rank], engine.device)
    return checkpoint["cursor"]
