"""单文件 checkpoint：可变权重、optimizer、epoch 游标与各 rank RNG；不含数据集。"""

from pathlib import Path
import random

import torch
import torch.distributed as dist


def codec_state(codec):
    return {
        name: p.detach().cpu().clone()
        for name, p in codec.named_parameters()
        if name.startswith(("read_alignment.", "write_alignment.", "compression."))
        or name.startswith("backbone.")
        and "lora_" in name
    }


def restore_codec(codec, state):
    expected = {
        name
        for name, _ in codec.named_parameters()
        if name.startswith(("read_alignment.", "write_alignment.", "compression."))
        or name.startswith("backbone.")
        and "lora_" in name
    }
    if set(state) != set(expected):
        raise ValueError("checkpoint codec parameters differ from this architecture")
    codec.load_state_dict(state, strict=False)


def capture_rng(device):
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(path, engine, run, cursor):
    rng = [capture_rng(engine.device)]
    if engine.world_size > 1:
        rng = [None] * engine.world_size
        dist.all_gather_object(rng, capture_rng(engine.device))
    if engine.rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "run": run,
                "stage": engine.model.codec.stage,
                "codec": codec_state(engine.model.codec),
                "optimizer": engine.optimizer.state_dict(),
                "cursor": cursor,
                "rng": rng,
            },
            temporary,
        )
        temporary.replace(path)
