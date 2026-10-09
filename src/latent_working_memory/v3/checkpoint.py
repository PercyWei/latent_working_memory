"""verl checkpoint 生命周期；目录内仅保存新增权重与精确续训状态。"""

from enum import Enum
from pathlib import Path
import re

import torch
import torch.distributed as dist
from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler, OrchestrationMode
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

from latent_working_memory.v3.config import DYNAMIC_METHODS
from latent_working_memory.v4.checkpoint import capture_rng, restore_rng


def read_checkpoint(path):
    checkpoint = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "run",
        "trainable",
        "optimizer",
        "cursor",
        "rng",
    }:
        raise ValueError("checkpoint requires exactly run, trainable, optimizer, cursor and rng")
    objective = checkpoint["run"]["config"]["objective"]
    if (
        objective["method"] in DYNAMIC_METHODS
        and objective["stage"] in {"warmup", "policy"}
        and "append_slots" not in objective
    ):
        raise ValueError(
            "dynamic QA checkpoint requires explicit objective.append_slots; "
            "cannot infer the capacity policy from current defaults"
        )
    return checkpoint


class TokenMemoryCheckpointManager(BaseCheckpointManager):
    """replicated 模型的小权重序列化；目录保留由 verl 基类实现。"""

    def __init__(self, engine, run, cursor, checkpoint_directory):
        # 基类构造面向分片模型并要求进程组；这里使用真实 replicated engine 上下文。
        self.engine, self.run, self.cursor = engine, run, cursor
        self.rank, self.world_size = engine.rank, engine.world_size
        self.previous_global_step = None
        self.previous_saved_paths = [
            str(path)
            for path in sorted(
                Path(checkpoint_directory).glob("global_step_*"),
                key=lambda path: int(path.name.removeprefix("global_step_")),
            )
            if (path / "state.pt").is_file()
            and all((path / f"data_{rank}.pt").is_file() for rank in range(self.world_size))
        ]

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        local_rng = capture_rng(self.engine.device)
        rng = [local_rng]
        if self.world_size > 1:
            rng = [None] * self.world_size
            dist.all_gather_object(rng, local_rng)
        if self.rank == 0:
            directory = Path(local_path)
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / "state.tmp"
            torch.save(
                {
                    "run": self.run,
                    "trainable": self.engine.model.trainable_state_dict(),
                    "optimizer": self.engine.optimizer.state_dict(),
                    "cursor": dict(self.cursor),
                    "rng": rng,
                },
                temporary,
            )
        self.previous_global_step = global_step
        if self.world_size > 1:
            dist.barrier()

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        checkpoint = read_checkpoint(local_path)
        if checkpoint["run"] != self.run:
            raise ValueError(
                "resume configuration, data, initialization, device or world size differs"
            )
        if len(checkpoint["rng"]) != self.world_size:
            raise ValueError("checkpoint RNG states do not match world size")
        self.engine.model.load_trainable_state_dict(checkpoint["trainable"])
        self.engine.optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"][self.rank], self.engine.device)
        self.cursor.clear()
        self.cursor.update(checkpoint["cursor"])


class LocalOrchestrationMode(Enum):
    LOCAL = "local"


class TokenMemoryCheckpointHandler(CheckpointHandler):
    """复用 verl 的 loader／tracker 保存与恢复，支持无进程组的单进程执行。"""

    def __init__(
        self, engine, train_dataloader, checkpoint_directory, run, cursor, resume_from_path=None
    ):
        self.default_local_dir = str(Path(checkpoint_directory).resolve())
        self.max_ckpt_to_keep = engine.config.save_total_limit
        self.default_hdfs_dir = None
        self.resume_mode = "resume_path" if resume_from_path is not None else "disable"
        self.resume_from_path = str(Path(resume_from_path).resolve()) if resume_from_path else None
        self.engine = engine
        self.train_dataloader = train_dataloader
        self.mode = (
            OrchestrationMode.SPMD if dist.is_initialized() else LocalOrchestrationMode.LOCAL
        )
        self.lora_train_meta = None
        self.rank = engine.rank
        self.is_mp_src_rank_with_outputs = engine.is_mp_src_rank_with_outputs()
        self.dp_rank = engine.get_data_parallel_rank()
        engine.checkpoint_manager = TokenMemoryCheckpointManager(
            engine, run, cursor, self.default_local_dir
        )

    def save_checkpoint(self, step):
        directory = Path(self.default_local_dir) / f"global_step_{step}"
        if self.rank == 0:
            # state.pt 同时表示整套 checkpoint 已保存成功；重写时先撤销旧完成边界。
            (directory / "state.pt").unlink(missing_ok=True)
        if self.mode == OrchestrationMode.SPMD:
            dist.barrier()
        super().save_checkpoint(step)
        if self.rank == 0:
            (directory / "state.tmp").replace(directory / "state.pt")
            manager = self.engine.checkpoint_manager
            manager.previous_saved_paths = [
                path for path in manager.previous_saved_paths if path != str(directory)
            ]
            manager.register_checkpoint(str(directory), self.max_ckpt_to_keep)
        if self.mode == OrchestrationMode.SPMD:
            dist.barrier()

    def _determine_resume_path(self):
        directory = super()._determine_resume_path()
        if directory is not None:
            if re.fullmatch(r"global_step_[0-9]+", Path(directory).name) is None:
                raise ValueError("checkpoint directory must be named global_step_<step>")
            for path in (Path(directory) / "state.pt", Path(directory) / f"data_{self.dp_rank}.pt"):
                if not path.is_file():
                    raise FileNotFoundError(path)
        return directory
