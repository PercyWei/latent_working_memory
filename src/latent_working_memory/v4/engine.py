"""verl BaseEngine＋replicated DDP；局部写入使用精确二阶 meta-gradient。"""

from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensordict import TensorDict
from verl.utils.tensordict_utils import assign_non_tensor, get_non_tensor_data
from verl.workers.config import FSDPOptimizerConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.engine import BaseEngine, EngineRegistry

@EngineRegistry.register(model_type="lwm_v4_meta", backend="replicated", device=["cpu", "cuda"])
class MetaLearningEngine(BaseEngine):
    def __init__(self, model, config, device):
        self.model, self.config, self.device = model, config, torch.device(device)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.optimizer_config = FSDPOptimizerConfig(
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            clip_grad=config.gradient_clip,
        )

    def initialize(self):
        self.parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        self.module = self.model
        if self.world_size > 1:
            self.module = DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        self.optimizer = build_optimizer(self.parameters, self.optimizer_config)

    @property
    def is_param_offload_enabled(self):
        return False

    @property
    def is_optimizer_offload_enabled(self):
        return False

    def get_data_parallel_rank(self):
        return self.rank

    def get_data_parallel_size(self):
        return self.world_size

    def get_data_parallel_group(self):
        return dist.group.WORLD if self.world_size > 1 else None

    def is_mp_src_rank_with_outputs(self):
        return True

    @contextmanager
    def train_mode(self, **kwargs):
        self.module.train()
        yield

    @contextmanager
    def eval_mode(self, **kwargs):
        was_training = self.model.training
        self.module.eval()
        try:
            yield
        finally:
            self.module.train(was_training)

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(
        self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs
    ):
        self.checkpoint_manager.save_checkpoint(
            local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False, **kwargs):
        self.checkpoint_manager.load_checkpoint(
            local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

    def optimizer_step(self):
        # Each rank uses the global target-token denominator; undo DDP averaging.
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(self.world_size)
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.config.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        return float(norm)

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        episodes = get_non_tensor_data(data, "episodes", None)
        boundary = self.model.config.pending_size + self.model.config.recent_size
        token_counts = [len(episode.input_ids) - boundary - 1 for episode in episodes]
        total_tokens = sum(token_counts)
        if not episodes or min(token_counts) < 1:
            raise ValueError("each episode must have a next-token target after its first write")
        local = list(range(self.rank, len(episodes), self.world_size))
        local.sort(key=lambda index: len(episodes[index].input_ids), reverse=True)
        size = self.config.micro_batch_size_per_gpu
        # Empty tail ranks still enter a real DDP forward and zero-weight backward.
        work = [local[start : start + size] for start in range(0, len(local), size)]
        if not work and not forward_only:
            work = [[0]]
        totals = torch.zeros(5, dtype=torch.float64, device=self.device)
        for position, indices in enumerate(work):
            expected_tokens = sum(token_counts[index] for index in indices)
            sync = (
                self.module.no_sync()
                if not forward_only and self.world_size > 1 and position + 1 < len(work)
                else nullcontext()
            )
            with sync:
                # No outer autocast: the model controls backbone dtype; fast state is fp32.
                output = self.module(
                    [
                        torch.tensor(episodes[index].input_ids, dtype=torch.long, device=self.device)
                        for index in indices
                    ],
                    differentiable=not forward_only,
                )
                if output["target_tokens"] != expected_tokens:
                    raise ValueError("model target count does not match the streaming boundary")
                if not forward_only:
                    loss = output["loss"] * (expected_tokens if local else 0) / total_tokens
                    loss.backward()
                    del loss
            if local:
                totals += torch.tensor(
                    [
                        float(output["loss"].detach()) * expected_tokens,
                        expected_tokens,
                        len(indices),
                        output["write_events"],
                        float(output["write_loss"]) * output["write_events"],
                    ],
                    dtype=torch.float64,
                    device=self.device,
                )
            del output
        if self.world_size > 1:
            dist.all_reduce(totals)
        nll, targets, samples, writes, write_loss = totals.tolist()
        return {
            "metrics": {
                "loss": nll / targets,
                "target_tokens": int(targets),
                "samples": int(samples),
                "source_tokens": sum(len(episode.input_ids) for episode in episodes),
                "write_events": int(writes),
                "write_loss": write_loss / writes,
            }
        }

    def step(self, episodes):
        data = TensorDict({}, batch_size=[])
        assign_non_tensor(data, episodes=tuple(episodes))
        with self.train_mode():
            return self.train_batch(data, loss_function=None)["metrics"]

    def eval_batch(self, episodes):
        data = TensorDict({}, batch_size=[])
        assign_non_tensor(data, episodes=tuple(episodes))
        with self.eval_mode():
            # BaseEngine.no_grad disables the reader graph. The model locally enables
            # gradients for its inference-time memory updates.
            return self.infer_batch(data, loss_function=None)["metrics"]
