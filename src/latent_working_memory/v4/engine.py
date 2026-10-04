"""verl BaseEngine＋replicated DDP；局部写入使用精确二阶 meta-gradient。"""

from contextlib import contextmanager, nullcontext
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensordict import TensorDict
from verl.utils.distributed import initialize_global_process_group
from verl.utils.tensordict_utils import assign_non_tensor, get_non_tensor_data
from verl.workers.config import FSDPOptimizerConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.engine import BaseEngine, EngineRegistry


def initialize_device(device):
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("v4 meta-training supports CPU and CUDA")
    if device.type == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if not visible or any(value.strip() not in {"0", "1"} for value in visible):
            raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select physical GPUs 0/1")
        index = int(os.environ.get("LOCAL_RANK", device.index or 0))
        if not 0 <= index < len(visible):
            raise ValueError("the requested logical CUDA device is not visible")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        if device.type == "cuda":
            initialize_global_process_group(timeout_second=7200)
        else:
            dist.init_process_group("gloo", timeout=timedelta(hours=2))
    return device


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
        # Empty tail ranks still enter a real DDP forward and zero-weight backward.
        work = local or ([0] if not forward_only else [])
        totals = torch.zeros(5, dtype=torch.float64, device=self.device)
        for position, index in enumerate(work):
            sync = (
                self.module.no_sync()
                if not forward_only and self.world_size > 1 and position + 1 < len(work)
                else nullcontext()
            )
            with sync:
                # No outer autocast: the model controls backbone dtype; fast state is fp32.
                output = self.module(
                    torch.tensor(episodes[index].input_ids, dtype=torch.long, device=self.device),
                    differentiable=not forward_only,
                )
                if output["target_tokens"] != token_counts[index]:
                    raise ValueError("model target count does not match the streaming boundary")
                if not forward_only:
                    loss = output["loss"] * (token_counts[index] if local else 0) / total_tokens
                    loss.backward()
            if local:
                totals += torch.tensor(
                    [
                        float(output["loss"].detach()) * token_counts[index],
                        token_counts[index],
                        1,
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
