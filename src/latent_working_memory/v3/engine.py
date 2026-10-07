"""verl 训练生命周期与 replicated DDP，按完整记忆轨迹平均损失。"""

from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensordict import TensorDict
from verl.utils.tensordict_utils import assign_non_tensor, get_non_tensor_data
from verl.workers.config import FSDPOptimizerConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.engine import BaseEngine, EngineRegistry

from latent_working_memory.v3.pretrain_data import PretrainExample
from latent_working_memory.v4.engine import initialize_device


@EngineRegistry.register(model_type="lwm_v3_token", backend="replicated", device=["cpu", "cuda"])
class TokenMemoryEngine(BaseEngine):
    """每次 model.forward 并行展开一个 microbatch，内部读写不触发分布式通信。"""

    def __init__(self, model, config, device):
        self.model, self.config, self.device = model, config, torch.device(device)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.global_batch_size = config.global_batch_size(self.world_size)
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
                find_unused_parameters=True,
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
        # backward 已除以全局真实轨迹数，抵消 DDP 对梯度额外执行的 rank 平均。
        if self.world_size > 1:
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(self.world_size)
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.optimizer_config.clip_grad, error_if_nonfinite=True
        )
        self.optimizer.step()
        return float(norm)

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        examples = get_non_tensor_data(data, "examples", None)
        epoch = get_non_tensor_data(data, "epoch", None)
        if not examples:
            raise ValueError("a batch must contain at least one trajectory")
        local = list(range(self.rank, len(examples), self.world_size))
        if isinstance(examples[0], PretrainExample):
            # 仅重排当前 rank 已分得的样本；优先接近读取长度，再接近写入长度。
            local.sort(
                key=lambda index: (
                    len(examples[index].target_ids),
                    len(examples[index].input_ids),
                ),
                reverse=True,
            )
        # 尾批空 rank 也执行真实 DDP forward/backward，仅将其训练权重置零。
        size = self.config.micro_batch_size_per_gpu
        work = [local[start : start + size] for start in range(0, len(local), size)] or [[0]]
        metric_names, totals = None, None
        for position, indices in enumerate(work):
            count = len(indices) if local else 0
            sync = (
                self.module.no_sync()
                if not forward_only and self.world_size > 1 and position + 1 < len(work)
                else nullcontext()
            )
            with sync:
                precision = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with precision:
                    output = self.module(
                        [examples[index] for index in indices],
                        epoch=epoch,
                        differentiable=not forward_only,
                        batched=True,
                    )
                if not forward_only:
                    loss = output["loss"] * count / len(examples)
                    loss.backward()
                    del loss
            names = sorted(output["metrics"])
            if metric_names is None:
                if {"loss", "samples", "grad_norm"}.intersection(names):
                    raise ValueError("model metrics must not redefine loss, samples or grad_norm")
                metric_names = names
                totals = torch.zeros(len(names) + 2, dtype=torch.float64, device=self.device)
            elif names != metric_names:
                raise ValueError("trajectory metric keys must remain fixed within a batch")
            if local:
                totals += torch.tensor(
                    [
                        float(output["loss"].detach()) * count,
                        count,
                        *(output["metrics"][name] * count for name in metric_names),
                    ],
                    dtype=torch.float64,
                    device=self.device,
                )
            del output
        if self.world_size > 1:
            dist.all_reduce(totals)
        total_loss, samples, *values = totals.tolist()
        return {
            "metrics": {
                "loss": total_loss / samples,
                "samples": int(samples),
                **{name: value / samples for name, value in zip(metric_names, values, strict=True)},
            }
        }

    def step(self, examples, epoch=0):
        data = TensorDict({}, batch_size=[])
        assign_non_tensor(data, examples=tuple(examples), epoch=epoch)
        with self.train_mode():
            return self.train_batch(data, loss_function=None)["metrics"]

    def eval_batch(self, examples, epoch=0):
        data = TensorDict({}, batch_size=[])
        assign_non_tensor(data, examples=tuple(examples), epoch=epoch)
        with self.eval_mode():
            metrics = self.infer_batch(data, loss_function=None)["metrics"]
        metrics["grad_norm"] = None
        return metrics


__all__ = ["TokenMemoryEngine", "EngineRegistry", "initialize_device"]
