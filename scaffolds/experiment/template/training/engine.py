"""按实际样本数累积与归约梯度，不叠加 DDP 的隐式平均。"""

import math

import torch
import torch.distributed as dist

from .. import method


def aggregate_metrics(totals, samples):
    parts = [(totals, samples)]
    if dist.is_initialized():
        parts = [None] * dist.get_world_size()
        dist.all_gather_object(parts, (totals, samples))
    count = sum(size for _, size in parts)
    if count == 0:
        raise ValueError("a global batch must contain at least one sample")
    sums = {}
    for metrics, _ in parts:
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + value
    return {**{name: value / count for name, value in sums.items()}, "samples": count}


class TrainingEngine:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.training = config["training"]
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.groups = {}
        for parameter in self.parameters:
            self.groups.setdefault((parameter.device, parameter.dtype), []).append(parameter)
        if dist.is_initialized():
            with torch.no_grad():
                for parameter in self.parameters:
                    dist.broadcast(parameter, src=0)
        self.optimizer = method.build_optimizer(model, config)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda completed: self._lr_factor(completed + 1),
        )

    def _lr_factor(self, step):
        warmup = self.training["warmup_steps"]
        if warmup and step <= warmup:
            return step / warmup
        progress = min(1.0, (step - warmup) / max(1, self.training["max_steps"] - warmup))
        cosine = (1 + math.cos(math.pi * progress)) / 2
        minimum = self.training["min_lr_ratio"]
        return minimum + (1 - minimum) * cosine

    @torch.no_grad()
    def _synchronize_gradients(self, samples):
        for (device, dtype), parameters in self.groups.items():
            size = sum(parameter.numel() for parameter in parameters)
            # 使用标记与梯度一同归约；全局未用参数保持 None，避免额外权重衰减。
            flat = torch.zeros(size + len(parameters), device=device, dtype=dtype)
            flat[size:] = torch.tensor(
                [parameter.grad is not None for parameter in parameters], device=device, dtype=dtype
            )
            offset = 0
            for parameter in parameters:
                count = parameter.numel()
                if parameter.grad is not None:
                    flat[offset : offset + count].copy_(parameter.grad.reshape(-1))
                offset += count
            if dist.is_initialized():
                dist.all_reduce(flat)
            flat[:size].div_(samples)
            offset = 0
            for parameter, used in zip(parameters, flat[size:].tolist()):
                count = parameter.numel()
                if used:
                    gradient = flat[offset : offset + count].view_as(parameter)
                    if parameter.grad is None:
                        parameter.grad = gradient.clone()
                    else:
                        parameter.grad.copy_(gradient)
                else:
                    parameter.grad = None
                offset += count

    def train_batch(self, samples, generator):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {}
        micro = self.training["micro_batch_size"]
        learning_rate = self.optimizer.param_groups[0]["lr"]
        for start in range(0, len(samples), micro):
            outputs = method.compute_losses(self.model, samples[start : start + micro], generator)
            outputs["loss"].sum().backward()
            for name, values in outputs.items():
                totals[name] = totals.get(name, 0.0) + values.detach().sum().item()
        metrics = aggregate_metrics(totals, len(samples))
        self._synchronize_gradients(metrics["samples"])
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.training["gradient_clip"])
        self.optimizer.step()
        self.scheduler.step()
        return {**metrics, "grad_norm": float(norm), "learning_rate": learning_rate}


def evaluate_dev(model, samples, micro_batch_size):
    """方法内部可保留 TTT 所需梯度；执行层不强制 inference_mode。"""
    model.eval()
    totals = {}
    for start in range(0, len(samples), micro_batch_size):
        outputs = method.compute_losses(model, samples[start : start + micro_batch_size])
        for name, values in outputs.items():
            totals[name] = totals.get(name, 0.0) + values.detach().sum().item()
    return aggregate_metrics(totals, len(samples))
