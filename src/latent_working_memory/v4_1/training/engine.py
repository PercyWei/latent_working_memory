"""按完整样本轨迹累积元梯度，跨进程归约后统一更新慢参数。"""

from contextlib import ExitStack
import math

import torch
import torch.distributed as dist


GRAD_NORM_EVERY = 20


def training_steps(config, sample_count, global_batch_size):
    """epoch 与 step 均为上限；尾批计一次更新，取有效预算的较小值。"""
    budgets = []
    if config.epochs is not None:
        budgets.append(config.epochs * math.ceil(sample_count / global_batch_size))
    if config.max_steps is not None:
        budgets.append(config.max_steps)
    return min(budgets)


def _aggregate_metrics(totals, samples, target_tokens, write_events, device):
    values = torch.tensor(
        [*totals.values(), samples, target_tokens, write_events],
        dtype=torch.float64, device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(values)
    values = values.tolist()
    samples = int(values[-3])
    if samples == 0:
        raise ValueError("the global batch must contain at least one episode")
    return {
        **{name: total / samples for name, total in zip(totals, values)},
        "samples": samples,
        "target_tokens": int(values[-2]),
        "write_events": int(values[-1]),
    }


class SampleCursor:
    """epoch 为已完成轮数；尾批不补样本，完成后立即推进到下轮起点。"""

    def __init__(self, size, seed):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.offset = 0
        self.epoch_end = False
        self.order = torch.randperm(size, generator=self.generator).tolist()

    def next_batch(self, batch_size):
        indices = self.order[self.offset:self.offset + batch_size]
        self.offset += len(indices)
        self.epoch_end = self.offset == self.size
        if self.epoch_end:
            self._advance_epoch()
        return indices

    def _advance_epoch(self):
        self.epoch += 1
        self.offset = 0
        self.order = torch.randperm(self.size, generator=self.generator).tolist()

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "offset": self.offset,
            "order": self.order,
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state, step, batch_size):
        epoch, offset, order = state["epoch"], state["offset"], state["order"]
        if type(epoch) is not int or epoch < 0 or type(offset) is not int or not 0 <= offset <= self.size:
            raise ValueError("invalid checkpoint sample cursor epoch or offset")
        if not isinstance(order, list) or any(type(index) is not int for index in order) or sorted(order) != list(range(self.size)):
            raise ValueError("checkpoint sample order must be a permutation of the training samples")
        steps_per_epoch = math.ceil(self.size / batch_size)
        completed = epoch * steps_per_epoch + math.ceil(offset / batch_size)
        if (offset != self.size and offset % batch_size) or type(step) is not int or step != completed:
            raise ValueError("checkpoint sample cursor differs from its optimizer step and global batch")
        self.epoch = state["epoch"]
        self.offset = state["offset"]
        self.order = list(state["order"])
        self.generator.set_state(state["generator"])
        self.epoch_end = offset == self.size or (epoch > 0 and offset == 0)
        # 已有 checkpoint 在尾批后尚未推进；按原 next_batch 的顺序归一化。
        if self.offset == self.size:
            self._advance_epoch()


class TrainingEngine:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        groups = {"decoder": [], "memory": [], "lora": []}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.dtype != torch.float32:
                raise ValueError(f"trainable parameter must use FP32: {name}")
            category = (
                "lora" if "lora_" in name
                else "decoder" if "decoder" in name and parameter.ndim >= 2
                else "memory"
            )
            groups[category].append(parameter)
        self.parameters = [parameter for group in groups.values() for parameter in group]
        self._parameter_groups = {}
        for parameter in self.parameters:
            self._parameter_groups.setdefault((parameter.device, parameter.dtype), []).append(parameter)
        if dist.is_initialized():
            self._broadcast_parameters()
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": parameters,
                    "name": name,
                    "lr": config.lora_learning_rate if name == "lora" else config.learning_rate,
                    "weight_decay": config.weight_decay if name == "decoder" else 0.0,
                }
                for name, parameters in groups.items() if parameters
            ],
            betas=(config.beta1, config.beta2),
            eps=config.adam_epsilon,
        )
        # 第 k 次 optimizer 更新使用 f(k)，最后一次更新恰好到达 min_lr_ratio。
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda completed: self._lr_factor(completed + 1)
        )

    def _lr_factor(self, step):
        if self.config.warmup_steps and step <= self.config.warmup_steps:
            return step / self.config.warmup_steps
        span = max(1, self.config.max_steps - self.config.warmup_steps)
        progress = min(1.0, (step - self.config.warmup_steps) / span)
        cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
        return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

    @torch.no_grad()
    def _broadcast_parameters(self):
        for parameters in self._parameter_groups.values():
            flat = torch.cat([parameter.reshape(-1) for parameter in parameters])
            dist.broadcast(flat, src=0)
            offset = 0
            for parameter in parameters:
                size = parameter.numel()
                parameter.copy_(flat[offset:offset + size].view_as(parameter))
                offset += size

    @torch.no_grad()
    def _synchronize_gradients(self, samples):
        if not dist.is_initialized():
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(samples)
            return
        for (device, dtype), parameters in self._parameter_groups.items():
            size = sum(parameter.numel() for parameter in parameters)
            # 同一个归约包携带梯度和使用标记；空 rank 不缺席任何 collective。
            flat = torch.zeros(size + len(parameters), device=device, dtype=dtype)
            flat[size:] = torch.tensor(
                [parameter.grad is not None for parameter in parameters], device=device, dtype=dtype,
            )
            offset = 0
            for parameter in parameters:
                count = parameter.numel()
                if parameter.grad is not None:
                    flat[offset:offset + count].copy_(parameter.grad.reshape(-1))
                offset += count
            dist.all_reduce(flat)
            flat[:size].div_(samples)
            used = flat[size:].tolist()
            offset = 0
            for parameter, active in zip(parameters, used):
                count = parameter.numel()
                if active:
                    gradient = flat[offset:offset + count].view_as(parameter)
                    if parameter.grad is None:
                        parameter.grad = gradient.clone()
                    else:
                        parameter.grad.copy_(gradient)
                    del gradient
                else:
                    # 全局未使用的参数保持 None，避免 AdamW 对其更新或施加衰减。
                    parameter.grad = None
                offset += count

    def _accumulate_loss_gradients(self, loss, totals):
        """诊断 VJP 保留 inner 元梯度与总损失反向图，不写入 parameter.grad。"""
        if not loss.requires_grad:
            return
        gradients = torch.autograd.grad(
            loss, self.parameters, retain_graph=True, create_graph=False, allow_unused=True,
        )
        with torch.no_grad():
            for parameter, gradient in zip(self.parameters, gradients):
                if gradient is None:
                    continue
                if parameter in totals:
                    totals[parameter].add_(gradient.detach())
                else:
                    totals[parameter] = gradient.detach().clone()

    @torch.no_grad()
    def _loss_grad_norms(self, gradients, samples):
        """在共同慢参数空间先求全局样本平均梯度，再求各项 L2 范数。"""
        norms = {}
        for name, totals in gradients.items():
            squared_norm = 0.0
            for (device, dtype), parameters in self._parameter_groups.items():
                flat = torch.zeros(
                    sum(parameter.numel() for parameter in parameters), device=device, dtype=dtype,
                )
                offset = 0
                for parameter in parameters:
                    count = parameter.numel()
                    if parameter in totals:
                        flat[offset:offset + count].copy_(totals[parameter].reshape(-1))
                    offset += count
                if dist.is_initialized():
                    dist.all_reduce(flat)
                flat.div_(samples)
                squared_norm += float(torch.linalg.vector_norm(flat)) ** 2
                del flat
            norms[f"{name}_grad_norm"] = math.sqrt(squared_norm)
        return norms

    def train_batch(self, episodes, teacher, generator, step):
        """step 是本次更新的全局绝对步数；空 rank 同步参与所需归约。"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {name: 0.0 for name in (
            "loss", "qa_loss", "outer_am", "outer_am_output", "outer_am_mass",
            "inner_before", "inner_after",
        )}
        target_tokens = write_events = 0
        loss_gradients = {"qa": {}, "outer_am": {}} if step % GRAD_NORM_EVERY == 0 else None
        learning_rates = {group["name"]: group["lr"] for group in self.optimizer.param_groups}
        for start in range(0, len(episodes), self.config.micro_batch_size):
            group = episodes[start:start + self.config.micro_batch_size]
            with ExitStack() as stack:
                trajectories = [stack.enter_context(teacher.episode(episode, generator)) for episode in group]
                if len(group) == 1:
                    outputs = [self.model(
                        group[0], teacher=trajectories[0], lambda_am=self.config.lambda_am,
                        generator=generator, differentiable=True,
                    )]
                else:
                    outputs = self.model.forward_batch(
                        group, teachers=trajectories, lambda_am=self.config.lambda_am,
                        generator=generator, differentiable=True,
                    )
            # Teacher 前向已完成，反向只经过 student 的完整元训练图。
            if loss_gradients is not None:
                for name, key in (("qa", "qa_loss"), ("outer_am", "outer_am")):
                    component_loss = torch.stack([output[key] for output in outputs]).sum()
                    self._accumulate_loss_gradients(component_loss, loss_gradients[name])
            torch.stack([output["loss"] for output in outputs]).sum().backward()
            for output in outputs:
                for name in totals:
                    totals[name] += float(output[name])
                target_tokens += output["target_tokens"]
                write_events += output["write_events"]
        metrics = _aggregate_metrics(
            totals, len(episodes), target_tokens, write_events, self.parameters[0].device,
        )
        self._synchronize_gradients(metrics["samples"])
        loss_grad_norms = (
            self._loss_grad_norms(loss_gradients, metrics["samples"])
            if loss_gradients is not None else {}
        )
        del loss_gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.config.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.scheduler.step()
        return {
            **metrics,
            **loss_grad_norms,
            "grad_norm": float(grad_norm),
            **{f"lr_{name}": value for name, value in learning_rates.items()},
        }


def evaluate_dev(model, episodes, teacher, lambda_am, generator):
    """验证本 rank 分片并汇总全局指标；inner 的局部求导由模型内部启用。"""
    model.eval()
    totals = {name: 0.0 for name in (
        "loss", "qa_loss", "outer_am", "outer_am_output", "outer_am_mass",
        "inner_before", "inner_after",
    )}
    target_tokens = write_events = 0
    with torch.no_grad():
        for episode in episodes:
            with teacher.episode(episode, generator) as trajectory:
                output = model(
                    episode, teacher=trajectory, lambda_am=lambda_am,
                    generator=generator, differentiable=False,
                )
            for name in totals:
                totals[name] += float(output[name])
            target_tokens += output["target_tokens"]
            write_events += output["write_events"]
    return _aggregate_metrics(
        totals, len(episodes), target_tokens, write_events, next(model.parameters()).device,
    )
