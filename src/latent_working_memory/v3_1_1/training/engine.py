"""按完整文档累积梯度，跨进程归约后统一更新压缩器和边界模块。"""

from dataclasses import asdict
import math

import torch
import torch.distributed as dist

from ..objective import post_backward
from ..rollout import execute, qa_nll


def _curriculum_sizes(bin_sizes, replay_ratio):
    sizes, history = [], 0
    for count in bin_sizes:
        replay = min(history, int(count * replay_ratio + 0.5))
        sizes.append(count + replay)
        history += count
    return sizes


def training_steps(config, sample_count, global_batch_size, bin_sizes=None):
    budgets = []
    if bin_sizes is not None:
        sizes = _curriculum_sizes(bin_sizes, config.curriculum_replay_ratio)
        budgets.append(config.curriculum_stage_epochs * sum(
            math.ceil(size / global_batch_size) for size in sizes
        ))
    elif config.epochs is not None:
        budgets.append(config.epochs * math.ceil(sample_count / global_batch_size))
    if config.max_steps is not None:
        budgets.append(config.max_steps)
    return min(budgets)


class SampleCursor:
    """普通全量随机，或按档位逐阶段训练并补充历史；尾批不跨轮。"""

    def __init__(self, size, seed, bin_sizes=None, stage_epochs=1, replay_ratio=0.2):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = self.offset = 0
        self.epoch_end = False
        self.bin_sizes = None if bin_sizes is None else tuple(bin_sizes)
        self.stage_epochs = stage_epochs
        self.stage_epoch = 0
        self.stage = None
        if self.bin_sizes is not None:
            self.pass_sizes = _curriculum_sizes(self.bin_sizes, replay_ratio)
            self.stage = next(index for index, count in enumerate(self.bin_sizes) if count)
        self.order = self._make_order()

    def _make_order(self):
        if self.bin_sizes is None:
            return torch.randperm(self.size, generator=self.generator).tolist()
        if self.stage == len(self.bin_sizes):
            return []
        start = sum(self.bin_sizes[:self.stage])
        indices = list(range(start, start + self.bin_sizes[self.stage]))
        remaining = self.pass_sizes[self.stage] - len(indices)
        if remaining:
            candidates = [index for index in torch.randperm(
                self.stage, generator=self.generator,
            ).tolist() if self.bin_sizes[index]]
            quotas = [0] * self.stage
            while remaining:
                share = max(1, remaining // len(candidates))
                for index in candidates:
                    take = min(share, self.bin_sizes[index] - quotas[index], remaining)
                    quotas[index] += take
                    remaining -= take
                candidates = [index for index in candidates
                              if quotas[index] < self.bin_sizes[index]]
            offset = 0
            for count, quota in zip(self.bin_sizes, quotas):
                if quota:
                    indices.extend((torch.randperm(count, generator=self.generator)[:quota]
                                    + offset).tolist())
                offset += count
        return [indices[index] for index in torch.randperm(
            len(indices), generator=self.generator,
        ).tolist()]

    def next_batch(self, batch_size):
        if not self.order:
            raise StopIteration("length curriculum is complete")
        indices = self.order[self.offset:self.offset + batch_size]
        self.offset += len(indices)
        self.epoch_end = self.offset == len(self.order)
        if self.epoch_end:
            self.epoch += 1
            self.offset = 0
            if self.bin_sizes is not None:
                self.stage_epoch += 1
                if self.stage_epoch == self.stage_epochs:
                    self.stage_epoch = 0
                    self.stage = next(
                        (index for index in range(self.stage + 1, len(self.bin_sizes))
                         if self.bin_sizes[index]), len(self.bin_sizes),
                    )
            self.order = self._make_order()
        return indices

    def state_dict(self):
        return {
            "epoch": self.epoch, "offset": self.offset, "order": list(self.order),
            "stage": self.stage, "stage_epoch": self.stage_epoch,
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state, step, batch_size):
        epoch, offset, order = state["epoch"], state["offset"], state["order"]
        stage, stage_epoch = state["stage"], state["stage_epoch"]
        if (
            set(state) != {"epoch", "offset", "order", "stage", "stage_epoch", "generator"}
            or type(epoch) is not int or epoch < 0
            or type(offset) is not int or offset < 0 or offset % batch_size
            or not isinstance(order, list) or any(type(index) is not int for index in order)
        ):
            raise ValueError("checkpoint has an invalid document cursor")
        if self.bin_sizes is None:
            if (stage is not None or stage_epoch != 0 or offset >= self.size
                or sorted(order) != list(range(self.size))):
                raise ValueError("checkpoint has an invalid document cursor")
            completed = epoch * math.ceil(self.size / batch_size) + math.ceil(offset / batch_size)
        else:
            if (type(stage) is not int or not 0 <= stage <= len(self.bin_sizes)
                or type(stage_epoch) is not int or not 0 <= stage_epoch < self.stage_epochs):
                raise ValueError("checkpoint has an invalid curriculum stage")
            completed = self.stage_epochs * sum(
                math.ceil(size / batch_size) for size in self.pass_sizes[:stage]
            )
            expected_epoch = self.stage_epochs * sum(count > 0 for count in self.bin_sizes[:stage])
            if stage == len(self.bin_sizes):
                if stage_epoch or offset or order:
                    raise ValueError("completed curriculum must have an empty cursor")
            else:
                count = self.bin_sizes[stage]
                start = sum(self.bin_sizes[:stage])
                selected = set(order)
                if (not count or len(order) != self.pass_sizes[stage]
                    or offset >= len(order) or len(selected) != len(order)
                    or any(not 0 <= index < start + count for index in order)
                    or not set(range(start, start + count)) <= selected):
                    raise ValueError("checkpoint has an invalid curriculum selection")
                completed += stage_epoch * math.ceil(len(order) / batch_size)
                completed += math.ceil(offset / batch_size)
                expected_epoch += stage_epoch
            if epoch != expected_epoch:
                raise ValueError("checkpoint epoch differs from its curriculum stage")
        if type(step) is not int or step != completed:
            raise ValueError("checkpoint document cursor differs from its optimizer step")
        self.epoch, self.offset, self.order = epoch, offset, list(order)
        self.stage, self.stage_epoch = stage, stage_epoch
        self.epoch_end = epoch > 0 and offset == 0
        self.generator.set_state(state["generator"])


def document_loss(model, document, config):
    """预训练目标或后训练的确定性 dev 评分；不执行反向传播。"""
    if config.stage == "pretrain":
        memory = model.compress(document.input_ids)
        loss = model.answer_nll((memory,), document.prompt_ids, document.target_ids)
        return loss, {
            f"{document.task}_nll": (float(loss.detach()), 1),
            "lm_fraction": (float(document.task == "lm"), 1),
        }
    blocks, actions, _, entropy, lengths = execute(
        model, document.input_ids, config.method.rollout.unit_tokens,
    )
    loss = qa_nll(model, blocks, document)
    return loss, {
        "qa_nll": (float(loss), 1), "blocks": (len(blocks), 1),
        "memory_slots": (len(blocks) * model.memory_slots, 1),
        "segment_tokens": (sum(lengths), len(lengths)),
        "commit_rate": (sum(actions), len(actions)),
        "policy_entropy": (float(entropy.sum()), len(actions)),
    }


def _accumulate(totals, loss, metrics):
    for name, (numerator, denominator) in {"loss": (float(loss.detach()), 1), **metrics}.items():
        pair = totals.setdefault(name, [0.0, 0.0])
        pair[0] += numerator
        pair[1] += denominator


def _aggregate_metrics(totals, samples):
    local = (totals, samples)
    if dist.is_initialized():
        shards = [None] * dist.get_world_size()
        dist.all_gather_object(shards, local)
    else:
        shards = [local]
    combined, global_samples = {}, 0
    for shard, count in shards:
        global_samples += count
        for name, (numerator, denominator) in shard.items():
            pair = combined.setdefault(name, [0.0, 0.0])
            pair[0] += numerator
            pair[1] += denominator
    if global_samples == 0:
        raise ValueError("the global selection must contain at least one document")
    return {
        **{name: numerator / denominator for name, (numerator, denominator) in combined.items()
           if denominator},
        "samples": global_samples,
    }


class TrainingEngine:
    def __init__(self, model, config):
        self.model, self.experiment, self.config = model, config, config.training
        if config.stage == "post":
            model.boundary.requires_grad_(True)
        groups = [{
            "name": "compressor", "params": list(model.compressor_parameters()),
            "lr": self.config.optimizer.compressor_learning_rate,
        }]
        if config.stage == "post":
            groups.append({
                "name": "boundary", "params": list(model.boundary_parameters()),
                "lr": self.config.optimizer.boundary_learning_rate,
            })
        self.parameters = [parameter for group in groups for parameter in group["params"]]
        if any(parameter.dtype != torch.float32 for parameter in self.parameters):
            raise ValueError("all trainable compressor and boundary parameters must use FP32")
        if dist.is_initialized():
            self._broadcast_parameters()
        self.optimizer = torch.optim.AdamW(
            groups, betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
            eps=self.config.optimizer.adam_epsilon, weight_decay=self.config.optimizer.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda completed: self._lr_factor(completed + 1),
        )

    def _lr_factor(self, step):
        if self.config.scheduler.warmup_steps and step <= self.config.scheduler.warmup_steps:
            return step / self.config.scheduler.warmup_steps
        progress = min(1.0, (step - self.config.scheduler.warmup_steps)
                       / max(1, self.config.loop.max_steps - self.config.scheduler.warmup_steps))
        return self.config.scheduler.min_lr_ratio + (1 - self.config.scheduler.min_lr_ratio) * (
            1 + math.cos(math.pi * progress)
        ) / 2

    @torch.no_grad()
    def _broadcast_parameters(self):
        flat = torch.cat([parameter.reshape(-1) for parameter in self.parameters])
        dist.broadcast(flat, src=0)
        offset = 0
        for parameter in self.parameters:
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
        size = sum(parameter.numel() for parameter in self.parameters)
        flat = torch.zeros(size + len(self.parameters), device=self.parameters[0].device)
        flat[size:] = torch.tensor(
            [parameter.grad is not None for parameter in self.parameters], device=flat.device,
        )
        offset = 0
        for parameter in self.parameters:
            count = parameter.numel()
            if parameter.grad is not None:
                flat[offset:offset + count].copy_(parameter.grad.reshape(-1))
            offset += count
        dist.all_reduce(flat)
        flat[:size].div_(samples)
        offset = 0
        for parameter, used in zip(self.parameters, flat[size:].tolist(), strict=True):
            count = parameter.numel()
            parameter.grad = flat[offset:offset + count].view_as(parameter).clone() if used else None
            offset += count

    def train_batch(self, documents, boundary_generator):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {}
        if self.experiment.stage == "post":
            method = asdict(self.experiment.method)
            for document in documents:
                loss, metrics = post_backward(
                    self.model, document, method, boundary_generator,
                    bptt_steps=self.config.backward.bptt_steps,
                )
                _accumulate(totals, loss, metrics)
        else:
            for start in range(0, len(documents), self.config.loop.micro_batch_size):
                with self.model.forward_context():
                    losses = []
                    for document in documents[start:start + self.config.loop.micro_batch_size]:
                        loss, metrics = document_loss(self.model, document, self.experiment)
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError(f"non-finite training loss: {document.id}")
                        losses.append(loss)
                        _accumulate(totals, loss, metrics)
                    batch_loss = torch.stack(losses).sum()
                batch_loss.backward()
        metrics = _aggregate_metrics(totals, len(documents))
        self._synchronize_gradients(metrics["samples"])
        for group in self.optimizer.param_groups:
            gradients = [parameter.grad.detach().norm() for parameter in group["params"]
                         if parameter.grad is not None]
            metrics[f"{group['name']}_grad_norm"] = (
                float(torch.stack(gradients).norm()) if gradients else 0.0
            )
            metrics[f"{group['name']}_learning_rate"] = group["lr"]
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.config.optimizer.gradient_clip, error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.scheduler.step()
        return {**metrics, "grad_norm": float(norm)}


@torch.no_grad()
def evaluate_dev(model, documents, config):
    model.eval()
    totals = {}
    for document in documents:
        with model.forward_context():
            loss, metrics = document_loss(model, document, config)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite dev loss: {document.id}")
        _accumulate(totals, loss, metrics)
    return _aggregate_metrics(totals, len(documents))
