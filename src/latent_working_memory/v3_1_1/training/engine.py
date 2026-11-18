"""按完整文档累积梯度，跨进程归约后统一更新压缩器和边界模块。"""

from dataclasses import asdict
import math

import torch
import torch.distributed as dist

from .rl import post_backward_batch
from .dev import evaluate_dev_conditions
from ..compression import compress_document, compress_documents_batch
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


def document_loss(model, document, config, temperature=1.0):
    """预训练目标或后训练的确定性 dev 评分；不执行反向传播。"""
    if not config.uses_boundary_policy:
        result = compress_document(
            model, document.input_ids, config.training.compression,
            recent_size=config.recent_size(),
        )
        if config.training.objective.type == "qa":
            loss = qa_nll(model, result.blocks, document, result.recent_ids)
            return loss, {"qa_nll": (float(loss.detach()), 1)}
        loss = model.answer_nll(
            result.blocks, (*result.recent_ids, *document.prompt_ids), document.target_ids,
        )
        return loss, {
            f"{document.task}_nll": (float(loss.detach()), 1),
            "lm_fraction": (float(document.task == "lm"), 1),
        }
    result = execute(
        model, document.input_ids, config.model.structure.pending_stride,
        temperature=temperature,
        pending_size=config.model.structure.pending_size,
        recent_size=config.recent_size(),
    )
    loss = qa_nll(model, result.blocks, document, result.retained_ids)
    decisions = sum(result.decision_mask)
    slots = len(result.blocks) * model.memory_slots
    prefix_length = len(document.input_ids) - len(result.recent_ids)
    prefix_retention = ((slots + len(result.pending_ids)) / prefix_length
                        if prefix_length else 0.0)
    objective = config.training.objective
    cost = prefix_retention ** objective.memory_cost_power
    return loss, {
        "qa_nll": (float(loss), 1), "blocks": (len(result.blocks), 1),
        "memory_slots": (slots, 1),
        "recent_tokens": (len(result.recent_ids), 1),
        "pending_tokens": (len(result.pending_ids), 1),
        "prefix_retention_ratio": (prefix_retention, 1),
        "memory_cost": (cost, 1),
        "objective": (float(loss) + objective.memory_cost_weight * cost, 1),
        "retention_ratio": ((slots + len(result.retained_ids))
                            / len(document.input_ids), 1),
        "segment_tokens": (sum(result.block_lengths), len(result.block_lengths)),
        "commit_rate": (sum(result.actions), decisions),
        "policy_entropy": (float(result.entropy.sum()), decisions),
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
        boundary_only = config.uses_boundary_policy and self.config.update_mode == "boundary_only"
        if boundary_only:
            model.writer.requires_grad_(False)
            model.memory_embeddings.requires_grad_(False)
        if config.uses_boundary_policy:
            model.boundary.requires_grad_(True)
        groups = [] if boundary_only else [{
            "name": "compressor", "params": list(model.compressor_parameters()),
            "lr": self.config.optimizer.compressor_learning_rate,
        }]
        if config.uses_boundary_policy:
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

    def train_batch(self, documents, boundary_generator, step):
        if self.experiment.uses_boundary_policy and self.config.update_mode == "boundary_only":
            self.model.eval()
            self.model.boundary.train()
        else:
            self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {}
        transition = None
        if self.experiment.uses_boundary_policy:
            pending_stride = self.experiment.model.structure.pending_stride
            objective = asdict(self.config.objective)
            transition = self.config.objective.policy_transition
            temperature = (1.0 if transition is None
                           else transition.temperature(step, self.config.loop.max_steps))
            for start in range(0, len(documents), self.config.loop.micro_batch_size):
                results = post_backward_batch(
                    self.model, documents[start:start + self.config.loop.micro_batch_size],
                    pending_stride, objective, boundary_generator,
                    bptt_steps=self.config.backward.bptt_steps, temperature=temperature,
                    pending_size=self.experiment.model.structure.pending_size,
                    recent_size=self.experiment.recent_size(training=True),
                    update_mode=self.config.update_mode,
                )
                for loss, metrics in results:
                    _accumulate(totals, loss, metrics)
        elif (self.config.objective.type == "qa"
              or self.config.compression.mode != "single"):
            documents = sorted(documents, key=lambda document: len(document.input_ids), reverse=True)
            for start in range(0, len(documents), self.config.loop.micro_batch_size):
                self._compressor_batch(
                    documents[start:start + self.config.loop.micro_batch_size], totals,
                )
        else:
            documents = sorted(documents, key=lambda document: len(document.input_ids), reverse=True)
            for start in range(0, len(documents), self.config.loop.micro_batch_size):
                microbatch = documents[start:start + self.config.loop.micro_batch_size]
                # AE 与 LM 的目标长度差异较大，组内批量前向可减少 padding。
                for task in ("ae", "lm"):
                    batch = [document for document in microbatch if document.task == task]
                    if not batch:
                        continue
                    with self.model.forward_context():
                        memories = self.model.compress_batch([document.input_ids for document in batch])
                        losses = self.model.answer_nll_batch(
                            [(memory,) for memory in memories],
                            [document.prompt_ids for document in batch],
                            [document.target_ids for document in batch],
                        )
                        if not bool(torch.isfinite(losses).all()):
                            raise RuntimeError(f"non-finite {task} training loss")
                        for loss in losses:
                            _accumulate(totals, loss, {
                                f"{task}_nll": (float(loss.detach()), 1),
                                "lm_fraction": (float(task == "lm"), 1),
                            })
                        batch_loss = losses.sum()
                    batch_loss.backward()
        metrics = _aggregate_metrics(totals, len(documents))
        if transition is not None:
            metrics["policy_temperature"] = temperature
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

    def _compressor_batch(self, documents, totals):
        """每篇采样一次压缩路径，逐题反传 Reader 后统一回传记忆梯度。"""
        with self.model.forward_context():
            results = compress_documents_batch(
                self.model, [document.input_ids for document in documents], self.config.compression,
                recent_size=self.experiment.recent_size(training=True),
            )
        leaves = [tuple(block.detach().requires_grad_(True) for block in result.blocks)
                  for result in results]
        qa_objective = self.config.objective.type == "qa"
        questions = [document.qas if qa_objective else (document,) for document in documents]
        document_losses = [0.0] * len(documents)
        for question_index in range(max(map(len, questions))):
            active = [index for index, rows in enumerate(questions) if question_index < len(rows)]
            selected = [questions[index][question_index] for index in active]
            with self.model.forward_context():
                losses = self.model.answer_nll_batch(
                    [leaves[index] for index in active],
                    [(*results[index].recent_ids, *question.prompt_ids)
                     for index, question in zip(active, selected, strict=True)],
                    [question.target_ids for question in selected],
                )
                if not bool(torch.isfinite(losses).all()):
                    raise RuntimeError("non-finite compressor training loss")
                normalized = losses / losses.new_tensor([len(questions[index]) for index in active])
            normalized.sum().backward()
            for index, value in zip(active, normalized.detach().tolist(), strict=True):
                document_losses[index] += value
        outputs, gradients = [], []
        for result, memories in zip(results, leaves, strict=True):
            for block, memory in zip(result.blocks, memories, strict=True):
                outputs.append(block)
                gradients.append(memory.grad)
        if outputs:
            torch.autograd.backward(outputs, gradients)
        for document, result, loss in zip(documents, results, document_losses, strict=True):
            task = "qa" if qa_objective else document.task
            slots = len(result.blocks) * self.model.memory_slots
            prefix_length = len(document.input_ids) - len(result.recent_ids)
            metrics = {
                f"{task}_nll": (loss, 1), "blocks": (len(result.blocks), 1),
                "recent_tokens": (len(result.recent_ids), 1),
                "prefix_retention_ratio": (slots / prefix_length, 1),
                "retention_ratio": ((slots + len(result.recent_ids)) / len(document.input_ids), 1),
                "segment_tokens": (sum(result.block_lengths), len(result.block_lengths)),
            }
            if not qa_objective:
                metrics["lm_fraction"] = (float(task == "lm"), 1)
            _accumulate(totals, self.model.memory_embeddings.new_tensor(loss), metrics)


def uses_condition_dev(config):
    if config.training.dev.fixed_ratios:
        return True
    if config.uses_boundary_policy:
        return config.recent_size() > 0
    return (config.training.compression.mode == "multi"
            or config.training.objective.type == "qa")


@torch.no_grad()
def evaluate_dev(model, documents, config, temperature=1.0, baseline_cache=None, records=None):
    model.eval()
    if uses_condition_dev(config):
        totals, details = evaluate_dev_conditions(model, documents, config, baseline_cache)
        if records is not None:
            records.extend(details)
        return _aggregate_metrics(totals, len(documents))
    totals = {}
    for document in documents:
        with model.forward_context():
            loss, metrics = document_loss(model, document, config, temperature)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite dev loss: {document.id}")
        _accumulate(totals, loss, metrics)
    return _aggregate_metrics(totals, len(documents))
