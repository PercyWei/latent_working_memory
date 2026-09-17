"""verl replicated engine，沿用 codex/verl-v1 的 BaseEngine＋DDP 执行方式。"""

from contextlib import contextmanager, nullcontext
from datetime import timedelta
import os
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensordict import TensorDict
from verl.utils.distributed import initialize_global_process_group
from verl.utils.tensordict_utils import assign_non_tensor, get_non_tensor_data
from verl.workers.config import FSDPOptimizerConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.engine import BaseEngine, EngineRegistry

from latent_working_memory.devices import validate_device


def initialize_device(device):
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("reconstruction training supports CPU and CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        if device.type == "cuda":
            initialize_global_process_group(timeout_second=7200)
        else:
            dist.init_process_group("gloo", timeout=timedelta(hours=2))
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", device.index or 0)))
        torch.cuda.set_device(device)
    validate_device(device)
    return device


def precision_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@EngineRegistry.register(
    model_type="lwm_v2_reconstruction", backend="replicated", device=["cpu", "cuda"]
)
class ReconstructionEngine(BaseEngine):
    def __init__(self, model, device):
        self.model, self.device = model, torch.device(device)
        config = model.config
        self.optimizer_config = FSDPOptimizerConfig(
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            clip_grad=config.gradient_clip,
        )
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.micro_batch_size = config.micro_batch_size
        self.micro_batch_encoder_tokens = config.micro_batch_encoder_tokens
        self.micro_batch_decoder_tokens = config.micro_batch_decoder_tokens
        self.mode = None

    def initialize(self):
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
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
        self.mode = "train"
        self.module.train()
        try:
            yield
        finally:
            self.mode = None

    @contextmanager
    def eval_mode(self, **kwargs):
        was_training = self.model.training
        self.mode = "eval"
        self.module.eval()
        try:
            yield
        finally:
            self.module.train(was_training)
            self.mode = None

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self):
        # Backward is normalized by true global samples; undo DDP's averaging.
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(self.world_size)
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.optimizer_config.clip_grad, error_if_nonfinite=True
        )
        self.optimizer.step()
        return float(norm)

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        rows = get_non_tensor_data(data, "trajectories", None)
        read_tasks = get_non_tensor_data(data, "read_tasks", None)
        local = list(range(self.rank, len(rows), self.world_size))
        # A short final batch still participates on every rank. Dummy work has zero weight.
        work = local or [0]
        pending, microbatches, encoder_lengths, decoder_lengths = {}, [], {}, {}
        for i in work:
            encoder_lengths[i] = tuple(
                end - start + (rows[i].capacity if step else 0)
                for step, (start, end) in enumerate(
                    zip((0,) + rows[i].write_ends[:-1], rows[i].write_ends, strict=True)
                )
            )
            decoder_lengths[i] = tuple(
                rows[i].capacity
                + (
                    len(self.model.ae_prompt) + end
                    if read_tasks[i] == "ae"
                    else len(self.model.lm_prompt) + len(rows[i].token_ids) - rows[i].write_ends[-1]
                )
                for end in rows[i].write_ends
            )
        # Length ordering reduces padding without requiring matching tasks or depths.
        work.sort(key=lambda i: (max(decoder_lengths[i]), max(encoder_lengths[i])), reverse=True)
        for i in work:
            key = rows[i].capacity
            group = pending.setdefault(key, [])
            candidate = [*group, i]
            fits = True
            for step in range(max(len(rows[j].write_ends) for j in candidate)):
                active = [j for j in candidate if step < len(rows[j].write_ends)]
                enc = [encoder_lengths[j][step] for j in active]
                dec = [decoder_lengths[j][step] for j in active]
                encoder_cost = (
                    sum(enc) if self.model.codec.config.padding_free else len(active) * max(enc)
                )
                decoder_cost = (
                    sum(dec) if self.model.codec.config.padding_free else len(active) * max(dec)
                )
                if (
                    encoder_cost > self.micro_batch_encoder_tokens
                    or decoder_cost > self.micro_batch_decoder_tokens
                ):
                    fits = False
                    break
            if group and not fits:
                microbatches.append(group)
                group = pending[key] = []
            group.append(i)
            if len(group) == self.micro_batch_size:
                microbatches.append(pending.pop(key))
        microbatches.extend(pending.values())
        microbatches.sort(key=min)
        records = []
        for position, group in enumerate(microbatches):
            final = position == len(microbatches) - 1
            sync = nullcontext() if final or self.world_size == 1 else self.module.no_sync()
            with sync, precision_context(self.device):
                single = len(group) == 1
                output = self.module(
                    rows[group[0]] if single else [rows[i] for i in group],
                    read_task=read_tasks[group[0]] if single else [read_tasks[i] for i in group],
                )
                loss = output["loss"] * (len(group) if local else 0) / len(rows)
                if not forward_only:
                    loss.backward()
            if local:
                values = (
                    [float(output["loss"].detach())]
                    if single
                    else output["sample_losses"].detach().cpu().tolist()
                )
                rounds = [output["rounds"]] if single else output["rounds"]
                records.extend(
                    {
                        "index": i,
                        "loss": value,
                        "rounds": reads,
                        "microbatch_size": len(group),
                        "microbatch_first": position == 0,
                        "batch_sizes": output["batch_sizes"] if position == 0 else [],
                    }
                    for position, (i, value, reads) in enumerate(
                        zip(group, values, rounds, strict=True)
                    )
                )
            del output, loss
        if self.world_size > 1:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, records)
            records = [row for rank_rows in gathered for row in rank_rows]
        records.sort(key=lambda row: row["index"])
        task_metrics = {}
        for name in ("ae", "lm"):
            selected = [r for r in records if read_tasks[r["index"]] == name]
            task_metrics[name] = (
                sum(sum(t[name] for t in r["rounds"]) / len(r["rounds"]) for r in selected)
                / len(selected)
                if selected
                else None
            )
            task_metrics[f"{name}_samples"] = len(selected)
            task_metrics[f"{name}_tokens"] = sum(
                t[f"{name}_tokens"] for r in selected for t in r["rounds"]
            )
        return {
            "metrics": {
                "loss": sum(r["loss"] for r in records) / len(rows),
                **task_metrics,
                "samples": len(rows),
                "microbatches": sum(r["microbatch_first"] for r in records),
                "max_microbatch_size": max(r["microbatch_size"] for r in records),
                "mean_active_microbatch_size": sum(sum(r["batch_sizes"]) for r in records)
                / sum(len(r["batch_sizes"]) for r in records),
                "batched_samples": sum(r["microbatch_size"] > 1 for r in records),
                "source_tokens": sum(r.write_ends[-1] for r in rows),
                "target_tokens": sum(
                    t["ae_tokens"] + t["lm_tokens"] for r in records for t in r["rounds"]
                ),
            }
        }

    def step(self, trajectories, step):
        if not trajectories:
            raise ValueError("a training batch cannot be empty")
        data = TensorDict({}, batch_size=[])
        # Sample before rank sharding or microbatch grouping. The checkpoint's next
        # optimizer-step cursor reproduces assignments without consuming dropout RNG.
        rng = random.Random(f"{self.model.config.seed}:read-task:{step}")
        read_tasks = tuple(
            "lm" if rng.random() < self.model.config.lm_ratio else "ae" for _ in trajectories
        )
        assign_non_tensor(data, trajectories=tuple(trajectories), read_tasks=read_tasks)
        with self.train_mode():
            return self.train_batch(data, loss_function=None)["metrics"]
