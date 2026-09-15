"""Memory trajectories and full/truncated backpropagation."""

from dataclasses import replace
import random

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from latent_working_memory.v1.backbone import ReadTokens, load_backbone
from latent_working_memory.v1.engine import MemoryEngine, EngineRegistry, initialize_device
from tensordict import TensorDict
from verl.utils.tensordict_utils import assign_non_tensor, get_non_tensor_data
from latent_working_memory.v1.dynamic.data import write_boundaries
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.v1.dynamic.squad import sample_reads
from latent_working_memory.v1.training import load_trainable_model_state, precision_context


def read_schedule(episode, tokenizer, recipe, model_config, seed, capacity, generation=False):
    if capacity > model_config.k_limit:
        raise ValueError("capacity exceeds model slot limit")
    previous = 0
    rng, visits, schedule = random.Random(seed), {}, {}
    boundaries = write_boundaries(episode, capacity)
    for end in boundaries:
        if end - previous + 1 > model_config.write_context_tokens:
            raise ValueError("initial compression or paragraph exceeds write context budget")
        previous = end
        jobs = []
        if end == boundaries[0]:
            schedule[end] = jobs
            continue
        for read in sample_reads(
            episode, end, recipe.new_count, recipe.history_count, rng, visits, recipe.max_visits
        ):
            tokens = ReadTokens(
                tuple(tokenizer.encode(read.prompt, add_special_tokens=False)),
                tuple(tokenizer.encode(read.references[0].text, add_special_tokens=False))
                + (tokenizer.eos_token_id,),
            )
            target_budget = (
                max(len(tokens.target_ids), recipe.generation_tokens)
                if generation
                else len(tokens.target_ids)
            )
            if (
                1 + capacity + len(tokens.prompt_ids) + target_budget
                > model_config.read_context_tokens
            ):
                raise ValueError(f"QA read exceeds context budget: {read.read_id}")
            jobs.append((read, tokens))
        schedule[end] = jobs
    if not any(schedule.values()):
        raise ValueError("episode has no selected reads")
    return schedule


def bptt_segments(schedule, recipe):
    """Resolve the existing truncation boundaries without constructing a graph."""
    segments, current = [], []
    start = 0
    pending = False
    last = next(reversed(schedule))
    for end, reads in schedule.items():
        current.append(end)
        pending = pending or bool(reads)
        span = end - start if recipe.bptt_unit == "tokens" else len(current)
        boundary = recipe.bptt_span and span >= recipe.bptt_span
        if not segments and not pending:
            boundary = False
        if boundary or end == last:
            segments.append(tuple(current))
            current, start, pending = [], end, False
    return segments


class DynamicModel(torch.nn.Module):
    def __init__(self, backbone, writer, model_config, recipe):
        super().__init__()
        self.backbone, self.writer = backbone, writer
        self.model_config, self.recipe = model_config, recipe

    def trainable_parameters(self):
        yield from self.backbone.trainable_parameters()
        yield from self.writer.parameters()

    def _read_loss(self, memory, tokens):
        return self.backbone.read_batch([memory], [tokens])[0].mean_nll

    def forward(self, episode, schedule, ends, state, previous, capacity, read_count, batch_size):
        pending, statistics = [], []
        for end in ends:
            # Preserve the original per-write autocast boundary: caching a cast
            # trainable weight across writes changes BF16 gradient accumulation.
            with precision_context(self.backbone.input_projection.weight.device):
                features = self.backbone.text_features(
                    [episode.input_ids[previous:end]], [previous]
                )[0]
                if state is None:
                    state = self.writer(
                        self.writer.initialize_state(features.dtype), features, first_slots=capacity
                    )
                else:
                    state = self.writer(state, features)
                for _, tokens in schedule[end]:
                    mean_nll = (
                        activation_checkpoint(
                            self._read_loss, state.values, tokens, use_reentrant=False
                        )
                        if self.recipe.qa_activation_checkpointing
                        else self._read_loss(state.values, tokens)
                    )
                    pending.append(mean_nll / read_count / batch_size)
                    statistics.append((float(mean_nll.detach()), len(tokens.target_ids)))
            previous = end
        return {
            "loss": torch.stack(pending).sum() if pending else None,
            "state": state,
            "reads": statistics,
        }


@EngineRegistry.register(model_type="lwm_dynamic", backend="replicated", device=["cpu", "cuda"])
class DynamicEngine(MemoryEngine):
    def __init__(self, model, device):
        recipe = model.recipe
        super().__init__(
            model, device, recipe.learning_rate, recipe.weight_decay, recipe.gradient_clip,
            recipe.optimizer_fused
        )
        self.recipe = recipe
        self.model_config = model.model_config

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        episodes = get_non_tensor_data(data, "episodes", None)
        schedules = get_non_tensor_data(data, "schedules", None)
        capacity = get_non_tensor_data(data, "capacity", None)
        batch_size = len(episodes)
        local = list(range(self.rank, batch_size, self.world_size))
        articles = []
        for i in local:
            episode, schedule = episodes[i], schedules[i]
            segments = bptt_segments(schedule, self.recipe)
            count = sum(len(jobs) for jobs in schedule.values())
            last_loss = max(
                j for j, ends in enumerate(segments) if any(schedule[end] for end in ends)
            )
            state, previous = None, 0
            loss_value = token_nll = 0.0
            target_tokens = 0
            segment_statistics = []
            for j, ends in enumerate(segments):
                # Exactly one synchronized backward per rank/global batch. Later
                # no-read writes still execute, but never start another collective.
                final = i == local[-1] and j == last_loss
                with self.gradient_context(synchronize=final):
                    output = self.module(
                        episode, schedule, ends, state, previous, capacity, count, batch_size
                    )
                    if output["loss"] is not None and not forward_only:
                        output["loss"].backward()
                for nll, tokens in output["reads"]:
                    loss_value += nll / count
                    token_nll += nll * tokens
                    target_tokens += tokens
                state = output["state"].detached()
                segment_statistics.append({"tokens": ends[-1] - previous, "updates": len(ends)})
                previous = ends[-1]
                del output
            articles.append(
                {
                    "episode_id": episode.episode_id,
                    "loss": loss_value,
                    "target_nll": token_nll / target_tokens,
                    "target_tokens": target_tokens,
                    "input_tokens": len(episode.input_ids),
                    "reads": count,
                    "writes": len(schedule),
                    "updates": len(schedule) - 1,
                    "initial_tokens": next(iter(schedule)),
                    "truncations": len(segments) - 1,
                    "bptt_segments": segment_statistics,
                }
            )
        if self.world_size > 1:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, articles)
            articles = [
                gathered[i % self.world_size][i // self.world_size] for i in range(batch_size)
            ]
        totals = {
            key: sum(article[key] for article in articles)
            for key in (
                "target_tokens",
                "input_tokens",
                "reads",
                "writes",
                "updates",
                "truncations",
            )
        }
        return {
            "metrics": {
                "samples": batch_size,
                "capacity": capacity,
                "sample_metrics": articles,
                "loss": sum(a["loss"] for a in articles) / batch_size,
                "target_nll": sum(a["target_nll"] * a["target_tokens"] for a in articles)
                / totals["target_tokens"],
                **totals,
            }
        }


class DynamicTrainer:
    def __init__(self, backbone, writer, model_config, recipe, device):
        self.backbone, self.writer = backbone, writer
        self.model_config, self.recipe = model_config, recipe
        self.device = initialize_device(device)
        if recipe.reader_loss_backend == "liger" and self.device.type != "cuda":
            raise ValueError("liger reader loss requires CUDA")
        self.backbone.reader_loss_backend = recipe.reader_loss_backend
        self.engine = EngineRegistry.new(
            model_type="lwm_dynamic",
            backend="replicated",
            model=DynamicModel(backbone, writer, model_config, recipe),
            device=self.device,
        )
        self.engine.initialize()
        self.parameters, self.optimizer = self.engine.parameters, self.engine.optimizer

    def step(self, episodes, tokenizer, seeds, capacity):
        batch_size = len(episodes)
        if len(seeds) != batch_size or batch_size != self.recipe.global_batch_size:
            raise ValueError("one optimizer step requires global_batch_size samples and seeds")
        if batch_size < self.engine.world_size or batch_size % self.engine.world_size:
            raise ValueError("global batch must divide evenly across training ranks")
        if capacity not in self.recipe.capacities:
            raise ValueError("capacity is not configured for dynamic training")
        schedules = {
            i: read_schedule(
                episodes[i], tokenizer, self.recipe, self.model_config, seeds[i], capacity
            )
            for i in range(self.engine.rank, batch_size, self.engine.world_size)
        }
        data = TensorDict({}, batch_size=[])
        assign_non_tensor(data, episodes=tuple(episodes), schedules=schedules, capacity=capacity)
        with self.engine.train_mode():
            metrics = self.engine.train_batch(data, loss_function=None)["metrics"]
        metrics["gradient_norm"] = metrics.pop("grad_norm")
        return metrics


def load_components(checkpoint, device, reader_loss_backend):
    config = replace(checkpoint.config, reader_loss_backend=reader_loss_backend)
    tokenizer, backbone = load_backbone(
        config, device, torch.bfloat16 if device.type == "cuda" else torch.float32
    )
    writer = JointMemoryWriter(
        config.d_mem, config.num_layers, config.num_heads, config.ffn_dim, config.k_limit
    ).to(device)
    value = GrowthValueNetwork(config.d_mem).to(device)
    load_trainable_model_state(checkpoint.model_state, backbone, writer, value)
    value.requires_grad_(False)
    if (
        max(config.write_context_tokens, config.read_context_tokens)
        > backbone.max_position_embeddings
    ):
        raise ValueError("configured windows exceed backbone context")
    return tokenizer, backbone, writer, value
