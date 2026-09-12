"""Memory trajectories and full/truncated backpropagation."""

import random

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from latent_working_memory.v1.backbone import ReadTokens, load_backbone
from latent_working_memory.v1.distributed import synchronize_gradients
from latent_working_memory.v1.dynamic_data import write_boundaries
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.v1.squad import sample_reads
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


class DynamicTrainer:
    def __init__(self, backbone, writer, model_config, recipe, device):
        self.backbone, self.writer = backbone, writer
        self.model_config, self.recipe, self.device = model_config, recipe, device
        self.parameters = list(backbone.trainable_parameters()) + list(writer.parameters())
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=recipe.learning_rate, weight_decay=recipe.weight_decay
        )

    def step(self, episodes, tokenizer, seeds, capacity):
        batch_size = len(episodes)
        if len(seeds) != batch_size or batch_size != self.recipe.global_batch_size:
            raise ValueError("one optimizer step requires global_batch_size samples and seeds")
        if capacity not in self.recipe.capacities:
            raise ValueError("capacity is not configured for dynamic training")
        self.backbone.train()
        self.writer.train()
        self.optimizer.zero_grad(set_to_none=True)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        articles = [
            self._backward_episode(episodes[i], tokenizer, seeds[i], batch_size, capacity)
            for i in range(rank, batch_size, world_size)
        ]
        if world_size > 1:
            synchronize_gradients(self.parameters)
            gathered = [None] * world_size
            dist.all_gather_object(gathered, articles)
            articles = [gathered[i % world_size][i // world_size] for i in range(batch_size)]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.recipe.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
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
            "samples": batch_size,
            "capacity": capacity,
            "sample_metrics": articles,
            "loss": sum(a["loss"] for a in articles) / batch_size,
            "target_nll": sum(a["target_nll"] * a["target_tokens"] for a in articles)
            / totals["target_tokens"],
            **totals,
            "gradient_norm": float(grad_norm),
        }

    def _read_loss(self, memory, tokens):
        return self.backbone.read_batch([memory], [tokens])[0].mean_nll

    def _backward_episode(self, episode, tokenizer, seed, batch_size, capacity):
        schedule = read_schedule(episode, tokenizer, self.recipe, self.model_config, seed, capacity)
        count = sum(len(jobs) for jobs in schedule.values())
        state = None
        previous = segment_start = 0
        pending, loss_value, token_nll, target_tokens = [], 0.0, 0.0, 0
        truncations = segment_updates = 0
        segments = []
        for end in schedule:
            with precision_context(self.device):
                features = self.backbone.text_features(
                    [episode.input_ids[previous:end]], [previous]
                )[0]
                if state is None:
                    state = self.writer(
                        self.writer.initialize_state(features.dtype),
                        features,
                        first_slots=capacity,
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
                    pending.append(mean_nll / count / batch_size)
                    loss_value += float(mean_nll.detach()) / count
                    token_nll += float(mean_nll.detach()) * len(tokens.target_ids)
                    target_tokens += len(tokens.target_ids)
            segment_updates += 1
            span = end - segment_start if self.recipe.bptt_unit == "tokens" else segment_updates
            boundary = self.recipe.bptt_span and span >= self.recipe.bptt_span
            if not segments and not pending:
                boundary = False
            if boundary or end == episode.write_ends[-1]:
                if pending:
                    torch.stack(pending).sum().backward()
                    pending.clear()
                state = state.detached()
                segments.append({"tokens": end - segment_start, "updates": segment_updates})
                segment_updates = 0
                segment_start = end
                truncations += int(end != episode.write_ends[-1])
            previous = end
        return {
            "episode_id": episode.episode_id,
            "loss": loss_value,
            "target_nll": token_nll / target_tokens,
            "target_tokens": target_tokens,
            "input_tokens": len(episode.input_ids),
            "reads": count,
            "writes": len(schedule),
            "updates": len(schedule) - 1,
            "initial_tokens": next(iter(schedule)),
            "truncations": truncations,
            "bptt_segments": segments,
        }


def load_components(checkpoint, device):
    config = checkpoint.config
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
