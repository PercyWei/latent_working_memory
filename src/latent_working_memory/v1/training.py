from __future__ import annotations

import os
import random
import tempfile
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import peft
import torch
from torch import Tensor, nn
import transformers
from transformers import PreTrainedTokenizerBase

from latent_working_memory.v1.backbone import (
    LatentMemoryBackbone,
    QuestionAnswerTokens,
    load_backbone,
    tokenize_question_answer,
)
from latent_working_memory.v1.checkpoint import (
    LoadedModelCheckpoint,
    capture_rng_state,
    load_model_checkpoint,
    restore_rng_state,
    save_model_checkpoint,
)
from latent_working_memory.v1.config import (
    FRAMEWORK_VERSION,
    ExperimentConfig,
    load_config,
    write_resolved_config,
)
from latent_working_memory.v1.data import (
    Episode,
    Probe,
    build_encoder_cells,
    read_episodes,
)
from latent_working_memory.v1.evaluation import exact_match
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.v1.objectives import (
    ReaderOutput,
    build_reader_output,
    teacher_student_kl,
)


TEACHER_CACHE_FIELDS = frozenset({"framework_version", "identity", "entries"})
TEACHER_CACHE_IDENTITY_FIELDS = frozenset(
    {
        "model_name_or_path",
        "model_revision",
        "model_dtype",
        "tokenizer_name_or_path",
        "tokenizer_revision",
    }
)
TEACHER_CACHE_ENTRY_FIELDS = frozenset({"input_ids", "target_length", "target_logits"})
TRAINABLE_MODEL_STATE_FIELDS = frozenset({"backbone", "writer", "value_network"})


@dataclass(frozen=True, slots=True)
class TeacherCacheIdentity:
    model_name_or_path: str
    model_revision: str | None
    model_dtype: str
    tokenizer_name_or_path: str
    tokenizer_revision: str | None

    def __post_init__(self) -> None:
        for name in ("model_name_or_path", "model_dtype", "tokenizer_name_or_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("model_revision", "tokenizer_revision"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be null or a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "model_revision": self.model_revision,
            "model_dtype": self.model_dtype,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "tokenizer_revision": self.tokenizer_revision,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> TeacherCacheIdentity:
        _require_exact_fields(raw, TEACHER_CACHE_IDENTITY_FIELDS, "teacher cache identity")
        return cls(**raw)


class TeacherLogitCache:
    def __init__(self, identity: TeacherCacheIdentity) -> None:
        self.identity = identity
        self._entries: dict[tuple[tuple[int, ...], int], Tensor] = {}
        self.hits = 0
        self.misses = 0

    def get_or_compute(
        self,
        backbone: LatentMemoryBackbone,
        prefix_ids: tuple[int, ...],
        tokens: QuestionAnswerTokens,
    ) -> ReaderOutput:
        serialized = backbone.teacher_input_ids(prefix_ids, tokens)
        key = (serialized, len(tokens.target_ids))
        cached = self._entries.get(key)
        if cached is None:
            output = backbone.teacher_output(prefix_ids, tokens)
            cached = output.target_logits.detach().to(device="cpu")
            self._entries[key] = cached
            self.misses += 1
        else:
            self.hits += 1

        device = backbone.memory_projection.weight.device
        logits = cached.to(device=device)
        target_ids = torch.tensor(tokens.target_ids, device=device, dtype=torch.long)
        return build_reader_output(logits, target_ids)

    def save(self, path: str | Path) -> None:
        entries = []
        for (input_ids, target_length), target_logits in sorted(
            self._entries.items(), key=lambda item: (len(item[0][0]), item[0][0], item[0][1])
        ):
            entries.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "target_length": target_length,
                    "target_logits": target_logits,
                }
            )
        payload = {
            "framework_version": FRAMEWORK_VERSION,
            "identity": self.identity.to_dict(),
            "entries": entries,
        }
        _atomic_torch_save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_identity: TeacherCacheIdentity,
    ) -> TeacherLogitCache:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError("teacher cache root must be a mapping")
        _require_exact_fields(payload, TEACHER_CACHE_FIELDS, "teacher cache")
        if payload["framework_version"] != FRAMEWORK_VERSION:
            raise ValueError(f"framework_version must be {FRAMEWORK_VERSION!r}")
        identity_raw = payload["identity"]
        if not isinstance(identity_raw, Mapping):
            raise TypeError("teacher cache identity must be a mapping")
        identity = TeacherCacheIdentity.from_mapping(identity_raw)
        if identity != expected_identity:
            raise ValueError("teacher cache identity does not match this run")
        entries_raw = payload["entries"]
        if not isinstance(entries_raw, list):
            raise TypeError("teacher cache entries must be a list")

        cache = cls(identity)
        for raw in entries_raw:
            if not isinstance(raw, Mapping):
                raise TypeError("teacher cache entry must be a mapping")
            _require_exact_fields(raw, TEACHER_CACHE_ENTRY_FIELDS, "teacher cache entry")
            input_ids = raw["input_ids"]
            target_length = raw["target_length"]
            target_logits = raw["target_logits"]
            if not isinstance(input_ids, Tensor) or input_ids.ndim != 1 or input_ids.numel() == 0:
                raise TypeError("teacher cache input_ids must be a rank-1 tensor")
            if input_ids.dtype != torch.long:
                raise TypeError("teacher cache input_ids must use torch.long")
            if type(target_length) is not int or target_length <= 0:
                raise ValueError("teacher cache target_length must be a positive integer")
            if (
                not isinstance(target_logits, Tensor)
                or target_logits.ndim != 2
                or target_logits.shape[0] != target_length
                or target_logits.shape[1] == 0
            ):
                raise ValueError(
                    "teacher cache target_logits must have shape [target_length, vocab_size]"
                )
            if not target_logits.is_floating_point():
                raise TypeError("teacher cache target_logits must use a floating-point dtype")
            key = (tuple(int(token_id) for token_id in input_ids.tolist()), target_length)
            if key in cache._entries:
                raise ValueError("teacher cache contains a duplicate serialized input")
            cache._entries[key] = target_logits
        return cache

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class P0Example:
    episode: Episode
    prefix_end: int
    capacity: int
    probes: tuple[Probe, ...]

    def __post_init__(self) -> None:
        if self.prefix_end > len(self.episode.input_ids):
            raise ValueError("prefix_end exceeds the episode length")
        if type(self.capacity) is not int or self.capacity < 16 or (self.capacity - 16) % 8 != 0:
            raise ValueError("P0 capacity must be 16 plus a non-negative multiple of 8")
        if not self.probes:
            raise ValueError("a P0 example must contain probes")
        if any(probe.prefix_end != self.prefix_end for probe in self.probes):
            raise ValueError("all P0 probes must belong to the sampled prefix")


@dataclass(frozen=True, slots=True)
class P0ForwardOutput:
    loss: Tensor
    gold_loss: Tensor
    distill_loss: Tensor
    teacher_gold_loss: Tensor
    memory: Tensor
    probe_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class P0StepMetrics:
    loss: float
    gold_nll: float
    distill_kl: float
    teacher_gold_nll: float
    gradient_norm: float
    episode_id: str
    prefix_end: int
    capacity: int
    probe_ids: tuple[str, ...]
    teacher_cache_hits: int
    teacher_cache_misses: int

    def to_record(self, step: int) -> dict[str, Any]:
        return {
            "step": step,
            "loss": self.loss,
            "gold_nll": self.gold_nll,
            "distill_kl": self.distill_kl,
            "teacher_gold_nll": self.teacher_gold_nll,
            "gradient_norm": self.gradient_norm,
            "episode_id": self.episode_id,
            "prefix_end": self.prefix_end,
            "capacity": self.capacity,
            "probe_ids": list(self.probe_ids),
            "teacher_cache_hits": self.teacher_cache_hits,
            "teacher_cache_misses": self.teacher_cache_misses,
        }


def sample_p0_example(
    episode: Episode,
    k_limit: int,
    probes_per_prefix: int,
) -> P0Example:
    if type(k_limit) is not int or k_limit < 16:
        raise ValueError("k_limit must be an integer of at least 16")
    if type(probes_per_prefix) is not int or probes_per_prefix <= 0:
        raise ValueError("probes_per_prefix must be a positive integer")
    by_prefix: dict[int, list[Probe]] = {}
    for probe in episode.probes:
        by_prefix.setdefault(probe.prefix_end, []).append(probe)
    eligible_prefixes = sorted(
        prefix_end
        for prefix_end, probes in by_prefix.items()
        if prefix_end >= 16 and len(probes) >= probes_per_prefix
    )
    if not eligible_prefixes:
        raise ValueError("episode has no P0-eligible probe prefix")

    prefix_end = random.choice(eligible_prefixes)
    maximum_capacity = min(k_limit, prefix_end)
    capacities = tuple(range(16, maximum_capacity + 1, 8))
    capacity = random.choice(capacities)
    probes = tuple(random.sample(by_prefix[prefix_end], probes_per_prefix))
    return P0Example(episode, prefix_end, capacity, probes)


def p0_forward(
    tokenizer: PreTrainedTokenizerBase,
    backbone: LatentMemoryBackbone,
    writer: JointMemoryWriter,
    teacher_cache: TeacherLogitCache,
    example: P0Example,
    config: ExperimentConfig,
) -> P0ForwardOutput:
    if example.capacity > writer.slot_limit:
        raise ValueError("sampled capacity exceeds the writer slot limit")
    prefix_ids = example.episode.input_ids[: example.prefix_end]
    cells = build_encoder_cells(prefix_ids, config.cell_tokens)
    frozen_encoding = backbone.frozen_cell_encoding(cells)
    features = backbone.project_cell_encoding(frozen_encoding)
    state = writer.initialize_state(example.capacity, dtype=features.dtype)
    state = writer(state, features, grow_by=0)

    gold_losses: list[Tensor] = []
    distill_losses: list[Tensor] = []
    teacher_gold_losses: list[Tensor] = []
    for probe in example.probes:
        tokens = tokenize_question_answer(tokenizer, probe.question, probe.answer)
        teacher = teacher_cache.get_or_compute(backbone, prefix_ids, tokens)
        student = backbone.student_output(state.values, tokens)
        gold_losses.append(student.mean_nll)
        distill_losses.append(
            teacher_student_kl(
                teacher.target_logits,
                student.target_logits,
                config.distill_temperature,
            )
        )
        teacher_gold_losses.append(teacher.mean_nll)

    gold_loss = torch.stack(gold_losses).mean()
    distill_loss = torch.stack(distill_losses).mean()
    teacher_gold_loss = torch.stack(teacher_gold_losses).mean()
    return P0ForwardOutput(
        loss=gold_loss + config.lambda_distill * distill_loss,
        gold_loss=gold_loss,
        distill_loss=distill_loss,
        teacher_gold_loss=teacher_gold_loss,
        memory=state.values,
        probe_ids=tuple(probe.probe_id for probe in example.probes),
    )


class P0Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        tokenizer: PreTrainedTokenizerBase,
        backbone: LatentMemoryBackbone,
        writer: JointMemoryWriter,
        teacher_cache: TeacherLogitCache,
        device: torch.device,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.writer = writer
        self.teacher_cache = teacher_cache
        self.device = device
        parameters = [*backbone.trainable_parameters(), *writer.parameters()]
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def step(self, example: P0Example) -> P0StepMetrics:
        self.backbone.train()
        self.writer.train()
        self.optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            output = p0_forward(
                self.tokenizer,
                self.backbone,
                self.writer,
                self.teacher_cache,
                example,
                self.config,
            )
        output.loss.backward()
        parameters = [
            parameter
            for parameter in (*self.backbone.trainable_parameters(), *self.writer.parameters())
            if parameter.grad is not None
        ]
        gradient_norm = nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
        self.optimizer.step()
        return P0StepMetrics(
            loss=float(output.loss.detach().float().item()),
            gold_nll=float(output.gold_loss.detach().float().item()),
            distill_kl=float(output.distill_loss.detach().float().item()),
            teacher_gold_nll=float(output.teacher_gold_loss.detach().float().item()),
            gradient_norm=float(gradient_norm.detach().float().item()),
            episode_id=example.episode.episode_id,
            prefix_end=example.prefix_end,
            capacity=example.capacity,
            probe_ids=output.probe_ids,
            teacher_cache_hits=self.teacher_cache.hits,
            teacher_cache_misses=self.teacher_cache.misses,
        )


def trainable_model_state(
    backbone: LatentMemoryBackbone,
    writer: JointMemoryWriter,
    value_network: GrowthValueNetwork,
) -> dict[str, Any]:
    return {
        "backbone": backbone.trainable_state_dict(),
        "writer": writer.state_dict(),
        "value_network": value_network.state_dict(),
    }


def load_trainable_model_state(
    state: Mapping[str, Any],
    backbone: LatentMemoryBackbone,
    writer: JointMemoryWriter,
    value_network: GrowthValueNetwork,
) -> None:
    _require_exact_fields(state, TRAINABLE_MODEL_STATE_FIELDS, "trainable model state")
    backbone_state = state["backbone"]
    if not isinstance(backbone_state, Mapping):
        raise TypeError("backbone trainable state must be a mapping")
    if not isinstance(state["writer"], Mapping):
        raise TypeError("writer state must be a mapping")
    if not isinstance(state["value_network"], Mapping):
        raise TypeError("value_network state must be a mapping")
    backbone.load_trainable_state_dict(backbone_state)
    writer.load_state_dict(state["writer"])
    value_network.load_state_dict(state["value_network"])


@dataclass(frozen=True, slots=True)
class P0RunResult:
    final_checkpoint: Path
    completed_steps: int
    teacher_cache_entries: int
    dev_metrics: dict[str, Any]


def evaluate_p0_example(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
    backbone: LatentMemoryBackbone,
    writer: JointMemoryWriter,
    teacher_cache: TeacherLogitCache,
    example: P0Example,
    device: torch.device,
) -> list[dict[str, Any]]:
    was_backbone_training = backbone.training
    was_writer_training = writer.training
    backbone.eval()
    writer.eval()
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    try:
        with torch.no_grad(), context:
            prefix_ids = example.episode.input_ids[: example.prefix_end]
            cells = build_encoder_cells(prefix_ids, config.cell_tokens)
            frozen_encoding = backbone.frozen_cell_encoding(cells)
            features = backbone.project_cell_encoding(frozen_encoding)
            state = writer.initialize_state(example.capacity, dtype=features.dtype)
            state = writer(state, features, grow_by=0)
            no_memory = state.values[:0]

            records: list[dict[str, Any]] = []
            for probe in example.probes:
                tokens = tokenize_question_answer(tokenizer, probe.question, probe.answer)
                teacher = teacher_cache.get_or_compute(backbone, prefix_ids, tokens)
                student_memory = backbone.student_output(state.values, tokens)
                student_no_memory = backbone.student_output(no_memory, tokens)
                generated = {
                    "teacher": backbone.greedy_teacher(
                        prefix_ids,
                        tokens.prompt_ids,
                        config.max_new_tokens,
                    ),
                    "student_memory": backbone.greedy_student(
                        state.values,
                        tokens.prompt_ids,
                        config.max_new_tokens,
                    ),
                    "student_no_memory": backbone.greedy_student(
                        no_memory,
                        tokens.prompt_ids,
                        config.max_new_tokens,
                    ),
                }
                record: dict[str, Any] = {
                    "episode_id": example.episode.episode_id,
                    "probe_id": probe.probe_id,
                    "probe_kind": probe.kind,
                    "prefix_end": example.prefix_end,
                    "capacity": example.capacity,
                    "answer_tokens": tokens.answer_token_count,
                    "target_tokens_with_eos": len(tokens.target_ids),
                    "reference": probe.answer,
                }
                for label, output in (
                    ("teacher", teacher),
                    ("student_memory", student_memory),
                    ("student_no_memory", student_no_memory),
                ):
                    token_nll = output.token_nll.detach().float()
                    record[f"{label}_nll_sum_with_eos"] = float(token_nll.sum().item())
                    record[f"{label}_nll_sum_without_eos"] = float(token_nll[:-1].sum().item())
                    generated_ids = generated[label]
                    prediction = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                    record[f"{label}_generated_ids"] = list(generated_ids)
                    record[f"{label}_prediction"] = prediction
                    record[f"{label}_exact_match"] = exact_match(prediction, probe.answer)
                records.append(record)
    finally:
        backbone.train(was_backbone_training)
        writer.train(was_writer_training)
    return records


def aggregate_p0_dev_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("P0 dev evaluation requires at least one probe record")
    answer_tokens = sum(record["answer_tokens"] for record in records)
    target_tokens = sum(record["target_tokens_with_eos"] for record in records)
    metrics: dict[str, Any] = {
        "episodes": len({record["episode_id"] for record in records}),
        "probes": len(records),
        "answer_tokens": answer_tokens,
        "target_tokens_with_eos": target_tokens,
    }
    for label in ("teacher", "student_memory", "student_no_memory"):
        without_eos = sum(record[f"{label}_nll_sum_without_eos"] for record in records)
        with_eos = sum(record[f"{label}_nll_sum_with_eos"] for record in records)
        mean_without_eos = without_eos / answer_tokens
        mean_with_eos = with_eos / target_tokens
        metrics[label] = {
            "mean_nll_without_eos": mean_without_eos,
            "perplexity_without_eos": math.exp(mean_without_eos),
            "mean_nll_with_eos": mean_with_eos,
            "perplexity_with_eos": math.exp(mean_with_eos),
            "exact_match": sum(record[f"{label}_exact_match"] for record in records) / len(records),
        }
    metrics["memory_nll_gain_without_eos"] = (
        metrics["student_no_memory"]["mean_nll_without_eos"]
        - metrics["student_memory"]["mean_nll_without_eos"]
    )
    metrics["memory_nll_gain_with_eos"] = (
        metrics["student_no_memory"]["mean_nll_with_eos"]
        - metrics["student_memory"]["mean_nll_with_eos"]
    )
    return metrics


def run_p0_training(
    config: ExperimentConfig,
    data_dir: str | Path,
    output_dir: str | Path,
    device: torch.device,
    max_steps: int,
    episode_limit: int,
    dev_episode_limit: int,
    save_every: int,
    resume: str | Path | None = None,
) -> P0RunResult:
    segment_started = time.perf_counter()
    for name, value in (
        ("max_steps", max_steps),
        ("episode_limit", episode_limit),
        ("dev_episode_limit", dev_episode_limit),
        ("save_every", save_every),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data_path = Path(data_dir)
    train_episodes = read_episodes(data_path / "train.jsonl")
    dev_episodes = read_episodes(data_path / "dev.jsonl")
    if len(train_episodes) < episode_limit:
        raise ValueError("train split contains fewer episodes than episode_limit")
    if len(dev_episodes) < dev_episode_limit:
        raise ValueError("dev split contains fewer episodes than dev_episode_limit")
    train_episodes = train_episodes[:episode_limit]
    dev_episodes = dev_episodes[:dev_episode_limit]

    destination = Path(output_dir)
    checkpoint: LoadedModelCheckpoint | None = None
    if resume is None:
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"output directory is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        next_step = 0
    else:
        checkpoint = load_model_checkpoint(resume)
        if checkpoint.phase != "p0":
            raise ValueError("resume checkpoint is not a P0 checkpoint")
        if checkpoint.config != config:
            raise ValueError("resume checkpoint config does not match the requested config")
        _require_exact_fields(
            checkpoint.progress,
            frozenset({"next_step", "episode_limit"}),
            "P0 checkpoint progress",
        )
        if checkpoint.progress["episode_limit"] != episode_limit:
            raise ValueError("resume checkpoint used a different episode_limit")
        next_step = checkpoint.progress["next_step"]
        if type(next_step) is not int or next_step < 0:
            raise ValueError("checkpoint next_step must be a non-negative integer")
        destination.mkdir(parents=True, exist_ok=True)
        resolved_path = destination / "config.resolved.json"
        if not resolved_path.exists() or load_config(resolved_path) != config:
            raise ValueError("output directory does not contain the matching resolved config")
        if not (destination / "manifest.json").exists():
            raise ValueError("output directory does not contain the run manifest")
        if next_step > 0 and not (destination / "memory_trace.jsonl").exists():
            raise ValueError("output directory does not contain the prior memory trace")
    if next_step > max_steps:
        raise ValueError("max_steps is smaller than the checkpoint training cursor")

    random.seed(config.model_seed)
    torch.manual_seed(config.model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.model_seed)
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer, backbone = load_backbone(config, device, model_dtype)
    writer = JointMemoryWriter(
        config.d_mem,
        config.num_layers,
        config.num_heads,
        config.ffn_dim,
        config.k_limit,
    ).to(device=device)
    value_network = GrowthValueNetwork(config.d_mem).to(device=device)
    identity = TeacherCacheIdentity(
        model_name_or_path=config.teacher_model_name_or_path,
        model_revision=config.teacher_model_revision,
        model_dtype=str(model_dtype),
        tokenizer_name_or_path=tokenizer.name_or_path,
        tokenizer_revision=config.model_revision,
    )
    cache_path = destination / "teacher_cache.pt"
    teacher_cache = (
        TeacherLogitCache.load(cache_path, identity)
        if cache_path.exists()
        else TeacherLogitCache(identity)
    )
    trainer = P0Trainer(
        config,
        tokenizer,
        backbone,
        writer,
        teacher_cache,
        device,
    )

    if checkpoint is not None:
        load_trainable_model_state(
            checkpoint.model_state,
            backbone,
            writer,
            value_network,
        )
        trainer.optimizer.load_state_dict(checkpoint.optimizer_state)
        restore_rng_state(checkpoint.rng_state)
    else:
        write_resolved_config(config, destination / "config.resolved.json")
        _write_json(
            destination / "manifest.json",
            _run_manifest(config, tokenizer.name_or_path, device, model_dtype),
        )

    log_directory = destination / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    segment = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    log_path = log_directory / f"p0-from-{next_step:06d}-{segment}.jsonl"
    trace_path = destination / "memory_trace.jsonl"
    trace_mode = "a" if resume is not None else "x"
    last_metrics: P0StepMetrics | None = None
    final_checkpoint = Path(resume) if resume is not None else destination / "checkpoints/p0.pt"
    with (
        log_path.open("x", encoding="utf-8") as log_handle,
        trace_path.open(trace_mode, encoding="utf-8") as trace_handle,
    ):
        for step in range(next_step, max_steps):
            episode = random.choice(train_episodes)
            example = sample_p0_example(
                episode,
                config.k_limit,
                config.probes_per_prefix,
            )
            last_metrics = trainer.step(example)
            json.dump(last_metrics.to_record(step), log_handle, ensure_ascii=False)
            log_handle.write("\n")
            log_handle.flush()
            json.dump(
                {
                    "phase": "p0",
                    "segment": segment,
                    "step": step,
                    "episode_id": example.episode.episode_id,
                    "prefix_start": 0,
                    "prefix_end": example.prefix_end,
                    "seen_tokens": example.prefix_end,
                    "slots_before": example.capacity,
                    "grow_by": 0,
                    "slots_after": example.capacity,
                    "probe_ids": list(last_metrics.probe_ids),
                },
                trace_handle,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            trace_handle.write("\n")
            trace_handle.flush()
            completed_steps = step + 1
            if completed_steps % save_every == 0:
                final_checkpoint = _save_p0_checkpoint(
                    destination,
                    completed_steps,
                    episode_limit,
                    config,
                    backbone,
                    writer,
                    value_network,
                    trainer,
                )
                teacher_cache.save(cache_path)

    final_checkpoint = _save_p0_checkpoint(
        destination,
        max_steps,
        episode_limit,
        config,
        backbone,
        writer,
        value_network,
        trainer,
    )

    dev_records: list[dict[str, Any]] = []
    for episode in dev_episodes:
        example = _deterministic_p0_dev_example(episode, config)
        dev_records.extend(
            evaluate_p0_example(
                config,
                tokenizer,
                backbone,
                writer,
                teacher_cache,
                example,
                device,
            )
        )
    dev_metrics = aggregate_p0_dev_metrics(dev_records)
    _write_jsonl(destination / "predictions.jsonl", dev_records)
    _write_json(
        destination / "metrics.json",
        {
            "phase": "p0",
            "completed_steps": max_steps,
            "last_train_step": (
                last_metrics.to_record(max_steps - 1) if last_metrics is not None else None
            ),
            "dev": dev_metrics,
        },
    )
    teacher_cache.save(cache_path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _append_resource_usage(
        destination / "resource_usage.json",
        {
            "phase": "p0",
            "segment": segment,
            "started_at_step": next_step,
            "completed_steps": max_steps,
            "steps_in_segment": max_steps - next_step,
            "wall_time_seconds": time.perf_counter() - segment_started,
            "device": str(device),
            "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
            "peak_cuda_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "peak_cuda_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            ),
            "teacher_cache_entries": len(teacher_cache),
        },
    )
    return P0RunResult(
        final_checkpoint=final_checkpoint,
        completed_steps=max_steps,
        teacher_cache_entries=len(teacher_cache),
        dev_metrics=dev_metrics,
    )


def _deterministic_p0_dev_example(episode: Episode, config: ExperimentConfig) -> P0Example:
    by_prefix: dict[int, list[Probe]] = {}
    for probe in episode.probes:
        by_prefix.setdefault(probe.prefix_end, []).append(probe)
    eligible = [
        prefix_end
        for prefix_end, probes in by_prefix.items()
        if prefix_end >= config.k_init and len(probes) >= config.probes_per_prefix
    ]
    if not eligible:
        raise ValueError("dev episode has no P0-eligible probe prefix")
    prefix_end = max(eligible)
    return P0Example(
        episode,
        prefix_end,
        config.k_init,
        tuple(by_prefix[prefix_end][: config.probes_per_prefix]),
    )


def _save_p0_checkpoint(
    destination: Path,
    completed_steps: int,
    episode_limit: int,
    config: ExperimentConfig,
    backbone: LatentMemoryBackbone,
    writer: JointMemoryWriter,
    value_network: GrowthValueNetwork,
    trainer: P0Trainer,
) -> Path:
    checkpoint_path = destination / "checkpoints" / f"p0-step-{completed_steps:06d}.pt"
    save_model_checkpoint(
        checkpoint_path,
        "p0",
        config,
        trainable_model_state(backbone, writer, value_network),
        trainer.optimizer.state_dict(),
        {"next_step": completed_steps, "episode_limit": episode_limit},
        capture_rng_state(),
    )
    return checkpoint_path


def _run_manifest(
    config: ExperimentConfig,
    tokenizer_name_or_path: str,
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[3]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        cwd=repository_root,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        cwd=repository_root,
        text=True,
    ).stdout
    return {
        "framework_version": FRAMEWORK_VERSION,
        "phase": "p0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "model_name_or_path": config.model_name_or_path,
        "model_revision": config.model_revision,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "model_seed": config.model_seed,
        "device": str(device),
        "model_dtype": str(model_dtype),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")


def _append_resource_usage(path: Path, segment: dict[str, Any]) -> None:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("resource usage root must be a JSON object")
        _require_exact_fields(
            payload,
            frozenset({"framework_version", "segments"}),
            "resource usage",
        )
        if payload["framework_version"] != FRAMEWORK_VERSION:
            raise ValueError(f"framework_version must be {FRAMEWORK_VERSION!r}")
        if not isinstance(payload["segments"], list):
            raise TypeError("resource usage segments must be an array")
        segments = [*payload["segments"], segment]
    else:
        segments = [segment]
    _write_json(path, {"framework_version": FRAMEWORK_VERSION, "segments": segments})


def _require_exact_fields(mapping: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(mapping)
    if actual != set(expected):
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"invalid {label} fields; missing={missing}, unknown={unknown}")


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
