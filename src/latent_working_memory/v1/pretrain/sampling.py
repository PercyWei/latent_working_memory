from __future__ import annotations

import math
import random
from dataclasses import dataclass
from collections import deque

from transformers import PreTrainedTokenizerBase

from latent_working_memory.v1.backbone import ReadTokens
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.data import Episode
from latent_working_memory.v1.pretrain.tokenization import TokenizationPool
from latent_working_memory.v1.pretrain.curriculum import (
    epoch_distribution,
    maximal_quotas,
    validate_curriculum,
)


@dataclass(frozen=True, slots=True)
class PretrainExample:
    episode: Episode
    ae: ReadTokens | None
    lm: ReadTokens | None
    capacity: int
    loss_weight: float = 1.0

    @property
    def input_length(self) -> int:
        return len(self.episode.input_ids)


def read_tokens(
    episode: Episode, tokenizer: PreTrainedTokenizerBase
) -> tuple[ReadTokens | None, ReadTokens | None]:
    read = episode.reads[0]
    ids = tuple(tokenizer.encode(read.references[0].text, add_special_tokens=False))
    if read.task == "ae" and ids != episode.input_ids:
        raise ValueError(
            "AE reference tokens must equal write input_ids; check tokenizer provenance"
        )
    target = ReadTokens(
        tuple(tokenizer.encode(read.prompt, add_special_tokens=False)),
        ids + (tokenizer.eos_token_id,),
    )
    return (target, None) if read.task == "ae" else (None, target)


def capacity_weights(
    config: ExperimentConfig,
    input_length: int,
    ae: ReadTokens | None,
    lm: ReadTokens | None,
    epoch: int,
) -> dict[int, float]:
    if (
        (ae is None and lm is None)
        or not 0 < input_length <= config.max_input_tokens
        or input_length + 1 > config.write_context_tokens
        or (lm is not None and len(lm.target_ids) - 1 > config.max_continuation_tokens)
    ):
        return {}
    progress = (
        1.0
        if config.ratio_curriculum_epochs == 1
        else min(max((epoch - 1) / (config.ratio_curriculum_epochs - 1), 0.0), 1.0)
    )
    candidates: dict[int, float] = {}
    for ratio, early, late in zip(
        config.pretrain_compression_ratios,
        config.ratio_weights_start,
        config.ratio_weights_end,
        strict=True,
    ):
        capacity = min(config.k_limit, max(config.pretrain_k_min, math.ceil(input_length / ratio)))
        if any(
            1 + capacity + len(t.prompt_ids) + len(t.target_ids) > config.read_context_tokens
            for t in (ae, lm)
            if t is not None
        ):
            continue
        candidates[capacity] = (
            candidates.get(capacity, 0.0) + early * (1 - progress) + late * progress
        )
    return candidates


class EpochSampler:
    """One maximum-size, exactly stratified, non-repeating sample plan per epoch."""

    def __init__(
        self,
        index,
        tokenizer,
        config,
        training,
        seed,
        batch_size,
        epochs,
        max_samples_per_epoch=None,
        tokenization=None,
        prefetch_batches=2,
    ):
        if type(epochs) is not int or epochs <= 0 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive integers")
        if type(prefetch_batches) is not int or prefetch_batches < 0:
            raise ValueError("prefetch_batches must be a non-negative integer")
        self.tokenization = tokenization or TokenizationPool(tokenizer, config)
        self.prefetch_batches = prefetch_batches
        self.pending = deque()
        self.submitted_cursor = 0
        self.index, self.tokenizer, self.config = index, tokenizer, config
        self.training, self.batch_size, self.epochs = training, batch_size, epochs
        self.max_samples_per_epoch = max_samples_per_epoch
        self.rng = random.Random(seed)
        self.capacity_rng = random.Random(f"{seed}:capacity")
        self.pools = {}
        for i, entry in enumerate(index.entries):
            source, task, size = entry[0], entry[6], entry[7]
            bound = next(b for b in config.input_length_bounds if size <= b)
            self.pools.setdefault((source, task, bound), []).append(i)
        validate_curriculum(training, index.sources, config.input_length_bounds)
        self.available = {k: len(v) for k, v in self.pools.items()}
        self.plans = []
        for epoch in range(1, epochs + 1):
            probabilities = epoch_distribution(training, epoch)
            quotas, limiting, unit = maximal_quotas(
                self.available, probabilities, batch_size, max_samples_per_epoch
            )
            self.plans.append((probabilities, quotas, limiting, unit))
        self.total_steps = sum(sum(q.values()) // batch_size for _, q, _, _ in self.plans)
        self.epoch, self.cursor, self.visits = 0, 0, 0
        self.order = []

    def start_epoch(self):
        self.epoch += 1
        if self.epoch > self.epochs:
            raise StopIteration
        _, quotas, _, _ = self.plans[self.epoch - 1]
        self.order = [i for key, n in quotas.items() for i in self.rng.sample(self.pools[key], n)]
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.submitted_cursor = 0

    def sample_batch(self):
        if self.cursor == len(self.order):
            self.start_epoch()
        if not self.pending:
            self._submit_next()
        batch = self.pending.popleft().result()
        examples = []
        for sample in batch:
            episode = sample.episode
            target = ReadTokens(sample.prompt_ids, sample.target_ids)
            ae, lm = (target, None) if episode.reads[0].task == "ae" else (None, target)
            candidates = capacity_weights(self.config, len(episode.input_ids), ae, lm, self.epoch)
            candidates = {k: w for k, w in candidates.items() if w > 0}
            if not candidates:
                raise ValueError(f"no legal weighted capacity for {episode.episode_id}")
            if self.config.compression_mode == "sample":
                capacity = self.capacity_rng.choices(list(candidates), list(candidates.values()))[0]
                examples.append(PretrainExample(episode, ae, lm, capacity))
            else:
                total = sum(candidates.values())
                examples.extend(
                    PretrainExample(episode, ae, lm, k, w / total) for k, w in candidates.items()
                )
        self.cursor += len(batch)
        self.visits += len(batch)
        # Only future text is prepared. Neither epoch RNG nor capacity RNG advances here.
        while len(self.pending) < self.prefetch_batches and self.submitted_cursor < len(self.order):
            self._submit_next()
        return examples

    def _submit_next(self):
        selected = self.order[self.submitted_cursor : self.submitted_cursor + self.batch_size]
        self.pending.append(self.tokenization.submit_batch(self.index.batch_entries(selected)))
        self.submitted_cursor += len(selected)

    def close(self):
        for future in self.pending:
            future.cancel()
        self.pending.clear()
        self.submitted_cursor = self.cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def epoch_report(self, epoch):
        probabilities, quotas, limiting, unit = self.plans[epoch - 1]

        def label(key):
            return "/".join(map(str, key))

        return {
            "epoch": epoch,
            "max_samples_per_epoch": self.max_samples_per_epoch,
            "samples": sum(quotas.values()),
            "steps": sum(quotas.values()) // self.batch_size,
            "quota_unit": unit,
            "active_available": sum(self.available.get(k, 0) for k in probabilities),
            "unselected": sum(self.available.get(k, 0) for k in probabilities)
            - sum(quotas.values()),
            "cells": [
                {
                    "cell": label(k),
                    "available": self.available.get(k, 0),
                    "probability": float(p),
                    "selected": quotas[k],
                }
                for k, p in probabilities.items()
            ],
            "bottlenecks": [label(k) for k in limiting],
        }

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "visits": self.visits,
            "order": self.order,
            "rng": self.rng.getstate(),
            "capacity_rng": self.capacity_rng.getstate(),
        }

    def load_state_dict(self, state):
        self.close()
        epoch, order, cursor = state["epoch"], state["order"], state["cursor"]
        if (
            not 1 <= epoch <= self.epochs
            or not 0 <= cursor <= len(order)
            or cursor % self.batch_size
        ):
            raise ValueError("invalid epoch progress")
        if len(order) != sum(self.plans[epoch - 1][1].values()) or len(set(order)) != len(order):
            raise ValueError("epoch plan differs from configured quotas")
        expected = self.plans[epoch - 1][1]
        actual = {}
        for i in order:
            if type(i) is not int or not 0 <= i < len(self.index.entries):
                raise ValueError("epoch plan contains an invalid sample index")
            entry = self.index.entries[i]
            key = (
                entry[0],
                entry[6],
                next(b for b in self.config.input_length_bounds if entry[7] <= b),
            )
            actual[key] = actual.get(key, 0) + 1
        if (
            actual != expected
            or state["visits"]
            != sum(sum(q.values()) for _, q, _, _ in self.plans[: epoch - 1]) + cursor
        ):
            raise ValueError("epoch data differ from configured quotas")
        self.epoch, self.order, self.cursor, self.visits = epoch, order, cursor, state["visits"]
        self.submitted_cursor = cursor
        self.rng.setstate(state["rng"])
        self.capacity_rng.setstate(state["capacity_rng"])
