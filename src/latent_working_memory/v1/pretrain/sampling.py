from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase

from latent_working_memory.v1.backbone import ReadTokens
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.data import Episode, EpisodeIndex


@dataclass(frozen=True, slots=True)
class PretrainExample:
    episode: Episode
    ae: ReadTokens | None
    lm: ReadTokens | None
    capacity: int

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
    step: int,
) -> dict[int, float]:
    if (
        (ae is None and lm is None)
        or not 0 < input_length <= config.max_input_tokens
        or input_length + 1 > config.write_context_tokens
        or (lm is not None and len(lm.target_ids) - 1 > config.max_continuation_tokens)
    ):
        return {}
    progress = min(max(step / config.ratio_curriculum_steps, 0.0), 1.0)
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


class PretrainSampler:
    """Choose a length pool, cycle through its samples, then choose capacity."""

    def __init__(
        self,
        index: EpisodeIndex,
        tokenizer: PreTrainedTokenizerBase,
        config: ExperimentConfig,
        example_limit: int | None = None,
    ) -> None:
        self.index, self.tokenizer, self.config = index, tokenizer, config
        self.rng = random.Random(config.data_seed)
        allowed = set(index.panel(example_limit, config.data_seed)) if example_limit else None
        self.groups = {}
        for document, indices in index.groups.items():
            selected = [
                i
                for i in indices
                if (allowed is None or i in allowed)
                and (config.ae_weight if index.tasks[i] == "ae" else config.lm_weight) > 0
            ]
            if selected:
                self.groups[document] = selected
        if not self.groups:
            raise ValueError("no training views have a positive task weight")
        self.documents = sorted(self.groups)
        self.pools = {}
        if config.input_length_weights is None:
            self.pools["all"] = [i for indices in self.groups.values() for i in indices]
            self.pool_weights = {"all": 1.0}
        else:
            self.pool_weights = {}
            lower = 0
            for upper, weight in zip(
                config.input_length_bounds, config.input_length_weights, strict=True
            ):
                pool = {}
                for document, indices in self.groups.items():
                    selected = [i for i in indices if lower < index.input_lengths[i] <= upper]
                    if selected:
                        pool[document] = selected
                end_weight = (
                    config.input_length_weights_end[config.input_length_bounds.index(upper)]
                    if config.input_length_weights_end is not None
                    else weight
                )
                if max(weight, end_weight) > 0 and lower < config.max_input_tokens:
                    if not pool:
                        raise ValueError(
                            f"positive length weight has no views in ({lower}, {upper}]"
                        )
                    self.pools[str(upper)] = [i for indices in pool.values() for i in indices]
                    self.pool_weights[str(upper)] = weight
                lower = upper
        if not self.pools:
            raise ValueError("no input lengths have a positive sampling weight")
        self.orders = {key: list(pool) for key, pool in self.pools.items()}
        for order in self.orders.values():
            self.rng.shuffle(order)
        self.cursors = dict.fromkeys(self.orders, 0)
        self.visits = 0

    def length_weights(self, step: int) -> dict[str, float]:
        if self.config.input_length_weights_end is None:
            return self.pool_weights.copy()
        progress = min(max(step / self.config.input_length_curriculum_steps, 0), 1)
        return {
            str(bound): early * (1 - progress) + late * progress
            for bound, early, late in zip(
                self.config.input_length_bounds,
                self.config.input_length_weights,
                self.config.input_length_weights_end,
                strict=True,
            )
            if str(bound) in self.pools
        }

    def sample(self, step: int) -> PretrainExample:
        weights = self.length_weights(step)
        key = self.rng.choices(list(weights), list(weights.values()))[0]
        order = self.orders[key]
        if self.cursors[key] == len(order):
            self.rng.shuffle(order)
            self.cursors[key] = 0
        index = order[self.cursors[key]]
        self.cursors[key] += 1
        self.visits += 1
        episode = self.index[index]
        ae, lm = read_tokens(episode, self.tokenizer)
        candidates = capacity_weights(self.config, len(episode.input_ids), ae, lm, step)
        if not candidates or sum(candidates.values()) <= 0:
            raise ValueError(f"no legal weighted capacities for {episode.episode_id}")
        capacity = self.rng.choices(list(candidates), list(candidates.values()))[0]
        return PretrainExample(episode, ae, lm, capacity)

    def state_dict(self) -> dict[str, Any]:
        return {
            "orders": {key: order.copy() for key, order in self.orders.items()},
            "cursors": self.cursors.copy(),
            "visits": self.visits,
            "rng": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state["orders"].keys() != self.pools.keys()
            or state["cursors"].keys() != self.pools.keys()
        ):
            raise ValueError("sampler checkpoint length pools differ from the training data")
        for key, order in state["orders"].items():
            if sorted(order) != sorted(self.pools[key]):
                raise ValueError("sampler checkpoint samples differ from the training data")
            if not 0 <= state["cursors"][key] <= len(order):
                raise ValueError("invalid sampler cursor")
        self.orders = {key: order.copy() for key, order in state["orders"].items()}
        self.cursors, self.visits = state["cursors"].copy(), state["visits"]
        self.rng.setstate(state["rng"])


def task_weights_at(config: ExperimentConfig, step: int) -> tuple[float, float]:
    if step < config.pretrain_ae_warmup_steps:
        return 1.0, 0.0
    return config.ae_weight, config.lm_weight


class BalancedPretrainSampler:
    """Exact task/source quotas per update, with independent shuffled pool streams."""

    def __init__(self, index, tokenizer, config):
        self.index, self.tokenizer, self.config = index, tokenizer, config
        self.pools = {}
        for i in range(len(index.offsets)):
            episode = index[i]
            source = episode.sources[0].provenance["boundary_variant"]
            key = f"{episode.reads[0].task}/{source}"
            self.pools.setdefault(key, []).append(i)
        expected = {f"{task}/{source}" for task in ("ae", "continuation")
                    for source in ("semantic", "random")}
        if self.pools.keys() != expected:
            raise ValueError("balanced pretraining requires both tasks and both boundary sources")
        self.orders = {key: self.pools[key].copy() for key in sorted(self.pools)}
        self.rngs = {key: random.Random(f"{config.data_seed}:{key}") for key in self.orders}
        for key, order in self.orders.items():
            self.rngs[key].shuffle(order)
        self.cursors = dict.fromkeys(self.orders, 0)
        self.visits = 0

    def sample_batch(self, step, size):
        weights = task_weights_at(self.config, step)
        tasks = [t for t, w in zip(("ae", "continuation"), weights, strict=True) if w > 0]
        keys = [f"{task}/{source}" for task in tasks for source in ("semantic", "random")]
        if size % len(keys):
            raise ValueError("global batch must divide evenly across enabled task/source cells")
        examples = []
        for key in keys:
            order, rng = self.orders[key], self.rngs[key]
            for _ in range(size // len(keys)):
                if self.cursors[key] == len(order):
                    rng.shuffle(order)
                    self.cursors[key] = 0
                episode = self.index[order[self.cursors[key]]]
                self.cursors[key] += 1
                self.visits += 1
                ae, lm = read_tokens(episode, self.tokenizer)
                capacities = capacity_weights(self.config, len(episode.input_ids), ae, lm, step)
                if not capacities or sum(capacities.values()) <= 0:
                    raise ValueError(f"no legal capacity for {episode.episode_id}")
                capacity = rng.choices(list(capacities), list(capacities.values()))[0]
                examples.append(PretrainExample(episode, ae, lm, capacity))
        return examples

    def length_weights(self, step):
        return {str(self.config.max_input_tokens): 1.0}

    def state_dict(self):
        return {"orders": {k: v.copy() for k, v in self.orders.items()},
                "cursors": self.cursors.copy(), "visits": self.visits,
                "rngs": {k: r.getstate() for k, r in self.rngs.items()}}

    def load_state_dict(self, state):
        if state["orders"].keys() != self.pools.keys():
            raise ValueError("balanced sampler pools differ")
        for key, order in state["orders"].items():
            if sorted(order) != sorted(self.pools[key]):
                raise ValueError("balanced sampler data differ")
            if not 0 <= state["cursors"][key] <= len(order):
                raise ValueError("invalid balanced sampler cursor")
        self.orders = {k: v.copy() for k, v in state["orders"].items()}
        self.cursors, self.visits = state["cursors"].copy(), state["visits"]
        for key, rng in self.rngs.items():
            rng.setstate(state["rngs"][key])
