from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any, Mapping

from transformers import PreTrainedTokenizerBase

from latent_working_memory.v1.config import GRANULARITIES, ExperimentConfig
from latent_working_memory.v1.data import Episode, Read, Reference, Source
from latent_working_memory.v1.sampling import capacity_weights, read_tokens
from latent_working_memory.data_preparation.dedup import source_key
from latent_working_memory.data_preparation.segmentation import sentence_spans

DATA_CONFIG_FIELDS = (
    "model_name_or_path",
    "model_revision",
    "pretrain_dataset",
    "pretrain_subset",
    "split_fractions",
    "views_per_granularity",
    "input_length_bounds",
    "data_seed",
    "max_input_tokens",
    "max_continuation_tokens",
    "write_context_tokens",
    "read_context_tokens",
    "ae_prompt",
    "lm_prompt",
    "pretrain_compression_ratios",
    "pretrain_k_min",
    "k_limit",
)


def data_contract(config: ExperimentConfig) -> dict[str, Any]:
    raw = json.loads(json.dumps(config.to_dict()))
    return {key: raw[key] for key in DATA_CONFIG_FIELDS}


def document_split(source_id: str, config: ExperimentConfig) -> str:
    digest = hashlib.blake2b(f"{config.data_seed}:{source_id}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / 2**64
    train, dev, _ = config.split_fractions
    return "train" if value < train else "dev" if value < train + dev else "test"


def validate_topic_ranges(ranges: list[list[int]], count: int) -> None:
    cursor = 0
    for start, end in ranges:
        if (
            type(start) is not int
            or type(end) is not int
            or start != cursor
            or not start < end <= count
        ):
            raise ValueError("topic ranges must be ordered, contiguous sentence index ranges")
        cursor = end
    if cursor != count:
        raise ValueError("topic ranges must cover all sentences")


def span_episode(
    record: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    start: int,
    end: int,
    target_end: int | None,
    variant: str,
    granularity: str,
) -> Episode | None:
    """Build one task from original character spans and enforce tokenizer/context budgets."""
    text = record["text"]
    x = text[start:end]
    target = text[end:target_end] if target_end is not None else x
    if not x.strip() or not target.strip():
        return None
    ids = tuple(tokenizer.encode(x, add_special_tokens=False))
    if not ids or len(ids) > config.max_input_tokens:
        return None
    task = "continuation" if target_end is not None else "ae"
    episode_id = f"{record['id']}:{variant}:{granularity}:{task}:{start}:{end}:{target_end}"
    provenance = {
        "dataset": config.pretrain_dataset,
        "subset": config.pretrain_subset,
        "url": record["url"],
        "date": record["date"],
        "dump": record["dump"],
        "file_path": record["file_path"],
        "parent_char_span": [start, target_end if target_end is not None else end],
        "x_char_span": [start, end],
        "y_char_span": [end, target_end] if target_end is not None else None,
        "granularity": granularity,
        "boundary_variant": variant,
        "boundary_method": "random_token" if variant == "random" else "pysbd",
        "tokenizer_name_or_path": config.model_name_or_path,
        "tokenizer_revision": config.model_revision,
    }
    episode = Episode(
        episode_id,
        ids,
        (len(ids),),
        (Source(source_key(record["url"]), record["id"], 0, len(ids), provenance),),
        (
            Read(
                episode_id + ":read",
                task,
                len(ids),
                config.lm_prompt if target_end is not None else config.ae_prompt,
                (Reference(target, ()),),
            ),
        ),
    )
    ae, lm = read_tokens(episode, tokenizer)
    return episode if capacity_weights(config, len(ids), ae, lm, 0) else None


def document_episodes(
    record: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    topic_annotation: dict[str, Any] | None = None,
) -> list[Episode]:
    """Generate sentence-boundary candidates; semantic quality is judged after construction."""
    text = record["text"]
    sentences = sentence_spans(text)
    if not sentences:
        return []
    rng = random.Random(f"{config.data_seed}:{record['id']}:semantic")
    paragraphs = []
    for i, sentence in enumerate(sentences):
        if not paragraphs or sentences[paragraphs[-1][0]].paragraph != sentence.paragraph:
            paragraphs.append((i, i + 1))
        else:
            paragraphs[-1] = (paragraphs[-1][0], i + 1)
    candidates = {g: set() for g in GRANULARITIES}
    candidates["sentence"].update((i, i + 1) for i in range(len(sentences)))
    candidates["paragraph"].update(paragraphs)
    candidates["paragraph_group"].update(
        (first[0], second[1]) for first, second in zip(paragraphs, paragraphs[1:])
    )
    cursor = 0
    while cursor < len(sentences):
        stop = cursor + 1
        while stop < len(sentences):
            if (
                len(
                    tokenizer.encode(
                        text[sentences[cursor].start : sentences[stop].end],
                        add_special_tokens=False,
                    )
                )
                > config.max_input_tokens
            ):
                break
            stop += 1
        if stop - cursor > 1:
            candidates["sentence_group"].add((cursor, stop))
        cursor = stop
    for _ in range(config.views_per_granularity * 16):
        start = rng.randrange(len(sentences))
        end = rng.randint(start + 1, len(sentences))
        if end - start > 1:
            candidates["sentence_group"].add((start, end))
    if topic_annotation is not None:
        validate_topic_ranges(topic_annotation["ranges"], len(sentences))
        candidates["topic_group"].update(tuple(pair) for pair in topic_annotation["ranges"])
    episodes = []
    for granularity in GRANULARITIES:
        ranges = sorted(candidates[granularity])
        rng.shuffle(ranges)
        by_length = defaultdict(list)
        for start, end in ranges:
            episode = span_episode(
                record,
                tokenizer,
                config,
                sentences[start].start,
                sentences[end - 1].end,
                None,
                "semantic",
                granularity,
            )
            if episode is None:
                continue
            provenance = episode.sources[0].provenance
            provenance["sentence_range"] = [start, end]
            provenance["continuation_sentence_range"] = None
            if granularity == "topic_group":
                provenance.update(
                    boundary_method="offline_topic", boundary_model=topic_annotation["model"]
                )
            bucket = next(b for b in config.input_length_bounds if len(episode.input_ids) <= b)
            by_length[bucket].append(episode)
        buckets = sorted(by_length)
        rng.shuffle(buckets)
        selected = []
        while buckets and len(selected) < config.views_per_granularity:
            remaining = []
            for bucket in buckets:
                selected.append(by_length[bucket].pop())
                if by_length[bucket]:
                    remaining.append(bucket)
                if len(selected) == config.views_per_granularity:
                    break
            buckets = remaining
        for episode in selected:
            episodes.append(episode)
            provenance = episode.sources[0].provenance
            start, end = provenance["sentence_range"]
            x_start, parent_end = provenance["parent_char_span"]
            cuts = [
                cut
                for cut in range(start + 1, end)
                if len(
                    tokenizer.encode(
                        text[sentences[cut - 1].end : parent_end], add_special_tokens=False
                    )
                )
                <= config.max_continuation_tokens
            ]
            if not cuts:
                continue
            cut = rng.choice(cuts)
            lm = span_episode(
                record,
                tokenizer,
                config,
                x_start,
                sentences[cut - 1].end,
                parent_end,
                "semantic",
                granularity,
            )
            if lm is not None:
                lm.sources[0].provenance.update(
                    sentence_range=[start, cut],
                    continuation_sentence_range=[cut, end],
                    boundary_method=provenance["boundary_method"],
                )
                if granularity == "topic_group":
                    lm.sources[0].provenance["boundary_model"] = provenance["boundary_model"]
                episodes.append(lm)
    return episodes
