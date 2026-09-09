from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from typing import Any, Mapping

from transformers import PreTrainedTokenizerBase

from latent_working_memory.v1.config import GRANULARITIES, ExperimentConfig
from latent_working_memory.v1.data import Episode, Read, Reference, Source
from latent_working_memory.v1.sampling import capacity_weights, read_tokens
from latent_working_memory.data_preparation.dedup import source_key
from latent_working_memory.data_preparation.segmentation import sentence_spans
from latent_working_memory.data_preparation.quality import (
    BIBLIOGRAPHY,
    QUALITY_FILTER,
    sentence_rejection_reason,
    sentence_quality_flags,
    document_rejection_reason,
)

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
    # JSON-normalized config is shared by preparation and training validation.
    raw = json.loads(json.dumps(config.to_dict()))
    return {key: raw[key] for key in DATA_CONFIG_FIELDS} | {"quality_filter": QUALITY_FILTER}


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


def document_episodes(
    record: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    topic_annotation: dict[str, Any] | None = None,
    quality_counts: Counter[str] | None = None,
    rejected_spans: tuple[tuple[int, int], ...] = (),
) -> list[Episode]:
    text, document_id, url = record["text"], record["id"], record["url"]
    if not all(isinstance(v, str) and v for v in (text, document_id, url)):
        raise ValueError("FineWeb text, id and url must be non-empty strings")
    counts = quality_counts if quality_counts is not None else Counter()
    rejection = document_rejection_reason(record)
    if rejection:
        counts[f"documents_rejected_{rejection}"] += 1
        return []
    sentences = sentence_spans(text)
    if not sentences:
        return []
    sentence_texts = [text[s.start : s.end] for s in sentences]
    reference_paragraphs = {
        paragraph
        for paragraph, line in enumerate(re.finditer(r"[^\r\n]+", text))
        if BIBLIOGRAPHY.search(line.group())
    }
    repetitions = Counter(" ".join(s.lower().split()) for s in sentence_texts)
    reasons = [
        "quality_review"
        if any(sentence.start < end and start < sentence.end for start, end in rejected_spans)
        else "bibliography"
        if sentence.paragraph in reference_paragraphs
        else "repeated_sentence"
        if repetitions[" ".join(s.lower().split())] > QUALITY_FILTER["max_sentence_occurrences"]
        else sentence_rejection_reason(s)
        for sentence, s in zip(sentences, sentence_texts, strict=True)
    ]
    counts["sentences_examined"] += len(sentences)
    counts.update(f"sentences_rejected_{reason}" for reason in reasons if reason)
    counts.update(
        f"sentences_flagged_{flag}"
        for text in sentence_texts
        for flag in sentence_quality_flags(text)
    )
    invalid_prefix = [0]
    for reason in reasons:
        invalid_prefix.append(invalid_prefix[-1] + (reason is not None))
    source_id = source_key(url)
    rng = random.Random(f"{config.data_seed}:{document_id}")
    paragraphs: list[tuple[int, int]] = []
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
    clean_runs = []
    for i, reason in enumerate(reasons):
        if reason is None:
            if clean_runs and clean_runs[-1][1] == i:
                clean_runs[-1] = (clean_runs[-1][0], i + 1)
            else:
                clean_runs.append((i, i + 1))
    # Contiguous runs preserve useful text on both sides of excluded sentences.
    for begin, end in clean_runs:
        cursor = begin
        while cursor < end:
            stop = cursor + 1
            while stop < end:
                ids = tokenizer.encode(
                    text[sentences[cursor].start : sentences[stop].end], add_special_tokens=False
                )
                if len(ids) > config.max_input_tokens:
                    break
                stop += 1
            if stop - cursor > 1:
                candidates["sentence_group"].add((cursor, stop))
            cursor = stop
    if clean_runs:
        for _ in range(config.views_per_granularity * 16):
            begin, stop = rng.choice(clean_runs)
            start = rng.randrange(begin, stop)
            end = rng.randint(start + 1, stop)
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
            if invalid_prefix[end] != invalid_prefix[start]:
                continue
            x_start, x_end = sentences[start].start, sentences[end - 1].end
            x = text[x_start:x_end]
            input_ids = tuple(tokenizer.encode(x, add_special_tokens=False))
            if not 0 < len(input_ids) <= config.max_input_tokens:
                continue
            legal_stops = []
            for stop in range(end + 1, len(sentences) + 1):
                if reasons[stop - 1] is not None:
                    break
                y = text[x_end : sentences[stop - 1].end]
                if (
                    len(tokenizer.encode(y, add_special_tokens=False))
                    > config.max_continuation_tokens
                ):
                    break
                legal_stops.append(stop)
            y_stop = rng.choice(legal_stops) if legal_stops else end
            y_end = sentences[y_stop - 1].end
            y = text[x_end:y_end]
            episode_id = f"{document_id}:{granularity}:{start}:{end}:{y_stop}"
            provenance = {
                "dataset": config.pretrain_dataset,
                "subset": config.pretrain_subset,
                "url": url,
                "date": record["date"],
                "dump": record["dump"],
                "file_path": record["file_path"],
                "x_char_span": [x_start, x_end],
                "y_char_span": [x_end, y_end] if legal_stops else None,
                "sentence_range": [start, end],
                "continuation_sentence_range": [end, y_stop] if legal_stops else None,
                "granularity": granularity,
                "boundary_method": "offline_topic" if granularity == "topic_group" else "pysbd",
                "boundary_model": topic_annotation["model"]
                if granularity == "topic_group"
                else None,
                "tokenizer_name_or_path": config.model_name_or_path,
                "tokenizer_revision": config.model_revision,
            }
            reads = [
                Read(
                    episode_id + ":ae",
                    "ae",
                    len(input_ids),
                    config.ae_prompt,
                    (Reference(x, ()),),
                )
            ]
            if legal_stops:
                reads.append(
                    Read(
                        episode_id + ":lm",
                        "continuation",
                        len(input_ids),
                        config.lm_prompt,
                        (Reference(y, ()),),
                    )
                )
            episode = Episode(
                episode_id,
                input_ids,
                (len(input_ids),),
                (Source(source_id, document_id, 0, len(input_ids), provenance),),
                tuple(reads),
            )
            ae, lm = read_tokens(episode, tokenizer)
            if not capacity_weights(config, len(input_ids), ae, lm, 0):
                continue
            bucket = next(
                (b for b in config.input_length_bounds if len(input_ids) <= b),
                config.max_input_tokens,
            )
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
        episodes.extend(selected)
    return episodes
