from __future__ import annotations

import hashlib
import json
import random
from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import Any, Mapping

from transformers import PreTrainedTokenizerBase

from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.v1.data import Episode, Read, Reference, Source
from latent_working_memory.data_preparation.dedup import source_key
from latent_working_memory.data_preparation.segmentation import sentence_spans

DATA_CONFIG_FIELDS = (
    "model_name_or_path",
    "model_revision",
    "pretrain_dataset",
    "pretrain_subset",
    "split_fractions",
    "data_seed",
    "ae_prompt",
    "lm_prompt",
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
    preparation: PreparationConfig,
    start: int,
    end: int,
    target_end: int | None,
    variant: str,
    granularity: str,
) -> Episode | None:
    """Build one task from original spans using data constraints and actual token counts."""
    text = record["text"]
    x = text[start:end]
    target = text[end:target_end] if target_end is not None else x
    if not x.strip() or not target.strip():
        return None
    ids = tuple(tokenizer.encode(x, add_special_tokens=False))
    target_length = (
        len(tokenizer.encode(target, add_special_tokens=False)) if target_end is not None else None
    )
    if not preparation.accepts_lengths(len(ids), target_length):
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
        "source_granularity": granularity,
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
    return episode


class SemanticSpans:
    """Draw X for each task independently; LM adds a contiguous sentence-boundary Y."""

    def __init__(
        self,
        record: Mapping[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        config: ExperimentConfig,
        preparation: PreparationConfig,
        topic_annotation: dict[str, Any] | None = None,
    ):
        self.record, self.tokenizer, self.config, self.recipe = (
            record,
            tokenizer,
            config,
            preparation,
        )
        if not tokenizer.is_fast:
            raise ValueError("semantic spans require a fast tokenizer with character offsets")
        text = record["text"]
        self.sentences = sentence_spans(text)
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        token_ends = [b for a, b in offsets if a < b]
        self.sentence_token_ends = [bisect_right(token_ends, s.end) for s in self.sentences]
        self.topic_annotation = topic_annotation
        ranges = {
            "sentence": {(i, i + 1) for i in range(len(self.sentences))},
            "paragraph": set(),
            "sentence_group": set(),
            "topic_group": set(),
        }
        paragraphs = []
        for i, sentence in enumerate(self.sentences):
            if paragraphs and self.sentences[paragraphs[-1][0]].paragraph == sentence.paragraph:
                paragraphs[-1] = (paragraphs[-1][0], i + 1)
            else:
                paragraphs.append((i, i + 1))
        ranges["paragraph"].update(paragraphs)
        if topic_annotation is not None:
            validate_topic_ranges(topic_annotation["ranges"], len(self.sentences))
            ranges["topic_group"].update(tuple(pair) for pair in topic_annotation["ranges"])
        # Propose continuous sentence groups across length intervals without a granularity quota.
        rng = random.Random(f"{config.data_seed}:{record['id']}:semantic:candidates")
        starts = rng.sample(
            range(len(self.sentences)),
            min(len(self.sentences), preparation.candidates_per_document),
        )
        for start in starts:
            origin = bisect_right(token_ends, self.sentences[start].start)
            for lower, upper in preparation.length_intervals():
                first = bisect_left(self.sentence_token_ends, origin + lower, lo=start + 1)
                stop = bisect_right(self.sentence_token_ends, origin + upper, lo=start + 1)
                if first < stop:
                    ranges["sentence_group"].add((start, rng.randrange(first, stop) + 1))
        self.candidates = defaultdict(list)
        for granularity, spans in ranges.items():
            for start, end in sorted(spans):
                x = text[self.sentences[start].start : self.sentences[end - 1].end]
                length = len(tokenizer.encode(x, add_special_tokens=False))
                if preparation.accepts_lengths(length, None):
                    bucket = next(b for b in preparation.length_bounds if length <= b)
                    self.candidates[bucket].append((granularity, start, end, length))

    def sample(self, task: str, lower: int, upper: int, rng: random.Random) -> Episode | None:
        candidates = self.candidates[upper]
        if task == "continuation":
            candidates = [c for c in candidates if self._target_ends(c[2], c[3])]
        if not candidates:
            return None
        granularity, start, end, length = rng.choice(candidates)
        target_sentence_end = None
        if task == "continuation":
            target_sentence_end = rng.choice(self._target_ends(end, length))
        episode = span_episode(
            self.record,
            self.tokenizer,
            self.config,
            self.recipe,
            self.sentences[start].start,
            self.sentences[end - 1].end,
            self.sentences[target_sentence_end - 1].end
            if target_sentence_end is not None
            else None,
            "semantic",
            granularity,
        )
        if episode is None or not lower <= len(episode.input_ids) <= upper:
            return None
        provenance = episode.sources[0].provenance
        provenance.update(
            sentence_range=[start, end],
            continuation_sentence_range=[end, target_sentence_end]
            if target_sentence_end is not None
            else None,
        )
        if granularity == "topic_group":
            provenance.update(
                boundary_method="offline_topic", boundary_model=self.topic_annotation["model"]
            )
        return episode

    def _target_ends(self, start: int, input_length: int) -> range:
        lower, upper = self.recipe.target_length_range(input_length)
        origin = self.sentence_token_ends[start - 1]
        first = bisect_left(self.sentence_token_ends, origin + lower, lo=start)
        stop = bisect_right(self.sentence_token_ends, origin + upper, lo=start)
        return range(first + 1, stop + 1)
