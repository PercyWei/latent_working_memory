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


def span_episode(
    record: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    preparation: PreparationConfig,
    start: int,
    end: int,
    target_end: int | None,
    variant: str,
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
    episode_id = f"{record['id']}:{variant}:{task}:{start}:{end}:{target_end}"
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
        "boundary_variant": variant,
        "boundary_method": "random_token" if variant == "random" else "pysbd_conservative",
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
        ranges = set()
        rng = random.Random(f"{config.data_seed}:{record['id']}:semantic:candidates")
        starts = rng.sample(
            range(len(self.sentences)),
            min(len(self.sentences), preparation.candidates_per_document),
        )
        for start in starts:
            origin = bisect_right(token_ends, self.sentences[start].start)
            for lower, upper in preparation.length_intervals():
                first = bisect_left(self.sentence_token_ends, origin + lower, lo=start)
                stop = bisect_right(self.sentence_token_ends, origin + upper, lo=start)
                if first < stop:
                    ranges.add((start, rng.randrange(first, stop) + 1))
        self.candidates = defaultdict(list)
        spans = sorted(ranges)
        # Batch exact slice tokenization; full-document offsets are only proposal estimates.
        texts = [text[self.sentences[a].start : self.sentences[b - 1].end] for a, b in spans]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"] if texts else []
        for (start, end), ids in zip(spans, encoded, strict=True):
            length = len(ids)
            if preparation.accepts_lengths(length, None):
                bucket = next(b for b in preparation.length_bounds if length <= b)
                self.candidates[bucket].append((start, end, length))
        self.continuations = {}
        self.lm_candidates = defaultdict(list)
        for bucket, candidates in self.candidates.items():
            for candidate in candidates:
                end, length = candidate[1:]
                ends = self._target_ends(end, length)
                self.continuations[(end, length)] = ends
                if ends:
                    self.lm_candidates[bucket].append(candidate)

    def available(self, task: str, lower: int, upper: int) -> bool:
        return bool((self.lm_candidates if task == "continuation" else self.candidates)[upper])

    def sample(self, task: str, lower: int, upper: int, rng: random.Random) -> Episode | None:
        candidates = (self.lm_candidates if task == "continuation" else self.candidates)[upper]
        if not candidates:
            return None
        start, end, length = rng.choice(candidates)
        target_sentence_end = None
        if task == "continuation":
            target_sentence_end = rng.choice(self.continuations[(end, length)])
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
        return episode

    def _target_ends(self, start: int, input_length: int) -> range:
        lower, upper = self.recipe.target_length_range(input_length)
        origin = self.sentence_token_ends[start - 1]
        first = bisect_left(self.sentence_token_ends, origin + lower, lo=start)
        stop = bisect_right(self.sentence_token_ends, origin + upper, lo=start)
        return range(first + 1, stop + 1)
