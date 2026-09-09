from __future__ import annotations

import random
from typing import Any, Mapping

from transformers import PreTrainedTokenizerBase

from latent_working_memory.data_preparation.fineweb import span_episode
from latent_working_memory.data_preparation.segmentation import sentence_spans
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.data import Episode


class RandomSpans:
    """Draw independent raw spans in an input-length interval, without paired examples."""

    def __init__(
        self,
        record: Mapping[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        config: ExperimentConfig,
    ):
        if not tokenizer.is_fast:
            raise ValueError("random spans require a fast tokenizer with character offsets")
        self.record, self.tokenizer, self.config = record, tokenizer, config
        self.offsets = [
            (a, b)
            for a, b in tokenizer(
                record["text"], add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
            if a < b
        ]
        sentences = sentence_spans(record["text"])
        self.sentence_starts = {s.start for s in sentences}
        self.sentence_ends = {s.end for s in sentences}

    def sample(self, task: str, lower: int, upper: int, rng: random.Random) -> Episode | None:
        continuation = task == "continuation"
        maximum = min(
            upper,
            self.config.max_input_tokens - int(continuation),
            len(self.offsets) - int(continuation),
        )
        if maximum <= lower:
            return None
        length = rng.randint(lower + 1, maximum)
        first = rng.randrange(len(self.offsets) - length + 1 - int(continuation))
        start, end = self.offsets[first][0], self.offsets[first + length - 1][1]
        target_end = None
        if continuation:
            target_length = rng.randint(
                1,
                min(
                    self.config.max_continuation_tokens,
                    len(self.offsets) - first - length,
                    self.config.max_input_tokens - length,
                ),
            )
            target_end = self.offsets[first + length + target_length - 1][1]
        episode = span_episode(
            self.record, self.tokenizer, self.config, start, end, target_end, "random", "random"
        )
        if episode is None or not lower < len(episode.input_ids) <= upper:
            return None
        # Isolated raw slices may tokenize differently at their boundary. Only the interval matters.
        text = self.record["text"]
        x = text[start:end]
        episode.sources[0].provenance.update(
            input_starts_at_sentence=start + len(x) - len(x.lstrip()) in self.sentence_starts,
            input_ends_at_sentence=start + len(x.rstrip()) in self.sentence_ends,
            target_ends_at_sentence=(end + len(text[end:target_end].rstrip()) in self.sentence_ends)
            if continuation
            else None,
        )
        return episode
