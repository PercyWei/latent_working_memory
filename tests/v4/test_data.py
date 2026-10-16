import hashlib
import json
import random

import pytest

from latent_working_memory.v4.data import (
    TokenizedEpisode,
    dataset_identity,
    load_tokenized_episodes,
)
from latent_working_memory.v4.training_data import EpochSampler


class IntegerTokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [int(token) for token in text.split()]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is True
        assert enable_thinking is False
        content = []
        for message in messages:
            if message["role"] == "system":
                continue
            text = message["content"].removeprefix("Document:\n")
            content.extend(text.replace("\n\nQuestion:\n", " ").split())
        return [int(token) for token in content]


def document_row(text):
    tokens = text.split()
    return {
        "text": " ".join(tokens[:-2]),
        "qas": [{"question": tokens[-2], "answer": tokens[-1]}],
    }


def test_data_keeps_exact_tokens_and_accepts_inclusive_length_bounds(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                document_row("1 2 3 4 5 6"),
                document_row("1 2 3 4 5 6 7 8"),
            ]
        )
        + "\n"
    )
    episodes = load_tokenized_episodes(path, IntegerTokenizer(), 5, 6, model_window=8)
    short_group = hashlib.blake2b(b"1 2 3 4", digest_size=16).hexdigest()
    long_group = hashlib.blake2b(b"1 2 3 4 5 6", digest_size=16).hexdigest()
    assert episodes == (
        TokenizedEpisode(f"{short_group}:0", (1, 2, 3, 4, 5), (6,), short_group),
        TokenizedEpisode(f"{long_group}:0", (1, 2, 3, 4, 5, 6, 7), (8,), long_group),
    )


@pytest.mark.parametrize("text", ["1 2 3 4 5", "1 2 3 4 5 6 7 8 9"])
def test_invalid_lengths_fail_without_truncating(tmp_path, text):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(document_row(text)) + "\n")
    with pytest.raises(ValueError, match="no episodes remain"):
        load_tokenized_episodes(path, IntegerTokenizer(), 5, 8, model_window=8)


@pytest.mark.parametrize(
    "rows, message",
    [
        ("", "at least one document"),
        ("\n", "invalid JSON"),
        ((json.dumps(document_row("1 2 3 4")) + "\n") * 2, "duplicate text"),
    ],
)
def test_canonical_jsonl_contract(tmp_path, rows, message):
    path = tmp_path / "episodes.jsonl"
    path.write_text(rows)
    with pytest.raises(ValueError, match=message):
        load_tokenized_episodes(path, IntegerTokenizer(), 1, 8, model_window=8)


def test_epoch_order_is_reproducible_without_consuming_global_rng():
    episodes = tuple(TokenizedEpisode(str(index), (index,), (index,)) for index in range(12))
    sampler = EpochSampler(episodes, 42, shuffle=True)
    before = random.getstate()
    order = list(sampler)
    assert random.getstate() == before
    assert order == list(sampler)
    sampler.set_epoch(1)
    assert order != list(sampler)
    assert set(order) == set(range(len(episodes)))


def test_resume_identity_covers_ids_order_and_tokenized_content():
    episodes = (TokenizedEpisode("a", (1,), (2,)), TokenizedEpisode("b", (3, 4), (5,)))
    identity = dataset_identity(episodes)
    assert identity["episodes"] == 2
    assert identity["tokens"] == 5
    assert identity == dataset_identity(episodes)
    assert identity != dataset_identity(episodes[::-1])
    assert identity != dataset_identity((TokenizedEpisode("a", (1,), (3,)), episodes[1]))
    assert identity != dataset_identity((TokenizedEpisode("c", (1,), (2,)), episodes[1]))
