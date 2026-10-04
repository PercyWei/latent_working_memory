import json
import random

import pytest

from latent_working_memory.v4.data import (
    TokenizedEpisode,
    dataset_identity,
    episode_order,
    load_tokenized_episodes,
)


class IntegerTokenizer:
    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [int(token) for token in text.split()]}


def test_data_keeps_exact_tokens_and_accepts_inclusive_length_bounds(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "short", "text": "1 2 3 4 5 6"},
                {"id": "long", "text": "1 2 3 4 5 6 7 8"},
            ]
        )
        + "\n"
    )
    episodes = load_tokenized_episodes(path, IntegerTokenizer(), 6, 8)
    assert episodes == (
        TokenizedEpisode("short", (1, 2, 3, 4, 5, 6)),
        TokenizedEpisode("long", (1, 2, 3, 4, 5, 6, 7, 8)),
    )


@pytest.mark.parametrize("text", ["1 2 3 4 5", "1 2 3 4 5 6 7 8 9"])
def test_invalid_lengths_fail_without_truncating(tmp_path, text):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps({"id": "length", "text": text}) + "\n")
    with pytest.raises(ValueError, match="no truncation"):
        load_tokenized_episodes(path, IntegerTokenizer(), 6, 8)


@pytest.mark.parametrize(
    "rows, message",
    [
        ("", "at least one episode"),
        ("\n", "invalid JSON"),
        ('{"id": "one", "text": "1 2", "label": 1}\n', "expected exactly"),
        ('{"id": 1, "text": "1 2"}\n', "expected exactly"),
        ('{"id": "one", "text": [1, 2]}\n', "expected exactly"),
        ('["one", "1 2"]\n', "expected exactly"),
        ('{"id": "one", "text": "1 2"}\n' * 2, "duplicate episode"),
    ],
)
def test_canonical_jsonl_contract(tmp_path, rows, message):
    path = tmp_path / "episodes.jsonl"
    path.write_text(rows)
    with pytest.raises(ValueError, match=message):
        load_tokenized_episodes(path, IntegerTokenizer(), 2, 8)


def test_epoch_order_is_reproducible_without_consuming_global_rng():
    episodes = tuple(TokenizedEpisode(str(index), (index,)) for index in range(12))
    before = random.getstate()
    order = episode_order(episodes, 42, 0)
    assert random.getstate() == before
    assert order == episode_order(episodes, 42, 0)
    assert order != episode_order(episodes, 42, 1)
    assert set(order) == set(episodes)


def test_resume_identity_covers_ids_order_and_tokenized_content():
    episodes = (TokenizedEpisode("a", (1, 2)), TokenizedEpisode("b", (3, 4, 5)))
    identity = dataset_identity(episodes)
    assert identity["episodes"] == 2
    assert identity["tokens"] == 5
    assert identity == dataset_identity(episodes)
    assert identity != dataset_identity(episodes[::-1])
    assert identity != dataset_identity((TokenizedEpisode("a", (1, 3)), episodes[1]))
    assert identity != dataset_identity((TokenizedEpisode("c", (1, 2)), episodes[1]))
