from __future__ import annotations

import json
from pathlib import Path

import pytest

from cdic_repro.experiments.msc.data import (
    SILENCE_TOKEN,
    load_msc_episodes,
    summarize_msc_episodes,
)


def write_msc_record(root: Path, record: dict[str, object]) -> None:
    split = root / "msc" / "msc_dialogue" / "session_4" / "train.txt"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps(record) + "\n", encoding="utf-8")


def make_record() -> dict[str, object]:
    return {
        "metadata": {"initial_data_id": "train:example", "session_id": 3},
        "previous_dialogs": [
            {
                "dialog": [
                    {"text": "q1"},
                    {"text": "r1"},
                    {"text": "q2"},
                    {"text": "r2"},
                ]
            }
        ],
        "dialog": [{"text": "q3"}, {"text": "r3"}],
    }


def test_load_msc_episode_flattens_sessions_in_chronological_order(tmp_path: Path) -> None:
    write_msc_record(tmp_path, make_record())

    episodes = load_msc_episodes(tmp_path, session_id=4, split="train")

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.episode_id == "train:example"
    assert [(turn.query, turn.response) for turn in episode.turns] == [
        ("q1", "r1"),
        ("q2", "r2"),
        ("q3", "r3"),
    ]
    assert [turn.session_index for turn in episode.turns] == [1, 1, 2]
    assert summarize_msc_episodes(episodes)["mean_source_utterances"] == 6.0


def test_msc_loader_supports_pilot_turn_limit(tmp_path: Path) -> None:
    write_msc_record(tmp_path, make_record())

    episode = load_msc_episodes(
        tmp_path,
        session_id=4,
        split="train",
        max_turns_per_episode=2,
    )[0]

    assert len(episode.turns) == 2


def test_msc_loader_rejects_odd_session_in_strict_mode(tmp_path: Path) -> None:
    record = make_record()
    record["dialog"] = [{"text": "unpaired"}]
    write_msc_record(tmp_path, record)

    with pytest.raises(ValueError, match="odd utterance count"):
        load_msc_episodes(tmp_path, session_id=4, split="train", strict_pairs=True)


def test_msc_loader_records_official_silence_and_unpaired_tail(tmp_path: Path) -> None:
    record = make_record()
    record["dialog"] = [{"text": ""}, {"text": "reply"}, {"text": "unpaired"}]
    write_msc_record(tmp_path, record)

    episode = load_msc_episodes(tmp_path, session_id=4, split="train")[0]

    assert episode.turns[-1].query == SILENCE_TOKEN
    assert episode.dropped_unpaired_utterances == 1
    assert episode.replaced_empty_utterances == 1
