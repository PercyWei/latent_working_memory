from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.data import (
    Episode,
    Event,
    Probe,
    build_encoder_cells,
    fact_snapshot,
    fixed_size_partition,
    generate_dataset,
    generate_episode,
    partition_cells,
    read_episodes,
    write_episodes,
)


class CharacterTokenizer:
    name_or_path = "test-character-tokenizer"

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def _episode() -> Episode:
    events = (
        Event("e0", 0, 2, "Ada", "code", "A1", "set"),
        Event("e1", 2, 4, "Ada", "code", "B2", "set"),
        Event("e2", 4, 6, "Ada", "code", "", "retract"),
    )
    probes = (
        Probe("p0", 2, "Current code?", "A1", "recall", ("e0",)),
        Probe("p1", 4, "Current code?", "B2", "update", ("e1",)),
        Probe("p2", 6, "Current code?", "unknown", "update", ("e2",)),
    )
    return Episode(1, "episode-0", tuple(range(6)), events, probes)


def test_episode_jsonl_round_trip_and_prefix_truth(tmp_path: Path) -> None:
    episode = _episode()
    output = tmp_path / "episodes.jsonl"
    write_episodes([episode], output)
    assert read_episodes(output) == [episode]

    assert fact_snapshot(episode.events, 2).current[("Ada", "code")].value == "A1"
    assert fact_snapshot(episode.events, 4).current[("Ada", "code")].value == "B2"
    assert ("Ada", "code") not in fact_snapshot(episode.events, 6).current


def test_episode_rejects_future_evidence() -> None:
    with pytest.raises(ValueError, match="fully visible"):
        Episode(
            1,
            "bad",
            tuple(range(4)),
            (Event("e0", 2, 4, "Ada", "code", "A1", "set"),),
            (Probe("p0", 2, "Code?", "A1", "recall", ("e0",)),),
        )


def test_cells_are_shared_across_update_partitions() -> None:
    cells = build_encoder_cells(tuple(range(10)), cell_tokens=3)
    assert [(cell.source_start, cell.source_end) for cell in cells] == [
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 10),
    ]
    fine = fixed_size_partition(cells, 1)
    coarse = partition_cells(cells, (2, 2))
    assert tuple(token for chunk in fine for token in chunk.input_ids) == tuple(range(10))
    assert tuple(token for chunk in coarse for token in chunk.input_ids) == tuple(range(10))
    assert coarse[0].cells == cells[:2]


def test_synthetic_episode_has_probes_at_every_committed_prefix() -> None:
    episode = generate_episode(
        CharacterTokenizer(),
        "train-00000",
        7,
        768,
        1536,
        128,
        3,
        0,
    )
    expected_prefixes = set(range(128, len(episode.input_ids) + 1, 128))
    expected_prefixes.add(len(episode.input_ids))
    observed_prefixes = {probe.prefix_end for probe in episode.probes}
    assert observed_prefixes == expected_prefixes
    assert all(
        len([probe for probe in episode.probes if probe.prefix_end == prefix]) == 3
        for prefix in expected_prefixes
    )


def test_dataset_splits_use_separate_template_families(tmp_path: Path) -> None:
    config = replace(
        ExperimentConfig(),
        train_episodes=1,
        dev_episodes=1,
        test_episodes=1,
        min_episode_tokens=768,
        max_episode_tokens=1536,
        cell_tokens=128,
    )
    generate_dataset(config, CharacterTokenizer(), tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "dataset_manifest.json",
    }
    assert all(
        len(read_episodes(tmp_path / f"{split}.jsonl")) == 1 for split in ("train", "dev", "test")
    )
    with pytest.raises(FileExistsError):
        generate_dataset(config, CharacterTokenizer(), tmp_path)
