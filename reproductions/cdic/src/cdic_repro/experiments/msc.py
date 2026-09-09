from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


SILENCE_TOKEN = "__SILENCE__"


@dataclass(frozen=True, slots=True)
class MscTurn:
    turn_id: str
    query: str
    response: str
    session_index: int
    pair_index: int


@dataclass(frozen=True, slots=True)
class MscEpisode:
    episode_id: str
    source_session_id: int
    turns: tuple[MscTurn, ...]
    dropped_unpaired_utterances: int = 0
    replaced_empty_utterances: int = 0

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if not self.turns:
            raise ValueError("MSC episode must contain at least one turn")


def msc_split_path(data_root: Path, session_id: int, split: str) -> Path:
    if session_id < 2 or session_id > 5:
        raise ValueError("session_id must be between 2 and 5")
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")
    if split == "train" and session_id == 5:
        raise ValueError("official MSC session 5 has no training split")
    path = data_root / "msc" / "msc_dialogue" / f"session_{session_id}" / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"MSC split file does not exist: {path}")
    return path


def load_msc_episodes(
    data_root: Path,
    session_id: int = 4,
    split: str = "train",
    max_episodes: int | None = None,
    max_turns_per_episode: int | None = None,
    strict_pairs: bool = False,
) -> tuple[MscEpisode, ...]:
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive when provided")
    if max_turns_per_episode is not None and max_turns_per_episode < 1:
        raise ValueError("max_turns_per_episode must be positive when provided")

    path = msc_split_path(data_root, session_id=session_id, split=split)
    episodes: list[MscEpisode] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
            record = json.loads(line)
            episodes.append(
                _parse_episode(
                    record,
                    source_session_id=session_id,
                    line_number=line_number,
                    max_turns=max_turns_per_episode,
                    strict_pairs=strict_pairs,
                )
            )
    if not episodes:
        raise ValueError(f"MSC split contains no episodes: {path}")
    return tuple(episodes)


def iter_episode_pairs(record: dict[str, object]) -> Iterator[tuple[int, int, str, str]]:
    previous_dialogs = record.get("previous_dialogs")
    current_dialog = record.get("dialog")
    if not isinstance(previous_dialogs, list) or not isinstance(current_dialog, list):
        raise TypeError("MSC record must contain previous_dialogs and dialog lists")

    sessions: list[object] = [
        previous.get("dialog") if isinstance(previous, dict) else None
        for previous in previous_dialogs
    ]
    sessions.append(current_dialog)
    for session_index, dialogue in enumerate(sessions, start=1):
        if not isinstance(dialogue, list):
            raise TypeError(f"MSC session {session_index} dialogue must be a list")
        for pair_index in range(0, len(dialogue) - 1, 2):
            query = _utterance_text(dialogue[pair_index])
            response = _utterance_text(dialogue[pair_index + 1])
            yield session_index, pair_index // 2 + 1, query, response


def summarize_msc_episodes(episodes: tuple[MscEpisode, ...]) -> dict[str, float | int]:
    turn_counts = [len(episode.turns) for episode in episodes]
    dropped = sum(episode.dropped_unpaired_utterances for episode in episodes)
    replaced_empty = sum(episode.replaced_empty_utterances for episode in episodes)
    source_utterances = 2 * sum(turn_counts) + dropped
    return {
        "episodes": len(episodes),
        "turns": sum(turn_counts),
        "source_utterances": source_utterances,
        "dropped_unpaired_utterances": dropped,
        "replaced_empty_utterances": replaced_empty,
        "min_turns": min(turn_counts),
        "max_turns": max(turn_counts),
        "mean_turns": sum(turn_counts) / len(turn_counts),
        "mean_source_utterances": source_utterances / len(turn_counts),
    }


def _parse_episode(
    record: object,
    source_session_id: int,
    line_number: int,
    max_turns: int | None,
    strict_pairs: bool,
) -> MscEpisode:
    if not isinstance(record, dict):
        raise TypeError(f"MSC line {line_number} must contain a JSON object")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"MSC line {line_number} is missing metadata")
    episode_id = metadata.get("initial_data_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError(f"MSC line {line_number} has no initial_data_id")

    sessions = record.get("previous_dialogs")
    current = record.get("dialog")
    if not isinstance(sessions, list) or not isinstance(current, list):
        raise TypeError(f"MSC line {line_number} has invalid dialogue fields")
    dialogue_lists = [
        previous.get("dialog") if isinstance(previous, dict) else None for previous in sessions
    ] + [current]
    if strict_pairs:
        for session_index, dialogue in enumerate(dialogue_lists, start=1):
            if not isinstance(dialogue, list):
                raise TypeError(f"MSC episode {episode_id} session {session_index} is invalid")
            if len(dialogue) % 2 != 0:
                raise ValueError(
                    f"MSC episode {episode_id} session {session_index} has an odd utterance count"
                )

    dropped_unpaired_utterances = sum(
        len(dialogue) % 2 for dialogue in dialogue_lists if isinstance(dialogue, list)
    )
    replaced_empty_utterances = sum(
        1
        for dialogue in dialogue_lists
        if isinstance(dialogue, list)
        for utterance in dialogue
        if _is_empty_utterance(utterance)
    )

    turns: list[MscTurn] = []
    for session_index, pair_index, query, response in iter_episode_pairs(record):
        turns.append(
            MscTurn(
                turn_id=f"{episode_id}:s{session_index}:p{pair_index}",
                query=query,
                response=response,
                session_index=session_index,
                pair_index=pair_index,
            )
        )
        if max_turns is not None and len(turns) >= max_turns:
            break
    return MscEpisode(
        episode_id=episode_id,
        source_session_id=source_session_id,
        turns=tuple(turns),
        dropped_unpaired_utterances=dropped_unpaired_utterances,
        replaced_empty_utterances=replaced_empty_utterances,
    )


def _utterance_text(value: object) -> str:
    if not isinstance(value, dict):
        raise TypeError("MSC utterance must be an object")
    text = value.get("text")
    if not isinstance(text, str):
        raise ValueError("MSC utterance text must be a string")
    return text.strip() or SILENCE_TOKEN


def _is_empty_utterance(value: object) -> bool:
    return (
        isinstance(value, dict) and isinstance(value.get("text"), str) and not value["text"].strip()
    )
