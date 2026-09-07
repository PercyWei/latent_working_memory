from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from transformers import PreTrainedTokenizerBase

from latent_working_memory.v1.config import ExperimentConfig


EPISODE_SCHEMA_VERSION = 1
PROBE_KINDS = frozenset({"recall", "update", "history", "compose"})
EVENT_OPERATIONS = frozenset({"set", "retract"})
ASSIGNED_HUB = "assigned hub"
REGION = "region"


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    token_start: int
    token_end: int
    entity: str
    relation: str
    value: str
    operation: str

    def __post_init__(self) -> None:
        if not self.event_id or not self.entity or not self.relation:
            raise ValueError("event_id, entity, and relation must not be empty")
        if self.operation not in EVENT_OPERATIONS:
            raise ValueError(f"operation must be one of {sorted(EVENT_OPERATIONS)}")
        if type(self.token_start) is not int or type(self.token_end) is not int:
            raise TypeError("event token spans must be integers")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("event token span must be non-empty and left-closed/right-open")
        if self.operation == "set" and not self.value:
            raise ValueError("set events must contain a value")
        if self.operation == "retract" and self.value:
            raise ValueError("retract event values must be empty")

    @classmethod
    def from_record(cls, raw: Mapping[str, Any]) -> Event:
        _require_exact_keys(
            raw,
            {
                "event_id",
                "token_start",
                "token_end",
                "entity",
                "relation",
                "value",
                "operation",
            },
            "event",
        )
        return cls(**raw)

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "entity": self.entity,
            "relation": self.relation,
            "value": self.value,
            "operation": self.operation,
        }


@dataclass(frozen=True, slots=True)
class Probe:
    probe_id: str
    prefix_end: int
    question: str
    answer: str
    kind: str
    evidence_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.probe_id or not self.question or not self.answer:
            raise ValueError("probe_id, question, and answer must not be empty")
        if type(self.prefix_end) is not int or self.prefix_end <= 0:
            raise ValueError("prefix_end must be a positive integer")
        if self.kind not in PROBE_KINDS:
            raise ValueError(f"kind must be one of {sorted(PROBE_KINDS)}")
        if any(not event_id for event_id in self.evidence_event_ids):
            raise ValueError("evidence_event_ids must not contain empty IDs")
        if len(set(self.evidence_event_ids)) != len(self.evidence_event_ids):
            raise ValueError("evidence_event_ids must be unique")

    @classmethod
    def from_record(cls, raw: Mapping[str, Any]) -> Probe:
        _require_exact_keys(
            raw,
            {"probe_id", "prefix_end", "question", "answer", "kind", "evidence_event_ids"},
            "probe",
        )
        values = dict(raw)
        evidence = values["evidence_event_ids"]
        if not isinstance(evidence, list):
            raise TypeError("probe evidence_event_ids must be an array")
        values["evidence_event_ids"] = tuple(evidence)
        return cls(**values)

    def to_record(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "prefix_end": self.prefix_end,
            "question": self.question,
            "answer": self.answer,
            "kind": self.kind,
            "evidence_event_ids": list(self.evidence_event_ids),
        }


@dataclass(frozen=True, slots=True)
class Episode:
    schema_version: int
    episode_id: str
    input_ids: tuple[int, ...]
    events: tuple[Event, ...]
    probes: tuple[Probe, ...]

    def __post_init__(self) -> None:
        if self.schema_version != EPISODE_SCHEMA_VERSION:
            raise ValueError(f"episode schema_version must be {EPISODE_SCHEMA_VERSION}")
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if not self.input_ids or any(
            type(token_id) is not int or token_id < 0 for token_id in self.input_ids
        ):
            raise ValueError("input_ids must contain non-negative integer token IDs")
        _validate_episode_relations(self)

    @classmethod
    def from_record(cls, raw: Mapping[str, Any]) -> Episode:
        _require_exact_keys(
            raw,
            {"schema_version", "episode_id", "input_ids", "events", "probes"},
            "episode",
        )
        if not isinstance(raw["input_ids"], list):
            raise TypeError("episode input_ids must be an array")
        if not isinstance(raw["events"], list) or not isinstance(raw["probes"], list):
            raise TypeError("episode events and probes must be arrays")
        return cls(
            schema_version=raw["schema_version"],
            episode_id=raw["episode_id"],
            input_ids=tuple(raw["input_ids"]),
            events=tuple(Event.from_record(event) for event in raw["events"]),
            probes=tuple(Probe.from_record(probe) for probe in raw["probes"]),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "input_ids": list(self.input_ids),
            "events": [event.to_record() for event in self.events],
            "probes": [probe.to_record() for probe in self.probes],
        }


@dataclass(frozen=True, slots=True)
class EncoderCell:
    input_ids: tuple[int, ...]
    source_start: int

    def __post_init__(self) -> None:
        if not self.input_ids:
            raise ValueError("an encoder cell must not be empty")
        if type(self.source_start) is not int or self.source_start < 0:
            raise ValueError("source_start must be a non-negative integer")

    @property
    def source_end(self) -> int:
        return self.source_start + len(self.input_ids)


@dataclass(frozen=True, slots=True)
class UpdateChunk:
    cells: tuple[EncoderCell, ...]

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("an update chunk must contain at least one cell")
        for previous, current in zip(self.cells, self.cells[1:], strict=False):
            if previous.source_end != current.source_start:
                raise ValueError("update chunk cells must be contiguous")

    @property
    def input_ids(self) -> tuple[int, ...]:
        return tuple(token_id for cell in self.cells for token_id in cell.input_ids)

    @property
    def source_start(self) -> int:
        return self.cells[0].source_start

    @property
    def source_end(self) -> int:
        return self.cells[-1].source_end


@dataclass(frozen=True, slots=True)
class FactValue:
    value: str
    event_id: str


@dataclass(frozen=True, slots=True)
class FactSnapshot:
    current: dict[tuple[str, str], FactValue]
    history: dict[tuple[str, str], tuple[Event, ...]]
    last_events: dict[tuple[str, str], Event]


def read_episodes(path: str | Path) -> list[Episode]:
    episodes: list[Episode] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {line_number}")
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise TypeError(f"episode at line {line_number} must be a JSON object")
            episode = Episode.from_record(raw)
            if episode.episode_id in seen_ids:
                raise ValueError(f"duplicate episode_id {episode.episode_id!r}")
            seen_ids.add(episode.episode_id)
            episodes.append(episode)
    return episodes


def write_episodes(episodes: Iterable[Episode], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    with output_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            if episode.episode_id in seen_ids:
                raise ValueError(f"duplicate episode_id {episode.episode_id!r}")
            seen_ids.add(episode.episode_id)
            json.dump(episode.to_record(), handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")


def build_encoder_cells(input_ids: tuple[int, ...], cell_tokens: int) -> tuple[EncoderCell, ...]:
    if not input_ids:
        raise ValueError("input_ids must not be empty")
    if type(cell_tokens) is not int or cell_tokens <= 0:
        raise ValueError("cell_tokens must be a positive integer")
    return tuple(
        EncoderCell(input_ids[start : start + cell_tokens], source_start=start)
        for start in range(0, len(input_ids), cell_tokens)
    )


def partition_cells(
    cells: tuple[EncoderCell, ...],
    group_sizes: tuple[int, ...],
) -> tuple[UpdateChunk, ...]:
    if not cells:
        raise ValueError("cells must not be empty")
    if not group_sizes or any(type(size) is not int or size <= 0 for size in group_sizes):
        raise ValueError("group_sizes must contain positive integers")
    if sum(group_sizes) != len(cells):
        raise ValueError("group_sizes must consume every cell exactly once")
    chunks: list[UpdateChunk] = []
    cursor = 0
    for size in group_sizes:
        chunks.append(UpdateChunk(cells[cursor : cursor + size]))
        cursor += size
    return tuple(chunks)


def fixed_size_partition(
    cells: tuple[EncoderCell, ...],
    cells_per_chunk: int,
) -> tuple[UpdateChunk, ...]:
    if type(cells_per_chunk) is not int or cells_per_chunk <= 0:
        raise ValueError("cells_per_chunk must be a positive integer")
    sizes = [cells_per_chunk] * (len(cells) // cells_per_chunk)
    remainder = len(cells) % cells_per_chunk
    if remainder:
        sizes.append(remainder)
    return partition_cells(cells, tuple(sizes))


def fact_snapshot(events: tuple[Event, ...], prefix_end: int) -> FactSnapshot:
    if type(prefix_end) is not int or prefix_end < 0:
        raise ValueError("prefix_end must be a non-negative integer")
    current: dict[tuple[str, str], FactValue] = {}
    histories: dict[tuple[str, str], list[Event]] = {}
    last_events: dict[tuple[str, str], Event] = {}
    for event in events:
        if event.token_end > prefix_end:
            continue
        key = (event.entity, event.relation)
        histories.setdefault(key, []).append(event)
        last_events[key] = event
        if event.operation == "set":
            current[key] = FactValue(event.value, event.event_id)
        else:
            current.pop(key, None)
    return FactSnapshot(
        current=current,
        history={key: tuple(history) for key, history in histories.items()},
        last_events=last_events,
    )


def generate_episode(
    tokenizer: PreTrainedTokenizerBase,
    episode_id: str,
    seed: int,
    min_tokens: int,
    max_tokens: int,
    cell_tokens: int,
    probes_per_prefix: int,
    template_family: int,
) -> Episode:
    if not episode_id:
        raise ValueError("episode_id must not be empty")
    if min_tokens <= 0 or max_tokens < min_tokens:
        raise ValueError("token bounds must satisfy 0 < min_tokens <= max_tokens")
    if cell_tokens <= 0 or probes_per_prefix <= 0:
        raise ValueError("cell_tokens and probes_per_prefix must be positive")
    if template_family not in (0, 1, 2):
        raise ValueError("template_family must be 0, 1, or 2")

    rng = random.Random(seed)
    people = [_identifier(rng, "Agent") for _ in range(12)]
    hubs = [_identifier(rng, "Hub") for _ in range(8)]
    regions = [_identifier(rng, "Region") for _ in range(6)]
    relations = ("access code", "signal color", "quota", "contact token")
    tokens: list[int] = []
    events: list[Event] = []
    current: dict[tuple[str, str], str] = {}
    all_keys: set[tuple[str, str]] = set()

    def append_event(entity: str, relation: str, value: str, operation: str) -> bool:
        text = _event_text(
            entity,
            relation,
            value,
            operation,
            template_family,
            len(events) % 2,
        )
        fragment = tokenizer.encode(text, add_special_tokens=False)
        if not fragment or any(type(token_id) is not int or token_id < 0 for token_id in fragment):
            raise ValueError("tokenizer must return non-negative integer token IDs")
        if len(tokens) + len(fragment) > max_tokens:
            return False
        event_id = f"{episode_id}:e{len(events):04d}"
        start = len(tokens)
        tokens.extend(fragment)
        event = Event(
            event_id=event_id,
            token_start=start,
            token_end=len(tokens),
            entity=entity,
            relation=relation,
            value=value,
            operation=operation,
        )
        events.append(event)
        key = (entity, relation)
        all_keys.add(key)
        if operation == "set":
            current[key] = value
        else:
            current.pop(key, None)
        return True

    for person, hub, region in zip(people[:3], hubs[:3], regions[:3], strict=True):
        append_event(person, ASSIGNED_HUB, hub, "set")
        append_event(hub, REGION, region, "set")
    for person in people[:6]:
        append_event(person, rng.choice(relations), _value(rng), "set")

    target_tokens = rng.randint(min_tokens, max_tokens)
    while len(tokens) < target_tokens:
        roll = rng.random()
        active_keys = sorted(current)
        known_keys = sorted(all_keys)
        if roll < 0.30 or not known_keys:
            entity = rng.choice(people)
            relation = rng.choice(relations)
            operation = "set"
            value = _value(rng)
        elif roll < 0.58 and active_keys:
            entity, relation = rng.choice(active_keys)
            operation = "set"
            value = _different_value(rng, current[(entity, relation)])
        elif roll < 0.70 and active_keys:
            entity, relation = rng.choice(active_keys)
            operation = "set"
            value = current[(entity, relation)]
        elif roll < 0.80 and active_keys:
            entity, relation = rng.choice(active_keys)
            operation = "retract"
            value = ""
        else:
            person = rng.choice(people)
            hub = rng.choice(hubs)
            if rng.random() < 0.5:
                entity, relation, value = person, ASSIGNED_HUB, hub
            else:
                entity, relation, value = hub, REGION, rng.choice(regions)
            operation = "set"
        if not append_event(entity, relation, value, operation):
            break

    if len(tokens) < min_tokens:
        raise ValueError("token bounds leave no room for a complete generated event")

    probes = _build_probes(
        episode_id,
        tuple(events),
        len(tokens),
        cell_tokens,
        probes_per_prefix,
        rng,
    )
    return Episode(
        schema_version=EPISODE_SCHEMA_VERSION,
        episode_id=episode_id,
        input_ids=tuple(tokens),
        events=tuple(events),
        probes=probes,
    )


def generate_dataset(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
    overwrite: bool = False,
) -> None:
    destination = Path(output_dir)
    outputs = {split: destination / f"{split}.jsonl" for split in ("train", "dev", "test")}
    manifest_path = destination / "dataset_manifest.json"
    existing = [path for path in (*outputs.values(), manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"dataset outputs already exist: {[str(path) for path in existing]}")
    destination.mkdir(parents=True, exist_ok=True)

    split_counts = {
        "train": config.train_episodes,
        "dev": config.dev_episodes,
        "test": config.test_episodes,
    }
    split_offsets = {"train": 0, "dev": 100_000, "test": 200_000}
    template_families = {"train": 0, "dev": 1, "test": 2}
    for split, count in split_counts.items():
        episodes = (
            generate_episode(
                tokenizer=tokenizer,
                episode_id=f"{split}-{index:05d}",
                seed=config.data_seed + split_offsets[split] + index,
                min_tokens=config.min_episode_tokens,
                max_tokens=config.max_episode_tokens,
                cell_tokens=config.cell_tokens,
                probes_per_prefix=config.probes_per_prefix,
                template_family=template_families[split],
            )
            for index in range(count)
        )
        write_episodes(episodes, outputs[split])

    manifest = {
        "schema_version": 1,
        "framework_version": config.framework_version,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", config.model_name_or_path),
        "model_revision": config.model_revision,
        "data_seed": config.data_seed,
        "split_counts": split_counts,
        "template_families": template_families,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _build_probes(
    episode_id: str,
    events: tuple[Event, ...],
    token_count: int,
    cell_tokens: int,
    probes_per_prefix: int,
    rng: random.Random,
) -> tuple[Probe, ...]:
    prefix_ends = list(range(cell_tokens, token_count + 1, cell_tokens))
    if not prefix_ends or prefix_ends[-1] != token_count:
        prefix_ends.append(token_count)

    probes: list[Probe] = []
    for prefix_index, prefix_end in enumerate(prefix_ends):
        snapshot = fact_snapshot(events, prefix_end)
        candidates = _probe_candidates(snapshot)
        rng.shuffle(candidates)
        selected: list[tuple[str, str, str, tuple[str, ...]]] = []
        seen_questions: set[str] = set()
        preferred_kinds = ("recall", "update", "history", "compose")
        for kind in preferred_kinds:
            for candidate in candidates:
                if candidate[0] == kind and candidate[1] not in seen_questions:
                    selected.append(candidate)
                    seen_questions.add(candidate[1])
                    break
            if len(selected) == probes_per_prefix:
                break
        for candidate in candidates:
            if len(selected) == probes_per_prefix:
                break
            if candidate[1] not in seen_questions:
                selected.append(candidate)
                seen_questions.add(candidate[1])
        while len(selected) < probes_per_prefix:
            suffix = len(selected)
            selected.append(
                (
                    "recall",
                    f"What is the current access code for Unknown-{prefix_index:03d}-{suffix}?",
                    "unknown",
                    (),
                )
            )

        for local_index, (kind, question, answer, evidence) in enumerate(selected):
            probes.append(
                Probe(
                    probe_id=f"{episode_id}:p{prefix_end:05d}:{local_index}",
                    prefix_end=prefix_end,
                    question=question,
                    answer=answer,
                    kind=kind,
                    evidence_event_ids=evidence,
                )
            )
    return tuple(probes)


def _probe_candidates(
    snapshot: FactSnapshot,
) -> list[tuple[str, str, str, tuple[str, ...]]]:
    candidates: list[tuple[str, str, str, tuple[str, ...]]] = []
    for (entity, relation), fact in sorted(snapshot.current.items()):
        candidates.append(
            (
                "recall",
                f"What is the current {relation} for {entity}?",
                fact.value,
                (fact.event_id,),
            )
        )
        set_events = [
            event for event in snapshot.history[(entity, relation)] if event.operation == "set"
        ]
        distinct_values = list(dict.fromkeys(event.value for event in set_events))
        if len(distinct_values) >= 2:
            previous = next(
                event for event in reversed(set_events[:-1]) if event.value != set_events[-1].value
            )
            candidates.append(
                (
                    "update",
                    f"After all updates, what is the current {relation} for {entity}?",
                    fact.value,
                    (previous.event_id, fact.event_id),
                )
            )
            candidates.append(
                (
                    "history",
                    f"What was the most recent different {relation} for {entity} before the current value {fact.value}?",
                    previous.value,
                    (previous.event_id, fact.event_id),
                )
            )

    for key, last_event in sorted(snapshot.last_events.items()):
        if key not in snapshot.current and last_event.operation == "retract":
            entity, relation = key
            previous_sets = [event for event in snapshot.history[key] if event.operation == "set"]
            evidence = (
                (previous_sets[-1].event_id, last_event.event_id)
                if previous_sets
                else (last_event.event_id,)
            )
            candidates.append(
                (
                    "update",
                    f"After all updates, what is the current {relation} for {entity}?",
                    "unknown",
                    evidence,
                )
            )

    for (entity, relation), hub_fact in sorted(snapshot.current.items()):
        if relation != ASSIGNED_HUB:
            continue
        region_fact = snapshot.current.get((hub_fact.value, REGION))
        if region_fact is None:
            continue
        candidates.append(
            (
                "compose",
                f"In which region is the hub assigned to {entity}?",
                region_fact.value,
                (hub_fact.event_id, region_fact.event_id),
            )
        )
    return candidates


def _event_text(
    entity: str,
    relation: str,
    value: str,
    operation: str,
    template_family: int,
    variant: int,
) -> str:
    if operation == "retract":
        templates = (
            (
                "Record: {entity}'s {relation} is withdrawn.\n",
                "Log entry: remove the {relation} assigned to {entity}.\n",
            ),
            (
                "Remove the recorded {relation} for {entity}.\n",
                "The {relation} previously listed for {entity} is void.\n",
            ),
            (
                "For {entity}, the earlier {relation} no longer applies.\n",
                "Withdraw {entity}'s recorded {relation}.\n",
            ),
        )
        return templates[template_family][variant].format(entity=entity, relation=relation)
    templates = (
        (
            "Record: {entity}'s {relation} is {value}.\n",
            "Log entry: for {entity}, {relation} is {value}.\n",
        ),
        (
            "Update for {entity}: {relation} equals {value}.\n",
            "The {relation} recorded for {entity} is {value}.\n",
        ),
        (
            "{entity} now has {value} as its {relation}.\n",
            "Assign {value} to {entity} as the current {relation}.\n",
        ),
    )
    return templates[template_family][variant].format(
        entity=entity,
        relation=relation,
        value=value,
    )


def _identifier(rng: random.Random, prefix: str) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return f"{prefix}-" + "".join(rng.choices(alphabet, k=6))


def _value(rng: random.Random) -> str:
    return _identifier(rng, "V")


def _different_value(rng: random.Random, previous: str) -> str:
    value = _value(rng)
    while value == previous:
        value = _value(rng)
    return value


def _validate_episode_relations(episode: Episode) -> None:
    event_ids: set[str] = set()
    previous_end = 0
    event_ends: dict[str, int] = {}
    for event in episode.events:
        if event.event_id in event_ids:
            raise ValueError(f"duplicate event_id {event.event_id!r}")
        if event.token_start < previous_end:
            raise ValueError("events must be ordered and must not overlap")
        if event.token_end > len(episode.input_ids):
            raise ValueError("event span exceeds episode input_ids")
        event_ids.add(event.event_id)
        event_ends[event.event_id] = event.token_end
        previous_end = event.token_end

    probe_ids: set[str] = set()
    for probe in episode.probes:
        if probe.probe_id in probe_ids:
            raise ValueError(f"duplicate probe_id {probe.probe_id!r}")
        if probe.prefix_end > len(episode.input_ids):
            raise ValueError("probe prefix_end exceeds episode input_ids")
        unknown_evidence = sorted(set(probe.evidence_event_ids) - event_ids)
        if unknown_evidence:
            raise ValueError(f"probe refers to unknown evidence events: {unknown_evidence}")
        if any(event_ends[event_id] > probe.prefix_end for event_id in probe.evidence_event_ids):
            raise ValueError("probe evidence must be fully visible at prefix_end")
        probe_ids.add(probe.probe_id)


def _require_exact_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"invalid {label} fields; missing={missing}, unknown={unknown}")
