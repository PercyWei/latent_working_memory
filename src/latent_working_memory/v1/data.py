from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

READ_TASKS = frozenset({"ae", "continuation", "dialogue", "qa"})


def _exact_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(raw) != expected:
        raise ValueError(f"invalid {label} fields: expected {sorted(expected)}, got {sorted(raw)}")


def _span(start: int, end: int) -> None:
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("token spans must be non-empty, left-closed/right-open integers")


@dataclass(frozen=True, slots=True)
class Source:
    source_id: str
    document_id: str
    token_start: int
    token_end: int
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        _span(self.token_start, self.token_end)
        if not self.source_id or not self.document_id or not isinstance(self.provenance, dict):
            raise ValueError("source requires identifiers and provenance")


@dataclass(frozen=True, slots=True)
class Reference:
    text: str
    evidence_spans: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("reference text must be non-empty")
        for start, end in self.evidence_spans:
            _span(start, end)


@dataclass(frozen=True, slots=True)
class Read:
    read_id: str
    task: str
    prefix_end: int
    prompt: str
    references: tuple[Reference, ...]

    def __post_init__(self) -> None:
        if not self.read_id or self.task not in READ_TASKS or not self.references:
            raise ValueError("read requires an ID, supported task and references")
        if type(self.prefix_end) is not int or self.prefix_end < 0:
            raise ValueError("prefix_end must be non-negative")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("read prompt must be non-empty")
        if self.task != "qa" and (len(self.references) != 1 or self.references[0].evidence_spans):
            raise ValueError("sequence tasks require one reference with empty evidence_spans")


@dataclass(frozen=True, slots=True)
class Episode:
    episode_id: str
    input_ids: tuple[int, ...]
    write_ends: tuple[int, ...]
    sources: tuple[Source, ...]
    reads: tuple[Read, ...]

    def __post_init__(self) -> None:
        if (
            not self.episode_id
            or not self.input_ids
            or any(type(token) is not int or token < 0 for token in self.input_ids)
        ):
            raise ValueError("episode requires an ID and non-empty token IDs")
        if (
            not self.write_ends
            or any(type(end) is not int or end <= 0 for end in self.write_ends)
            or tuple(sorted(set(self.write_ends))) != self.write_ends
            or self.write_ends[-1] != len(self.input_ids)
        ):
            raise ValueError("write_ends must increase and end at input length")
        if not self.sources or any(
            source.token_end > len(self.input_ids) for source in self.sources
        ):
            raise ValueError("sources must cover valid input positions")
        if len({s.source_id for s in self.sources}) != len(self.sources):
            raise ValueError("source IDs must be unique within an episode")
        if len({r.read_id for r in self.reads}) != len(self.reads):
            raise ValueError("read IDs must be unique within an episode")
        for read in self.reads:
            if read.prefix_end not in {0, *self.write_ends}:
                raise ValueError("reads must align with committed write boundaries")
            for ref in read.references:
                if any(end > read.prefix_end for _, end in ref.evidence_spans):
                    raise ValueError("reference evidence must have arrived before the read")
                if (
                    read.task == "qa"
                    and not ref.evidence_spans
                    and read.prefix_end != len(self.input_ids)
                ):
                    raise ValueError("QA without localized evidence requires the complete input")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, raw: Mapping[str, Any]) -> Episode:
        _exact_keys(raw, {"episode_id", "input_ids", "write_ends", "sources", "reads"}, "episode")
        sources = []
        for source in raw["sources"]:
            _exact_keys(
                source,
                {"source_id", "document_id", "token_start", "token_end", "provenance"},
                "source",
            )
            sources.append(Source(**source))
        reads = []
        for read in raw["reads"]:
            _exact_keys(read, {"read_id", "task", "prefix_end", "prompt", "references"}, "read")
            refs = []
            for ref in read["references"]:
                _exact_keys(ref, {"text", "evidence_spans"}, "reference")
                refs.append(Reference(ref["text"], tuple(tuple(s) for s in ref["evidence_spans"])))
            reads.append(
                Read(read["read_id"], read["task"], read["prefix_end"], read["prompt"], tuple(refs))
            )
        return cls(
            raw["episode_id"],
            tuple(raw["input_ids"]),
            tuple(raw["write_ends"]),
            tuple(sources),
            tuple(reads),
        )


def read_episodes(path: str | Path) -> list[Episode]:
    with Path(path).open(encoding="utf-8") as handle:
        episodes = [Episode.from_record(json.loads(line)) for line in handle]
    if len({e.episode_id for e in episodes}) != len(episodes):
        raise ValueError("duplicate episode_id")
    return episodes


def write_episodes(episodes: Iterable[Episode], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_record(), ensure_ascii=False) + "\n")


class EpisodeIndex:
    """Keep document/view offsets in RAM; read token payloads only when sampled."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.groups: dict[str, dict[str, list[int]]] = {}
        self.offsets: list[int] = []
        self.ids: list[str] = []
        self.input_lengths: list[int] = []
        self.source_ids: set[str] = set()
        self.cluster_ids: set[str] = set()
        seen_ids = set()
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                episode = Episode.from_record(json.loads(line))
                if episode.episode_id in seen_ids:
                    raise ValueError("duplicate episode_id")
                seen_ids.add(episode.episode_id)
                if (
                    len(episode.sources) != 1
                    or episode.write_ends != (len(episode.input_ids),)
                    or tuple(read.task for read in episode.reads)
                    not in {("ae",), ("ae", "continuation")}
                    or any(read.prefix_end != len(episode.input_ids) for read in episode.reads)
                ):
                    raise ValueError(
                        "pretraining requires one write with AE and optional continuation"
                    )
                source = episode.sources[0]
                if (source.token_start, source.token_end) != (0, len(episode.input_ids)):
                    raise ValueError("pretraining source must cover the complete write input")
                granularity = source.provenance["granularity"]
                group = self.groups.setdefault(source.document_id, defaultdict(list))
                group[granularity].append(len(self.offsets))
                self.source_ids.add(source.source_id)
                self.cluster_ids.add(source.provenance["dedup_cluster"])
                self.offsets.append(offset)
                self.ids.append(episode.episode_id)
                self.input_lengths.append(len(episode.input_ids))
        if not self.offsets:
            raise ValueError(f"no episodes in {self.path}")

    def __getitem__(self, index: int) -> Episode:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return Episode.from_record(json.loads(handle.readline()))

    def panel(self, limit: int, seed: int) -> list[int]:
        # Round-robin documents before selecting a second view from any document.
        rng = random.Random(seed)
        documents = list(self.groups)
        rng.shuffle(documents)
        queues = []
        for document in documents:
            indices = [i for group in self.groups[document].values() for i in group]
            rng.shuffle(indices)
            queues.append(indices)
        result = []
        while queues and len(result) < limit:
            for queue in queues:
                result.append(queue.pop())
                if len(result) == limit:
                    break
            queues = [q for q in queues if q]
        return result

    def evaluation_panel(self, limit: int, seed: int) -> list[int]:
        rng = random.Random(seed)
        cells = defaultdict(list)
        for document, groups in self.groups.items():
            for granularity, indices in groups.items():
                for i in indices:
                    bucket = sum(self.input_lengths[i] > upper for upper in (32, 128, 512))
                    cells[(granularity, bucket)].append((document, i))
        for values in cells.values():
            rng.shuffle(values)
        keys = list(cells)
        rng.shuffle(keys)
        result, seen = [], set()
        while keys and len(result) < limit:
            active = []
            for key in keys:
                values = cells[key]
                while values and values[-1][0] in seen:
                    values.pop()
                if not values:
                    continue
                document, i = values.pop()
                result.append(i)
                seen.add(document)
                active.append(key)
                if len(result) == limit:
                    break
            keys = active
        return result
