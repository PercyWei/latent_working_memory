from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random

from transformers import AutoTokenizer

from latent_working_memory.data_preparation.squad import load_articles
from latent_working_memory.v1.data import Episode, Read, Reference, Source


QA_PROMPT = (
    "Answer the question using the information stored in memory. Give only the answer.\n"
    "Question: {question}\nAnswer:"
)


class SquadDataset:
    """Read original articles; tokenize selected trajectories on demand.

    Episode.reads contains all question candidates at their earliest legal boundary,
    not a mandatory execution schedule.
    """

    def __init__(self, index_path: Path) -> None:
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.index["tokenizer"], local_files_only=True
        )
        self.records = {r["document_id"]: r for r in self.index["articles"]}
        self.articles = {}
        for split, path in self.index["source_files"].items():
            self.articles.update({a["document_id"]: a for a in load_articles(Path(path), split)})

    def select(self, split: str, min_tokens: int, max_tokens: int) -> list[str]:
        """Select complete articles in an inclusive length interval."""
        if split not in {"train", "dev", "test"} or not 0 <= min_tokens <= max_tokens:
            raise ValueError("invalid split or length interval")
        return [
            doc
            for doc, r in self.records.items()
            if r["split"] == split and min_tokens <= r["input_tokens"] <= max_tokens
        ]

    def episode(
        self, document_id: str, paragraph_count: int | None = None, paragraph_start: int = 0
    ) -> Episode:
        """Create an episode from a continuous range of original paragraphs."""
        article = self.articles[document_id]
        if self.records[document_id]["split"] == "excluded":
            raise ValueError("article excluded by source isolation")
        all_paragraphs = article["paragraphs"]
        if type(paragraph_start) is not int or not 0 <= paragraph_start < len(all_paragraphs):
            raise ValueError("paragraph_start must identify an article paragraph")
        count = (
            len(all_paragraphs) - paragraph_start if paragraph_count is None else paragraph_count
        )
        if type(count) is not int or not 1 <= count <= len(all_paragraphs) - paragraph_start:
            raise ValueError("paragraph_count must identify a non-empty paragraph range")
        paragraph_end = paragraph_start + count
        paragraphs = all_paragraphs[paragraph_start:paragraph_end]
        episode_id = f"{document_id}:paragraphs:{paragraph_start}:{paragraph_end}"
        ids, ends, sources, reads = [], [], [], []
        for i, paragraph in enumerate(paragraphs):
            start = len(ids)
            ids.extend(
                self.tokenizer.encode(paragraph["context"] + "\n\n", add_special_tokens=False)
            )
            end = len(ids)
            ends.append(end)
            sources.append(
                Source(
                    f"{episode_id}:p{i}",
                    document_id,
                    start,
                    end,
                    {
                        "official_split": article["official_split"],
                        "title": article["title"],
                        "paragraph_index": paragraph_start + i,
                        "context": paragraph["context"],
                        "char_span": [0, len(paragraph["context"])],
                        "questions": paragraph["qas"],
                    },
                )
            )
            for qa in paragraph["qas"]:
                reads.append(
                    Read(
                        f"{episode_id}:{qa['id']}",
                        "qa",
                        end,
                        QA_PROMPT.format(question=qa["question"]),
                        tuple(
                            Reference(text, ((start, end),))
                            for text in dict.fromkeys(a["text"] for a in qa["answers"])
                        ),
                    )
                )
        if [b - a for a, b in zip((0, *ends[:-1]), ends)] != self.records[document_id][
            "paragraph_tokens"
        ][paragraph_start:paragraph_end]:
            raise ValueError("source/tokenizer lengths differ from index; rebuild the index")
        return Episode(episode_id, tuple(ids), tuple(ends), tuple(sources), tuple(reads))


def sample_reads(
    episode: Episode,
    prefix_end: int,
    new_count: int,
    history_count: int,
    rng: random.Random,
    visits: dict[str, int],
    max_visits: int,
) -> tuple[Read, ...]:
    """Choose current and historical questions without changing memory or source data.

    visits belongs to one episode execution and is updated only for selected reads.
    Calls without question budget select nothing, including at the final boundary.
    """
    if prefix_end not in episode.write_ends:
        raise ValueError("read position must be a committed write boundary")
    if any(type(n) is not int or n < 0 for n in (new_count, history_count)):
        raise ValueError("read counts must be non-negative integers")
    if type(max_visits) is not int or max_visits < 1:
        raise ValueError("max_visits must be positive")
    selected = []
    for count, pool in (
        (new_count, [r for r in episode.reads if r.prefix_end == prefix_end]),
        (history_count, [r for r in episode.reads if r.prefix_end < prefix_end]),
    ):
        eligible = [r for r in pool if visits.get(r.read_id, 0) < max_visits]
        selected.extend(rng.sample(eligible, min(count, len(eligible))))
    for read in selected:
        visits[read.read_id] = visits.get(read.read_id, 0) + 1
    return tuple(replace(r, prefix_end=prefix_end) for r in selected)
