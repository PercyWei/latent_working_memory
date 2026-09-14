from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random

from latent_working_memory.data_preparation.squad import (
    EXCLUSION_REASON,
    RECORD_FIELDS,
    SERIALIZATION,
    SPLITS,
    assign_sources,
    can_reuse_reference_lengths,
    load_articles,
    paragraph_lengths,
    split_statistics,
    tokenizer_identity,
)
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

    def __init__(self, dataset_dir: Path, tokenizer) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.tokenizer = tokenizer
        expected_files = {"preparation.json", *(f"{s}.jsonl" for s in SPLITS)}
        if {p.name for p in self.dataset_dir.iterdir()} != expected_files:
            raise ValueError(
                "SQuAD dataset requires preparation.json and train/dev/test.jsonl only"
            )
        self.preparation = json.loads((self.dataset_dir / "preparation.json").read_text())
        meta = self.preparation
        if meta["format"] != "squad-article-references-v1" or meta["source_version"] != "1.1":
            raise ValueError("unsupported SQuAD preparation format")
        if not isinstance(meta["preparation_id"], str) or not meta["preparation_id"]:
            raise ValueError("SQuAD preparation_id required")
        if type(meta["seed"]) is not int or meta["seed"] < 0:
            raise ValueError("invalid SQuAD split seed")
        if set(meta["source_files"]) != {"train", "dev"}:
            raise ValueError("SQuAD requires official train and dev sources")
        articles = [
            article
            for split in ("train", "dev")
            for article in load_articles(self.dataset_dir / meta["source_files"][split], split)
        ]
        sources = {a["document_id"]: a for a in articles}
        assignments = assign_sources(articles, meta["seed"])
        current = tokenizer_identity(tokenizer)
        self.reference_lengths = can_reuse_reference_lengths(
            meta["reference_tokenizer"], current, meta["serialization"]
        )
        rows = {
            s: [
                json.loads(line)
                for line in (self.dataset_dir / f"{s}.jsonl").read_text().splitlines()
            ]
            for s in SPLITS
        }
        rows["excluded"] = meta["excluded"]
        self.records, self.articles = {}, {}
        seen = set()
        for split, records in rows.items():
            for record in records:
                expected = RECORD_FIELDS | ({"reason"} if split == "excluded" else set())
                if set(record) != expected:
                    raise ValueError("invalid SQuAD article reference fields")
                doc = record["document_id"]
                if doc in seen or doc not in sources:
                    raise ValueError("duplicate or unknown SQuAD document_id")
                seen.add(doc)
                article = sources[doc]
                assignment = assignments[doc]
                if (
                    record["official_split"] != article["official_split"]
                    or type(record["article_index"]) is not int
                    or doc != f"squad:{record['official_split']}:{record['article_index']}"
                    or record["title"] != article["title"]
                    or record["group_id"] != assignment["group"]
                    or split != assignment["split"]
                ):
                    raise ValueError("SQuAD source location, group or split differs")
                lengths = record["reference_paragraph_tokens"]
                if (
                    type(record["paragraph_count"]) is not int
                    or record["paragraph_count"] != len(article["paragraphs"])
                    or type(record["question_count"]) is not int
                    or record["question_count"] != sum(len(p["qas"]) for p in article["paragraphs"])
                    or not isinstance(lengths, list)
                    or len(lengths) != record["paragraph_count"]
                    or any(type(n) is not int or n <= 0 for n in lengths)
                    or type(record["reference_input_tokens"]) is not int
                    or record["reference_input_tokens"] != sum(lengths)
                ):
                    raise ValueError("SQuAD article attributes or reference lengths differ")
                if split == "excluded":
                    if record["reason"] != EXCLUSION_REASON:
                        raise ValueError("invalid SQuAD exclusion reason")
                    continue
                actual_lengths = (
                    list(lengths)
                    if self.reference_lengths
                    else paragraph_lengths(article, tokenizer)
                )
                self.records[doc] = dict(
                    record,
                    split=split,
                    input_tokens=sum(actual_lengths),
                    paragraph_tokens=actual_lengths,
                    questions=record["question_count"],
                )
                self.articles[doc] = article
            if meta["splits"][split] != split_statistics(records):
                raise ValueError("SQuAD split statistics differ")
        if seen != sources.keys():
            raise ValueError("SQuAD source articles missing from splits or exclusions")
        self.index = {
            "format": "squad-runtime-v1",
            "preparation_id": meta["preparation_id"],
            "tokenizer": {
                k: v for k, v in current.items() if k not in {"name_or_path", "revision"}
            },
            "serialization": SERIALIZATION,
            "articles": list(self.records.values()),
        }
        if current["backend_fingerprint"] is None:
            self.index["tokenizer"]["name_or_path"] = current["name_or_path"]

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
                self.tokenizer.encode(
                    paragraph["context"] + SERIALIZATION["paragraph_suffix"],
                    add_special_tokens=False,
                    truncation=False,
                )
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
            raise ValueError(
                "source/tokenizer lengths differ; rebuild reference lengths or reload runtime data"
            )
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
