from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import os
import shutil
import tempfile
import uuid
from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any
from zoneinfo import ZoneInfo

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_articles(path: Path, official_split: str) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw["version"] != "1.1":
        raise ValueError("SQuAD 1.1 input required")
    articles = []
    seen_questions = set()
    for index, article in enumerate(raw["data"]):
        if not isinstance(article["title"], str) or not article["title"]:
            raise ValueError("article title required")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if not isinstance(context, str) or not context.strip():
                raise ValueError("non-empty paragraph required")
            for qa in paragraph["qas"]:
                if not qa["id"] or qa["id"] in seen_questions:
                    raise ValueError("missing or duplicate question ID")
                seen_questions.add(qa["id"])
                if not qa["question"].strip() or not qa["answers"]:
                    raise ValueError(f"missing question or answers: {qa['id']}")
                for answer in qa["answers"]:
                    start, text = answer["answer_start"], answer["text"]
                    if (
                        type(start) is not int
                        or start < 0
                        or not isinstance(text, str)
                        or not text.strip()
                        or context[start : start + len(text)] != text
                    ):
                        raise ValueError(f"answer annotation mismatch: {qa['id']}")
        articles.append(
            dict(
                article,
                document_id=f"squad:{official_split}:{index}",
                official_split=official_split,
            )
        )
    return articles


def assign_sources(articles: list[dict[str, Any]], seed: int) -> dict[str, dict[str, str]]:
    parents = list(range(len(articles)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, article in enumerate(articles):
        keys = [("title", article["title"])] + [
            ("paragraph", p["context"]) for p in article["paragraphs"]
        ]
        for key in keys:
            if key in owners:
                parents[root(i)] = root(owners[key])
            else:
                owners[key] = i
    groups = {}
    for i, article in enumerate(articles):
        groups.setdefault(root(i), []).append(article)
    ordered = sorted(groups.values(), key=lambda g: min(a["document_id"] for a in g))
    train_groups = [g for g in ordered if all(a["official_split"] == "train" for a in g)]
    random.Random(seed).shuffle(train_groups)
    dev_count = max(1, round(len(train_groups) * 0.1)) if len(train_groups) > 1 else 0
    dev_ids = {a["document_id"] for g in train_groups[:dev_count] for a in g}
    assignments = {}
    for group in ordered:
        group_id = min(a["document_id"] for a in group)
        has_official_dev = any(a["official_split"] == "dev" for a in group)
        for article in group:
            doc = article["document_id"]
            split = (
                "test"
                if article["official_split"] == "dev"
                else "excluded"
                if has_official_dev
                else "dev"
                if doc in dev_ids
                else "train"
            )
            assignments[doc] = {"group": group_id, "split": split}
    return assignments


SPLITS = ("train", "dev", "test")
SERIALIZATION = {
    "paragraph_suffix": "\n\n",
    "tokenize_paragraphs_independently": True,
    "add_special_tokens": False,
    "includes_title_questions_answers": False,
}
RECORD_FIELDS = frozenset(
    {
        "document_id",
        "official_split",
        "article_index",
        "title",
        "group_id",
        "paragraph_count",
        "question_count",
        "reference_input_tokens",
        "reference_paragraph_tokens",
    }
)
EXCLUSION_REASON = "source_group_overlaps_official_dev"


def tokenizer_identity(tokenizer):
    """Only a complete fast-tokenizer backend can certify reference-length reuse.

    Slow tokenizers remain supported, but their lengths are recalculated at runtime.
    Padding/truncation are transient batch state and are explicitly disabled for measurement.
    """
    fingerprint = None
    if tokenizer.is_fast:
        backend = json.loads(tokenizer.backend_tokenizer.to_str())
        backend.pop("padding", None)
        backend.pop("truncation", None)
        behavior = {
            "backend": backend,
            "special_tokens": tokenizer.special_tokens_map,
            "split_special_tokens": tokenizer.split_special_tokens,
        }
        fingerprint = hashlib.sha256(
            json.dumps(behavior, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
    identity = {
        "name_or_path": str(tokenizer.name_or_path),
        "revision": tokenizer.init_kwargs.get("revision"),
        "tokenizer_class": type(tokenizer).__name__,
        "backend_fingerprint": fingerprint,
        "transformers_version": version("transformers"),
        "tokenizers_version": version("tokenizers"),
    }
    if not tokenizer.is_fast:
        identity["slow_configuration"] = json.loads(
            json.dumps(
                {
                    "vocabulary": tokenizer.get_vocab(),
                    "special_tokens": tokenizer.special_tokens_map,
                    "initialization": tokenizer.init_kwargs,
                },
                default=str,
            )
        )
    return identity


def can_reuse_reference_lengths(reference, current, serialization):
    keys = ("tokenizer_class", "backend_fingerprint", "transformers_version", "tokenizers_version")
    return (
        serialization == SERIALIZATION
        and current["backend_fingerprint"] is not None
        and all(reference.get(key) == current[key] for key in keys)
    )


def paragraph_lengths(article, tokenizer):
    return [
        len(ids)
        for ids in tokenizer(
            [p["context"] + SERIALIZATION["paragraph_suffix"] for p in article["paragraphs"]],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
    ]


def split_statistics(rows):
    return {
        "articles": len(rows),
        "paragraphs": sum(r["paragraph_count"] for r in rows),
        "questions": sum(r["question_count"] for r in rows),
        "reference_input_tokens": sum(r["reference_input_tokens"] for r in rows),
    }


def prepare_squad(
    train_path: Path,
    dev_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    seed: int = 20260907,
) -> dict[str, Any]:
    """Save article references and splits; original text and QA stay in the source files."""
    if output_dir.exists():
        raise FileExistsError("use a new SQuAD dataset directory")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    articles = load_articles(train_path, "train") + load_articles(dev_path, "dev")
    assignments = assign_sources(articles, seed)
    records = {split: [] for split in (*SPLITS, "excluded")}
    for article in articles:
        lengths = paragraph_lengths(article, tokenizer)
        assignment = assignments[article["document_id"]]
        record = {
            "document_id": article["document_id"],
            "official_split": article["official_split"],
            "article_index": int(article["document_id"].rsplit(":", 1)[1]),
            "title": article["title"],
            "group_id": assignment["group"],
            "paragraph_count": len(lengths),
            "question_count": sum(len(p["qas"]) for p in article["paragraphs"]),
            "reference_input_tokens": sum(lengths),
            "reference_paragraph_tokens": lengths,
        }
        if assignment["split"] == "excluded":
            record["reason"] = EXCLUSION_REASON
        records[assignment["split"]].append(record)
    report = {
        "format": "squad-article-references-v1",
        "preparation_id": str(uuid.uuid4()),
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S UTC+08:00"),
        "source_version": "1.1",
        "seed": seed,
        "split_policy": "Group articles transitively by exact title or paragraph equality. "
        "Use official dev as test; exclude overlapping official train articles. "
        "Shuffle remaining source groups by seed and use approximately 10% for dev.",
        "reference_tokenizer": tokenizer_identity(tokenizer),
        "serialization": SERIALIZATION,
        "source_files": {
            "train": os.path.relpath(train_path.resolve(), output_dir.resolve()),
            "dev": os.path.relpath(dev_path.resolve(), output_dir.resolve()),
        },
        "splits": {split: split_statistics(rows) for split, rows in records.items()},
        "excluded": records["excluded"],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output_dir.name + "-", dir=output_dir.parent))
    try:
        for split in SPLITS:
            with (staging / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for record in records[split]:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        (staging / "preparation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError("SQuAD destination was created during preparation")
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SQuAD article references and reference lengths"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if set(config) != {"train_file", "dev_file", "reference_tokenizer", "output_dir", "seed"}:
        raise ValueError("SQuAD preparation configuration fields differ")
    tokenizer = AutoTokenizer.from_pretrained(config["reference_tokenizer"], local_files_only=True)
    report = prepare_squad(
        Path(config["train_file"]),
        Path(config["dev_file"]),
        tokenizer,
        Path(config["output_dir"]),
        config["seed"],
    )
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
