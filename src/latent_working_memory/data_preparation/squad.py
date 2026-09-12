from __future__ import annotations

import argparse
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


def prepare_squad(
    train_path: Path,
    dev_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    output_file: Path,
    seed: int = 20260907,
) -> dict[str, Any]:
    """Index original articles without materializing training trajectories."""
    if output_file.exists():
        raise FileExistsError("use a new SQuAD record file")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    articles = load_articles(train_path, "train") + load_articles(dev_path, "dev")
    assignments = assign_sources(articles, seed)
    records = []
    for article in articles:
        paragraphs = article["paragraphs"]
        lengths = [
            len(ids)
            for ids in tokenizer(
                [p["context"] + "\n\n" for p in paragraphs], add_special_tokens=False
            )["input_ids"]
        ]
        records.append(
            {
                "document_id": article["document_id"],
                "official_split": article["official_split"],
                "article_index": int(article["document_id"].rsplit(":", 1)[1]),
                "title": article["title"],
                **assignments[article["document_id"]],
                "input_tokens": sum(lengths),
                "paragraph_tokens": lengths,
                "questions": sum(len(p["qas"]) for p in paragraphs),
            }
        )
    report = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S UTC+08:00"),
        "seed": seed,
        "tokenizer": str(Path(tokenizer.name_or_path).resolve()),
        "source_files": {"train": str(train_path.resolve()), "dev": str(dev_path.resolve())},
        "articles": records,
        "splits": {},
    }
    for split in ("train", "dev", "test", "excluded"):
        rows = [r for r in records if r["split"] == split]
        report["splits"][split] = {
            "articles": len(rows),
            "paragraphs": sum(len(r["paragraph_tokens"]) for r in rows),
            "questions": sum(r["questions"] for r in rows),
            "input_tokens": sum(r["input_tokens"] for r in rows),
        }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Index original SQuAD articles and lengths")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = prepare_squad(args.train_file, args.dev_file, tokenizer, args.output_file, args.seed)
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
