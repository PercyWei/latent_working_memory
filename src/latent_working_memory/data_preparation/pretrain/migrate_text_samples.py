"""把已有 semantic/random Episode 语料迁移为文本，保留样本、顺序和 split。"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from latent_working_memory.data_preparation.pretrain.sources import load_sources
from latent_working_memory.data_preparation.pretrain.audit import compare_preparations
from latent_working_memory.data_preparation.pretrain.text_samples import TextSample, compact_metadata
from latent_working_memory.v1.data import Episode


def migrate(root, tokenizer):
    variants = ("semantic", "random")
    metadata = {v: json.loads((root / v / "preparation.json").read_text()) for v in variants}
    for variant, meta in metadata.items():
        if "contract" not in meta:
            raise ValueError(f"{variant} already uses text samples")
        expected = {
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "preparation.json",
            "documents.jsonl",
            "sample-decisions.jsonl",
            "progress.json",
            "audit.json",
        }
        if {p.name for p in (root / variant).iterdir()} - expected:
            raise ValueError(
                f"unexpected files in {variant}; preserve and inspect before migration"
            )
    documents = {row["record"]["id"]: row for row in load_sources(root / "source-pool.json")
                 if row["status"] == "eligible"}
    stage = root / ".text-migration"
    stage.mkdir()
    report = {}
    try:
        for variant in variants:
            destination = stage / variant
            destination.mkdir()
            meta = metadata[variant]
            counts = Counter()
            histogram = {
                s: {
                    t: {str(b): 0 for b in meta["recipe"]["length_bounds"]}
                    for t in ("ae", "continuation")
                }
                for s in ("train", "dev", "test")
            }
            seen = set()
            for split in ("train", "dev", "test"):
                path = root / variant / f"{split}.jsonl"
                with path.open() as source, (destination / path.name).open("w") as out:
                    for line in source:
                        episode = Episode.from_record(json.loads(line))
                        original = documents[episode.sources[0].document_id]
                        if original["split"] != split:
                            raise ValueError("source split mismatch")
                        sample = TextSample.from_episode(
                            episode, original["record"]["text"], tokenizer
                        )
                        if sample.dedup_cluster != original["cluster"] or sample.sample_id in seen:
                            raise ValueError("source cluster mismatch or duplicate sample ID")
                        seen.add(sample.sample_id)
                        out.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")
                        counts[f"{split}/{sample.task}"] += 1
                        bucket = next(
                            b
                            for b in meta["recipe"]["length_bounds"]
                            if sample.reference_input_tokens <= b
                        )
                        histogram[split][sample.task][str(bucket)] += 1
                print(
                    json.dumps({"variant": variant, "split": split, "counts": dict(counts)}),
                    flush=True,
                )
            if (
                any(meta["statistics"][key] != count for key, count in counts.items())
                or histogram != meta["input_histogram"]
            ):
                raise ValueError("migration changed sample counts or reference length distribution")
            compact = compact_metadata(meta, meta["audit"])
            (destination / "preparation.json").write_text(
                json.dumps(compact, ensure_ascii=False, indent=2) + "\n"
            )
            report[variant] = {
                "samples": sum(counts.values()),
                "before_bytes": sum(p.stat().st_size for p in (root / variant).iterdir()),
                "after_bytes": sum(p.stat().st_size for p in destination.iterdir()),
            }
        comparison = compare_preparations(stage, compact)
        # Keep the old directories until both replacements succeed; no long-term duplicate.
        moved = []
        try:
            for variant in variants:
                (root / variant).rename(stage / f"old-{variant}")
                moved.append(variant)
                (stage / variant).rename(root / variant)
            temporary = stage / "comparison.json"
            temporary.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(root / "comparison.json")
        except BaseException:
            for variant in reversed(moved):
                if (root / variant).exists():
                    (root / variant).rename(stage / variant)
                (stage / f"old-{variant}").rename(root / variant)
            raise
    except BaseException:
        if not any((stage / f"old-{v}").exists() for v in variants):
            shutil.rmtree(stage)
        raise
    shutil.rmtree(stage)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.data_root / "semantic/preparation.json").read_text())
    contract = meta["contract"]
    tokenizer = AutoTokenizer.from_pretrained(
        contract["model_name_or_path"], revision=contract["model_revision"], local_files_only=True
    )
    print(json.dumps(migrate(args.data_root, tokenizer), indent=2))


if __name__ == "__main__":
    main()
