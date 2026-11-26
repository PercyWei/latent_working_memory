"""Run CPU statistics over a dataset adapter's final documents or slices."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from itertools import islice
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from transformers import AutoTokenizer

from .data_statistics.adapters.fineweb_factqa import iter_fineweb_factqa
from .data_statistics.adapters.locomo import iter_locomo
from .data_statistics.adapters.longmemeval import iter_longmemeval
from .data_statistics.adapters.qasper import iter_qasper
from .data_statistics.metrics import compute_metrics
from .data_statistics.report import build_summary, write_report
from .data_statistics.schema import MetricConfig

ADAPTERS = {
    "fineweb_factqa": iter_fineweb_factqa,
    "locomo": iter_locomo,
    "longmemeval": iter_longmemeval,
    "qasper": iter_qasper,
}


def _max_block_tokens(value):
    return None if value == "none" else int(value)


def main():
    defaults = MetricConfig()
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--dataset", choices=ADAPTERS, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--min-tokens", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--blocks", type=int, nargs="+", default=defaults.blocks)
    parser.add_argument(
        "--min-block-tokens", type=int, nargs="+", default=defaults.min_block_tokens
    )
    parser.add_argument(
        "--max-block-tokens", type=_max_block_tokens, nargs="+", default=defaults.max_block_tokens
    )
    parser.add_argument("--window-tokens", type=int, default=defaults.window_tokens)
    parser.add_argument("--recent-tokens", type=int, nargs="+", default=defaults.recent_tokens)
    args = parser.parse_args()
    if args.dataset == "locomo":
        if args.split is not None:
            parser.error("LoCoMo has no split; omit --split")
        documents = iter_locomo(args.dataset_dir)
    else:
        if args.split is None:
            parser.error(f"--split is required for {args.dataset}")
        documents = ADAPTERS[args.dataset](args.dataset_dir, args.split)
    if args.min_tokens is not None and args.min_tokens < 1:
        parser.error("min_tokens must be positive when specified")
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("max_tokens must be positive when specified")
    if (
        args.min_tokens is not None
        and args.max_tokens is not None
        and args.max_tokens < args.min_tokens
    ):
        parser.error("max_tokens must be at least min_tokens")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("max_samples must be positive")
    config = MetricConfig(
        blocks=tuple(args.blocks),
        min_block_tokens=tuple(args.min_block_tokens),
        max_block_tokens=tuple(args.max_block_tokens),
        window_tokens=args.window_tokens,
        recent_tokens=tuple(args.recent_tokens),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer with character offsets is required")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
            "%Y%m%d %H:%M:%S Asia/Shanghai"
        ),
        "dataset": args.dataset,
        "dataset_dir": str(args.dataset_dir.expanduser().resolve()),
        "split": args.split,
        "tokenizer": args.tokenizer,
        "tokenizer_class": type(tokenizer).__name__,
        "max_samples": args.max_samples,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "metrics": asdict(config),
        "skipped_segmentations": [
            {"blocks": blocks, "min_block_tokens": minimum, "max_block_tokens": maximum}
            for blocks, minimum, maximum in config.skipped_segmentations
        ],
    }
    (output / "config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.max_samples is not None:
        documents = islice(documents, args.max_samples)
    records = []
    with (output / "documents.jsonl").open("w", encoding="utf-8") as stream:
        for document in documents:
            encoded = tokenizer(
                document.text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
                verbose=False,
            )
            offsets = encoded["offset_mapping"]
            length = len(offsets)
            reason = None
            if not length:
                reason = "empty_text"
            elif args.min_tokens is not None and length < args.min_tokens:
                reason = "too_short"
            elif args.max_tokens is not None and length > args.max_tokens:
                reason = "too_long"
            record = {
                "doc_id": document.doc_id,
                "dataset": document.dataset,
                "split": document.split,
                "unit_kind": document.unit_kind,
                "source_ref": document.source_ref,
                "status": reason or "retained",
            }
            if reason:
                record.update(
                    metrics={"doc_tokens": length},
                    qas=[
                        {
                            "qa_id": qa.qa_id,
                            "evidence_granularity": qa.evidence_granularity,
                            "status": "document_excluded",
                            "metrics": {},
                        }
                        for qa in document.qas
                    ],
                )
            else:
                record.update(compute_metrics(document, offsets, config))
                if record["metrics"]["valid_qa_count"] == 0:
                    record["status"] = "no_answerable_qa"
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            records.append(record)
    summary = build_summary(records)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(output, summary)
    counts = dict(Counter(record["status"] for record in records))
    print(json.dumps({"output_dir": str(output), "documents": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
