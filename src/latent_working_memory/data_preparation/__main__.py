from __future__ import annotations

import argparse
import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer

from latent_working_memory.v1.config import load_config
from latent_working_memory.data_preparation.pipeline import prepare_sources, prepare_variant
from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.data_preparation.sources import parquet_records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare independent, balanced FineWeb AE/LM datasets"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--stage", choices=("sources", "semantic", "random", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, help="Local HuggingFaceFW-fineweb directory")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--topic-annotations", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.stage not in {"semantic", "random"}:
        parser.error("recovery requires an explicit semantic or random stage")
    config = load_config(args.config)
    recipe = PreparationConfig.load(args.recipe)
    if args.max_documents is not None:
        recipe = replace(recipe, max_documents=args.max_documents)
    report = {}
    if args.stage in {"sources", "all"}:
        if args.dataset_dir is None:
            parser.error("source preparation requires --dataset-dir")
        files = sorted((args.dataset_dir / config.pretrain_subset).glob("*.parquet"))
        if not files:
            parser.error("no local Parquet files found")
        with closing(parquet_records(files, config.data_seed)) as records:
            report["sources"] = prepare_sources(records, config, args.output_dir, recipe)
    if args.stage != "sources":
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, revision=config.model_revision, local_files_only=True
        )
        annotations = (
            json.loads(args.topic_annotations.read_text()) if args.topic_annotations else None
        )
        for variant in ("semantic", "random") if args.stage == "all" else (args.stage,):
            report[variant] = prepare_variant(
                args.output_dir,
                variant,
                tokenizer,
                config,
                recipe,
                annotations,
                resume=args.resume,
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
