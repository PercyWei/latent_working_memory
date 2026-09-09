from __future__ import annotations

import argparse
import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer

from latent_working_memory.v1.config import load_config
from latent_working_memory.data_preparation.pipeline import prepare_fineweb
from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.data_preparation.sources import parquet_records
from latent_working_memory.data_preparation.scoring import DocumentScorer
from latent_working_memory.devices import validate_device


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare natural FineWeb AE/LM episodes")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--score-cache", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Local HuggingFaceFW-fineweb repository directory",
    )
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--topic-annotations", type=Path)
    args = parser.parse_args(argv)
    if args.max_documents is not None and args.max_documents <= 0:
        parser.error("--max-documents must be positive")
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    config = load_config(args.config)
    recipe = PreparationConfig.load(args.recipe) if args.recipe else PreparationConfig()
    if args.max_documents is not None:
        recipe = replace(recipe, max_documents=args.max_documents)
    device = torch.device(args.device)
    validate_device(device)
    scorer = None
    if recipe.review_model_name_or_path or recipe.fluency_model_name_or_path:
        if args.score_cache is None:
            parser.error("model scoring requires --score-cache")
        scorer = DocumentScorer(recipe, args.score_cache, device)
    annotations = None
    if args.topic_annotations:
        annotations = json.loads(args.topic_annotations.read_text())
    if config.granularity_weights[-1] > 0 and annotations is None:
        parser.error("a positive topic_group weight requires --topic-annotations")
    parquet_dir = args.dataset_dir / config.pretrain_subset
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        parser.error(f"no local Parquet input in {parquet_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, revision=config.model_revision, local_files_only=True
    )
    with closing(parquet_records(files, config.data_seed)) as records:
        metadata = prepare_fineweb(
            records, tokenizer, config, args.output_dir, recipe, annotations, scorer
        )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
