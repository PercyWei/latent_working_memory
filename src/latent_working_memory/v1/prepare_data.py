from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from latent_working_memory.v1.config import load_config, write_resolved_config
from latent_working_memory.v1.data import generate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the canonical v1 synthetic dataset.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
        use_fast=True,
    )
    generate_dataset(config, tokenizer, args.output_dir, overwrite=args.overwrite)
    write_resolved_config(config, args.output_dir / "config.resolved.json")


if __name__ == "__main__":
    main()
