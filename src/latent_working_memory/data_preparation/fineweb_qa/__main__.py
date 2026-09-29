"""Run one explicit stage of the local FineWeb QA pilot."""

import argparse
import json
from pathlib import Path

from latent_working_memory.data_preparation.fineweb_qa.pipeline import (
    annotate,
    diagnose,
    finalize,
    prepare,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "annotate", "diagnose", "finalize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Number of frozen documents to annotate")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage != "annotate" and args.limit is not None:
        parser.error("--limit applies only to annotate")
    if args.stage == "prepare":
        result = prepare(config)
    elif args.stage == "annotate":
        result = annotate(config, args.limit)
    elif args.stage == "diagnose":
        result = diagnose(config)
    else:
        result = finalize(config)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
