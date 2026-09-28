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
    parser = argparse.ArgumentParser(description="FineWeb QA construction pilot")
    parser.add_argument("stage", choices=("prepare", "annotate", "diagnose", "finalize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=(4, 24), default=24)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "annotate":
        annotate(config, args.limit)
    else:
        {"prepare": prepare, "diagnose": diagnose, "finalize": finalize}[args.stage](config)


if __name__ == "__main__":
    main()
