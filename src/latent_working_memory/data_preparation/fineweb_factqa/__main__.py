"""Run a source, annotation or finalization stage for FineWeb QA."""

import argparse
import json
from latent_working_memory.data_preparation.fineweb_factqa.campaign import initialize
from latent_working_memory.data_preparation.fineweb_factqa.config import (
    add_run_arguments,
    load_config,
    run_config,
)
from latent_working_memory.data_preparation.fineweb_factqa.pipeline import (
    annotate,
    finalize,
    prepare,
)
from latent_working_memory.data_preparation.fineweb_factqa.storage import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare-pool", "prepare", "annotate", "finalize"))
    add_run_arguments(parser)
    parser.add_argument("--limit", type=int, help="Number of frozen trajectories to annotate")
    args = parser.parse_args()
    if args.stage != "annotate" and args.limit is not None:
        parser.error("--limit applies only to annotate")
    if args.stage == "prepare-pool":
        config = run_config(
            load_config(args.config),
            args.output_root,
            args.artifacts_root,
            args.run_id,
            args.previous_datasets,
        )
        pool = initialize(config)
        result = {
            "stage": "prepare-pool",
            "pool_id": pool["pool_id"],
            "statistics": pool["statistics"],
            "split_counts": pool["split_counts"],
        }
    else:
        config = load_json(args.config)
        if args.stage == "prepare":
            result = prepare(config)
        elif args.stage == "annotate":
            result = annotate(config, args.limit)
        else:
            result = finalize(config)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
