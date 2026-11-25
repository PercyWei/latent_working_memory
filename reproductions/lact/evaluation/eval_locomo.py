"""LaCT 的 LoCoMo 独立评估入口。"""

import argparse
from pathlib import Path

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from benchmarks.locomo.records import file_identity
from reproductions.ttt_locomo import INPUT_FORMAT
from .locomo_adapter import load_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/lact")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    settings = {
        "checkpoint": str(checkpoint), "checkpoint_files": file_identity(checkpoint),
        "inference_mode": "optimized",
        "parameter_dtype": "bfloat16", "mixed_precision": False,
        "input_format": dict(INPUT_FORMAT),
        "continuation": "context prefill followed by tokenwise cached question and answer",
    }
    return run_evaluation(args, "lact", settings, load_adapter,
                          output_name=args.checkpoint.name)


if __name__ == "__main__":
    main()
