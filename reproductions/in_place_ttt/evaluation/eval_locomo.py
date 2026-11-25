"""In-Place TTT 的 LoCoMo 独立评估入口。"""

import argparse
from pathlib import Path

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from benchmarks.locomo.records import file_identity
from reproductions.ttt_locomo import INPUT_FORMAT
from .locomo_adapter import load_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/in_place_ttt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mlp-chunk-size", type=int, default=0, help="MLP 分块长度，默认 0")
    args = parser.parse_args(argv)
    if args.mlp_chunk_size < 0:
        parser.error("--mlp-chunk-size must be nonnegative")
    checkpoint = args.checkpoint.expanduser().resolve()
    settings = {
        "checkpoint": str(checkpoint), "checkpoint_files": file_identity(checkpoint),
        "inference_mode": "optimized",
        "attention_implementation": "flash_attention_2",
        "input_format": dict(INPUT_FORMAT),
        "continuation": "context prefill followed by tokenwise cached question and answer",
        "mlp_chunk_size": args.mlp_chunk_size,
    }
    return run_evaluation(args, "in-place-ttt", settings, load_adapter,
                          output_name=args.checkpoint.name)


if __name__ == "__main__":
    main()
