"""AutoCompressors 的 LoCoMo 独立评估入口。"""

import argparse
from pathlib import Path

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from benchmarks.locomo.records import file_identity
from .locomo_adapter import INPUT_FORMAT, summarize_metrics
from .model import load_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/autocompressors")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="AutoCompressor-Llama-2-7b-6k 的完整本地 checkpoint 目录")
    parser.add_argument("--segment-length", type=int, default=2048,
                        help="每段正文的 token 数上限，默认 2048")
    args = parser.parse_args(argv)
    if args.segment_length <= 0:
        parser.error("--segment-length must be positive")
    source = Path(__file__).resolve().parents[1] / "source"
    revision = (source / "revision").read_text().strip()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    settings = {
        "checkpoint": str(checkpoint), "checkpoint_files": file_identity(checkpoint),
        "upstream_revision": revision,
        "segment_length": args.segment_length,
        "input_format": dict(INPUT_FORMAT),
        "dtype": "bfloat16",
    }
    return run_evaluation(args, "autocompressors", settings, load_adapter,
                          output_name=f"{checkpoint.name}_seg{args.segment_length}",
                          summarize_method=summarize_metrics)


if __name__ == "__main__":
    main()
