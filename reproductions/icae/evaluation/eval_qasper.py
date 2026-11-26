"""使用 ICAE v2 压缩 QASPER 论文并评估回答。"""

import argparse
from pathlib import Path

from benchmarks.qasper.data import SYSTEM_PROMPT
from benchmarks.qasper.evaluate import add_evaluation_arguments, run_evaluation
from .locomo_adapter import add_arguments, configuration, load_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/icae")
    add_arguments(parser)
    args = parser.parse_args(argv)
    settings = {
        **configuration(args, SYSTEM_PROMPT),
        "dtype": "bfloat16", "attention_implementation": "flash_attention_2",
    }
    return run_evaluation(args, "icae", settings, load_adapter,
                          output_name=(f"{args.checkpoint.stem}_mem{args.fixed_mem_size}"
                                       f"_cr{args.mean_compression_rate}"))


if __name__ == "__main__":
    main()
