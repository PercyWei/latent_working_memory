"""使用 LaCT 评估 LongMemEval-S 与 Oracle，生成回答后自动评分。"""

import argparse
from pathlib import Path

from benchmarks.longmemeval.evaluate import add_evaluation_arguments, run_evaluation
from .eval_locomo import parse_checkpoint
from .locomo_adapter import load_longmemeval_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/lact")
    args, settings = parse_checkpoint(parser, argv)
    return run_evaluation(args, "lact", settings, load_longmemeval_adapter,
                          output_name=args.checkpoint.name)


if __name__ == "__main__":
    main()
