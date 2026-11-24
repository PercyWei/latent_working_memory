"""ICAE v2 LoCoMo 评估入口。"""

import argparse
from pathlib import Path

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from .locomo_adapter import add_arguments, configuration, load_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/icae")
    add_arguments(parser)
    args = parser.parse_args(argv)
    return run_evaluation(args, "icae", configuration(args), load_adapter)


if __name__ == "__main__":
    main()
