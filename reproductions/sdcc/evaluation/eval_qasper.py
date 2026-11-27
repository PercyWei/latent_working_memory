"""使用 SDCC 的半动态压缩权重评估 QASPER。"""

import argparse
from pathlib import Path

from benchmarks.qasper.evaluate import add_evaluation_arguments, run_evaluation
from .model import add_arguments, configuration, load_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/sdcc", checkpoint_layout=True)
    add_arguments(parser)
    args = parser.parse_args(argv)
    settings = configuration(parser, args)
    scale_label = str(args.compress_ratio_scale).removesuffix(".0")
    return run_evaluation(args, "sdcc", settings, load_adapter,
                          checkpoint_name=Path(settings["checkpoint"]).name,
                          output_name=f"scale{scale_label}")


if __name__ == "__main__":
    main()
