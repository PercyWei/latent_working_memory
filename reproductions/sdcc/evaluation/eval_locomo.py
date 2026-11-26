"""使用同基座的 SDCC 半动态压缩权重评估 LoCoMo。"""

import argparse
from pathlib import Path

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from .locomo_adapter import summarize_metrics
from .model import add_arguments, configuration, load_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/sdcc")
    add_arguments(parser)
    args = parser.parse_args(argv)
    settings = configuration(parser, args)
    scale_label = str(args.compress_ratio_scale).removesuffix(".0")
    return run_evaluation(args, "sdcc", settings, load_adapter,
                          output_name=f"{Path(settings['checkpoint']).name}_scale{scale_label}",
                          summarize_method=summarize_metrics)


if __name__ == "__main__":
    main()
