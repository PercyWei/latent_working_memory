"""AutoCompressors 的 LoCoMo 独立评估入口。"""

import argparse
from pathlib import Path

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from .locomo_adapter import summarize_metrics
from .model import add_arguments, configuration, load_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/autocompressors", checkpoint_layout=True)
    add_arguments(parser, 2048)
    args = parser.parse_args(argv)
    settings = configuration(parser, args)
    return run_evaluation(args, "autocompressors", settings, load_adapter,
                          checkpoint_name=Path(settings["checkpoint"]).name,
                          output_name=f"seg{args.segment_length}",
                          summarize_method=summarize_metrics)


if __name__ == "__main__":
    main()
