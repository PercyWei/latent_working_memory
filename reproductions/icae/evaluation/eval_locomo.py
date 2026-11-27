"""ICAE v2 LoCoMo 评估入口。"""

import argparse
from pathlib import Path

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from .locomo_adapter import add_arguments, configuration, load_adapter, summarize_metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/icae", checkpoint_layout=True)
    add_arguments(parser)
    args = parser.parse_args(argv)
    settings = configuration(args)
    return run_evaluation(args, "icae", settings, load_adapter,
                          output_name=f"mem{args.fixed_mem_size}_cr{args.mean_compression_rate}",
                          checkpoint_name=Path(settings["checkpoint"]).stem,
                          summarize_method=summarize_metrics)


if __name__ == "__main__":
    main()
