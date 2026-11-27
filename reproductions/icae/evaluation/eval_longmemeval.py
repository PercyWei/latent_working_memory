"""使用 ICAE v2 的预训练或微调权重生成 LongMemEval 回答并自动评分。"""

import argparse
from pathlib import Path

from benchmarks.longmemeval.evaluate import add_evaluation_arguments, run_evaluation
from .locomo_adapter import add_arguments, configuration, load_longmemeval_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/icae", checkpoint_layout=True)
    add_arguments(parser)
    args = parser.parse_args(argv)
    settings = {**configuration(args), "dtype": "bfloat16", "attention_implementation": "flash_attention_2"}
    return run_evaluation(args, "icae", settings, load_longmemeval_adapter,
                          output_name=f"mem{args.fixed_mem_size}_cr{args.mean_compression_rate}",
                          checkpoint_name=Path(settings["checkpoint"]).stem)


if __name__ == "__main__":
    main()
