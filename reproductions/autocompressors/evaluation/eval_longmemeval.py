"""使用 AutoCompressors 压缩完整会话历史，生成 LongMemEval 回答并自动评分。"""

import argparse
from pathlib import Path

from benchmarks.longmemeval.evaluate import add_evaluation_arguments, run_evaluation
from .model import add_arguments, configuration, load_longmemeval_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/autocompressors")
    add_arguments(parser, 2304)
    args = parser.parse_args(argv)
    settings = {**configuration(parser, args), "fit_segments": True,
                "attention_implementation": "flash_attention_2"}
    return run_evaluation(args, "autocompressors", settings, load_longmemeval_adapter,
                          output_name=f"{Path(settings['checkpoint']).name}_seg{args.segment_length}")


if __name__ == "__main__":
    main()
