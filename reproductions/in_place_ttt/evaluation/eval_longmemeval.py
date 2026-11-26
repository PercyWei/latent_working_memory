"""使用 In-Place TTT 评估 LongMemEval-S 与 Oracle，生成回答后自动评分。"""

import argparse
import json
from pathlib import Path

from benchmarks.longmemeval.evaluate import add_evaluation_arguments, run_evaluation
from .eval_locomo import parse_checkpoint
from .locomo_adapter import load_longmemeval_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/in_place_ttt")
    parser.add_argument("--max-seq-len", type=int,
                        help="输入和生成的总位置预算，默认取 checkpoint 配置；增大时使用原 RoPE 外推")
    args, settings = parse_checkpoint(parser, argv)
    if args.max_seq_len is not None and args.max_seq_len < 1:
        parser.error("--max-seq-len must be positive")
    config = json.loads((Path(settings["checkpoint"]) / "config.json").read_text())
    settings["checkpoint_max_position_embeddings"] = config["max_position_embeddings"]
    settings["max_seq_len"] = args.max_seq_len or config["max_position_embeddings"]
    return run_evaluation(args, "in-place-ttt", settings, load_longmemeval_adapter,
                          output_name=args.checkpoint.name)


if __name__ == "__main__":
    main()
