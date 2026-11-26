"""使用完整上下文评估 LongMemEval-S 与 Oracle，生成回答后自动评分。"""

import argparse
from pathlib import Path

from benchmarks.longmemeval.evaluate import add_evaluation_arguments, run_evaluation
from .longmemeval import PREFILL_CHUNK_SIZE, load_adapter
from .model import parse_chat_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/baselines/full_context")
    args, settings = parse_chat_evaluation(parser, argv)
    settings["prefill_chunk_size"] = PREFILL_CHUNK_SIZE
    model_name = Path(settings["model"]["model_name_or_path"]).name
    return run_evaluation(args, "full-context", settings, load_adapter, output_name=model_name)


if __name__ == "__main__":
    main()
