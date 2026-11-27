"""使用完整上下文评估 QASPER，生成回答后计算官方 Answer F1。"""

import argparse
from pathlib import Path

from benchmarks.qasper.evaluate import add_evaluation_arguments, run_evaluation
from .locomo import load_adapter
from .model import parse_chat_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/baselines/full_context", checkpoint_layout=True)
    args, settings = parse_chat_evaluation(parser, argv)
    model_name = Path(settings["model"]["model_name_or_path"]).name
    return run_evaluation(args, "full-context", settings, load_adapter,
                          output_name="", checkpoint_name=model_name)


if __name__ == "__main__":
    main()
