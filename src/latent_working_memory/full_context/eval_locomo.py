"""Full-context 接入公共 LoCoMo 评估。"""

import argparse
from pathlib import Path

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from .locomo import load_adapter
from .model import parse_chat_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/baselines/full_context")
    args, settings = parse_chat_evaluation(parser, argv)
    model_name = Path(settings["model"]["model_name_or_path"]).name
    return run_evaluation(args, "full-context", settings, load_adapter, output_name=model_name)


if __name__ == "__main__":
    main()
