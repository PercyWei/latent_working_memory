"""使用 AutoCompressors 压缩 MuSiQue-Ans 候选段落并评估回答。"""

import argparse
from pathlib import Path

from benchmarks.musique.evaluate import add_evaluation_arguments, run_evaluation
from .model import add_arguments, configuration, load_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/autocompressors", checkpoint_layout=True)
    add_arguments(parser, 2048)
    args = parser.parse_args(argv)
    settings = {**configuration(parser, args), "fit_segments": True,
                "attention_implementation": "flash_attention_2"}
    settings["input_format"]["answer_prefix"] = "\nAnswer:"
    return run_evaluation(args, "autocompressors", settings, load_adapter,
                          checkpoint_name=Path(settings["checkpoint"]).name,
                          output_name=f"seg{args.segment_length}")


if __name__ == "__main__":
    main()
