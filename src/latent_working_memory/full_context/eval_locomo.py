"""Full-context 接入公共 LoCoMo 评估。"""

import argparse
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from benchmarks.locomo.records import file_identity
from latent_working_memory.prompts import qa_input_format
from .locomo import load_adapter
from .model import ATTENTION_IMPLEMENTATIONS, WEIGHT_DTYPES, validate_compute


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/baselines/full_context")
    parser.add_argument("--model-path", type=Path, default=Path("~/models/Qwen3-4B-Instruct-2507"),
                        help="本地基座模型目录，默认 ~/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dtype", choices=tuple(WEIGHT_DTYPES), default="fp32",
                        help="基座权重精度，默认 fp32")
    parser.add_argument("--attention-implementation", choices=ATTENTION_IMPLEMENTATIONS, default="sdpa")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="为 FP32 权重启用 PyTorch BF16 autocast，默认关闭")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    model_config = {
        "model_name_or_path": str(args.model_path.expanduser().resolve()),
        "backbone_dtype": WEIGHT_DTYPES[args.dtype],
        "attention_implementation": args.attention_implementation,
        "mixed_precision": args.mixed_precision,
    }
    validate_compute(model_config)
    model_path = Path(model_config["model_name_or_path"])
    settings = {
        "model": model_config,
        "model_files": file_identity(model_path),
        "model_window": AutoConfig.from_pretrained(model_path).max_position_embeddings,
        "input_format": qa_input_format(AutoTokenizer.from_pretrained(model_path)),
        "seed": args.seed,
    }
    return run_evaluation(args, "full-context", settings, load_adapter, output_name=model_path.name)


if __name__ == "__main__":
    main()
