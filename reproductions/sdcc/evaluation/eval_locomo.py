"""使用同基座的 SDCC 半动态压缩权重评估 LoCoMo。"""

import argparse
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from benchmarks.locomo.records import file_identity
from .locomo_adapter import sdcc_input_format, summarize_metrics
from .model import load_adapter, lora_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    add_evaluation_arguments(parser, root / "artifacts/reproductions/sdcc")
    parser.add_argument("--model-path", type=Path, required=True, help="本地基座模型目录")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SDCC 半动态压缩权重目录")
    parser.add_argument("--compress-ratio-scale", type=float, default=0.0,
                        help="加在选择器预测值上的偏移量，默认 0，增大时倾向使用更少的记忆")
    args = parser.parse_args(argv)
    if not math.isfinite(args.compress_ratio_scale):
        parser.error("--compress-ratio-scale must be finite")
    base = args.model_path.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    if config["base_encoder_model_path"] != config["base_decoder_model_path"]:
        parser.error("--checkpoint must declare the same encoder and decoder base for --model-path")
    if (config["feature_extract_method"] != "mean_pooling"
            or not isinstance(config["comp_ratio_or_len"], list) or "discretize_ratio_mode" not in config):
        parser.error("--checkpoint must contain the semi-dynamic mean-pooling model")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    checkpoint_files = {}
    for path in (checkpoint / "config.json", checkpoint / "projector.pth",
                 lora_directory(checkpoint / "encoder"), lora_directory(checkpoint / "decoder")):
        checkpoint_files.update(file_identity(path))
    revision = (root / "reproductions/sdcc/source/revision").read_text().strip()
    settings = {
        "model_path": str(base), "model_files": file_identity(base),
        "checkpoint": str(checkpoint), "checkpoint_files": checkpoint_files,
        "upstream_revision": revision, "compress_ratio_scale": args.compress_ratio_scale,
        "retention_options": config["comp_ratio_or_len"],
        "input_format": sdcc_input_format(tokenizer, config["placeholder_token_id"]),
        "dtype": "bfloat16", "attention_implementation": "flash_attention_2",
    }
    scale_label = str(args.compress_ratio_scale).removesuffix(".0")
    return run_evaluation(args, "sdcc", settings, load_adapter,
                          output_name=f"{checkpoint.name}_scale{scale_label}",
                          summarize_method=summarize_metrics)


if __name__ == "__main__":
    main()
