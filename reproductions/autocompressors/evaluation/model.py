"""加载官方 Llama AutoCompressor 及其 tokenizer。"""

from pathlib import Path

from .reader import BatchedLlamaAutoCompressor
import torch
from transformers import AutoTokenizer

from benchmarks.locomo.records import file_identity
from latent_working_memory.prompts import QA_SYSTEM_PROMPT
from .locomo_adapter import AutoCompressorLoCoMo, INPUT_FORMAT


def add_arguments(parser, segment_length):
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="AutoCompressor-Llama-2-7b-6k 的完整本地 checkpoint 目录")
    parser.add_argument("--segment-length", type=int, default=segment_length,
                        help="每段正文的 token 数上限，默认 %(default)s")


def configuration(parser, args):
    if args.segment_length <= 0:
        parser.error("--segment-length must be positive")
    source = Path(__file__).resolve().parents[1] / "source"
    revision = (source / "revision").read_text().strip()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    return {
        "checkpoint": str(checkpoint), "checkpoint_files": file_identity(checkpoint),
        "upstream_revision": revision,
        "segment_length": args.segment_length,
        "input_format": dict(INPUT_FORMAT),
        "dtype": "bfloat16",
    }


def load_adapter(settings):
    if not torch.cuda.is_available():
        raise RuntimeError("AutoCompressors evaluation requires CUDA")
    checkpoint = settings["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model, loading = BatchedLlamaAutoCompressor.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, output_loading_info=True,
    )
    failures = {key: loading[key] for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
    ) if loading[key]}
    if failures:
        raise ValueError(f"incompatible AutoCompressor state dict: {failures}")
    if not model.config.accumulate_summary or model.config.summary_length != 50:
        raise ValueError("expected 50 accumulated summary vectors per segment")
    model.requires_grad_(False).eval().cuda()
    return AutoCompressorLoCoMo(model, tokenizer, settings["segment_length"], settings["input_format"],
                                fit_segments=settings.get("fit_segments", False),
                                system_prompt=settings.get("system_prompt", QA_SYSTEM_PROMPT))


def load_longmemeval_adapter(settings):
    return load_adapter(settings)
