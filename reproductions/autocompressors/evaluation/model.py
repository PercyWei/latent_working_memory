"""加载官方 Llama AutoCompressor 及其 tokenizer。"""

from auto_compressor import LlamaAutoCompressorModel
import torch
from transformers import AutoTokenizer

from .locomo_adapter import AutoCompressorLoCoMo


def load_adapter(settings):
    if not torch.cuda.is_available():
        raise RuntimeError("AutoCompressors LoCoMo requires CUDA")
    checkpoint = settings["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model, loading = LlamaAutoCompressorModel.from_pretrained(
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
    return AutoCompressorLoCoMo(model, tokenizer, settings["segment_length"], settings["input_format"])
