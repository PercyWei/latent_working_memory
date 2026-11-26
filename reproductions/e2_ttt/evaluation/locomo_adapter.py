"""E²-TTT LoCoMo 的 MLP／SwiGLU 模型与递归状态适配。"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.locomo.causal import CausalLoCoMo
from reproductions.e2_ttt.optimized.model import optimize_model
from .locomo_state import fork_cache


def load_model(settings):
    if not torch.cuda.is_available():
        raise RuntimeError("E²-TTT inference requires CUDA")
    checkpoint = settings["checkpoint"]
    # optimized.model 导入官方 e2_ttt 包，注册其 AutoModel 类型。
    model, loading = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16,
        local_files_only=True, output_loading_info=True,
    )
    failures = {key: loading[key] for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
    ) if loading[key]}
    if failures:
        raise ValueError(f"Checkpoint did not load strictly: {failures}")
    model.requires_grad_(False).eval().to("cuda")
    optimize_model(model)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    return model, tokenizer


def load_adapter(settings):
    model, tokenizer = load_model(settings)
    return CausalLoCoMo(model, tokenizer, fork_cache, settings["input_format"])


def load_longmemeval_adapter(settings):
    return load_adapter(settings)
