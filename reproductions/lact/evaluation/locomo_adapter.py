"""LaCT LoCoMo 的模型加载与会话状态适配。"""

from lact_model import LaCTSWIGLUConfig
import torch
from transformers import AutoTokenizer

from benchmarks.locomo.causal import CausalLoCoMo
from reproductions.lact.optimized.model import OptimizedLaCTForCausalLM
from .locomo_state import fork_cache


def load_adapter(settings):
    if not torch.cuda.is_available():
        raise RuntimeError("LaCT LoCoMo requires CUDA")
    checkpoint = settings["checkpoint"]
    config = LaCTSWIGLUConfig.from_pretrained(checkpoint, local_files_only=True)
    model, loading = OptimizedLaCTForCausalLM.from_pretrained(
        checkpoint, config=config, torch_dtype=torch.float32,
        local_files_only=True, output_loading_info=True,
    )
    failures = {key: loading[key] for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
    ) if loading[key]}
    if failures:
        raise ValueError(f"Checkpoint did not load strictly: {failures}")
    model.requires_grad_(False).eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    return CausalLoCoMo(model, tokenizer, fork_cache, settings["input_format"], mixed_precision=True)
