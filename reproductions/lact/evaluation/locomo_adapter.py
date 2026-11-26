"""LaCT LoCoMo 的模型加载与会话状态适配。"""

from lact_model import LaCTSWIGLUConfig
import torch
from transformers import AutoTokenizer

from benchmarks.locomo.causal import CausalLoCoMo
from reproductions.lact.optimized.model import OptimizedLaCTForCausalLM
from reproductions.ttt_longmemeval import TTTLongMemEval
from .locomo_state import fork_cache


def load_model(settings):
    if not torch.cuda.is_available():
        raise RuntimeError("LaCT inference requires CUDA")
    checkpoint = settings["checkpoint"]
    config = LaCTSWIGLUConfig.from_pretrained(checkpoint, local_files_only=True)
    model, loading = OptimizedLaCTForCausalLM.from_pretrained(
        checkpoint, config=config, torch_dtype=getattr(torch, settings["parameter_dtype"]),
        local_files_only=True, output_loading_info=True,
    )
    failures = {key: loading[key] for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
    ) if loading[key]}
    if failures:
        raise ValueError(f"Checkpoint did not load strictly: {failures}")
    model.requires_grad_(False).eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    return model, tokenizer


def load_adapter(settings):
    model, tokenizer = load_model(settings)
    return CausalLoCoMo(model, tokenizer, fork_cache, settings["input_format"],
                       mixed_precision=settings["mixed_precision"])


def load_longmemeval_adapter(settings):
    model, tokenizer = load_model(settings)
    return TTTLongMemEval(model, tokenizer, settings)
