"""In-Place TTT LoCoMo 的完整 attention 与快速权重适配。"""

import torch
from transformers import AutoTokenizer

from benchmarks.locomo.causal import CausalLoCoMo
from reproductions.in_place_ttt.optimized.model import OptimizedQwen3ForCausalLM
from reproductions.ttt_longmemeval import TTTLongMemEval
from .locomo_state import fork_cache


def load_model(settings):
    if not torch.cuda.is_available():
        raise RuntimeError("In-Place TTT inference requires CUDA")
    checkpoint = settings["checkpoint"]
    model, loading = OptimizedQwen3ForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, attn_implementation=settings["attention_implementation"],
        local_files_only=True, output_loading_info=True,
    )
    failures = {key: loading[key] for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
    ) if loading[key]}
    if failures:
        raise ValueError(f"Checkpoint did not load strictly: {failures}")
    model.requires_grad_(False).eval().to("cuda")
    model.set_optimization(True, settings["mlp_chunk_size"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    return model, tokenizer


def load_adapter(settings):
    model, tokenizer = load_model(settings)
    return CausalLoCoMo(model, tokenizer, fork_cache, settings["input_format"],
                       max_positions=model.config.max_position_embeddings)


def load_longmemeval_adapter(settings):
    model, tokenizer = load_model(settings)
    return TTTLongMemEval(model, tokenizer, settings, max_positions=settings["max_seq_len"])
