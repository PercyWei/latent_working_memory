"""小型 LaCT 对照全前缀重算，覆盖完整块及尾块的状态分支。"""

from lact_model import LaCTSWIGLUConfig
import torch

from benchmarks.locomo.causal import CausalLoCoMo
from reproductions.ttt_locomo import INPUT_FORMAT
from reproductions.lact.optimized.model import OptimizedLaCTForCausalLM
from reproductions.locomo_validation import tokenizer_for_validation, validate_context
from .locomo_state import fork_cache, state_tensors


def main():
    torch.manual_seed(24)
    config = LaCTSWIGLUConfig(
        vocab_size=64, hidden_size=64, num_hidden_layers=2, num_attn_heads=4,
        num_lact_heads=2, lact_chunk_size=8, window_size=16,
        max_position_embeddings=512, intermediate_size=128,
        eos_token_id=1, pad_token_id=0, use_fused_kernel=False,
        fuse_norm=False, last_layer_fuse_norm=False, fuse_swiglu=False,
        fuse_cross_entropy=False, fp32_states=True,
    )
    model = OptimizedLaCTForCausalLM(config).eval().to("cuda")
    adapter = CausalLoCoMo(model, tokenizer_for_validation(), fork_cache, INPUT_FORMAT,
                          mixed_precision=True)
    validate_context(adapter, model, 8, state_tensors, recompute=True)


if __name__ == "__main__":
    main()
