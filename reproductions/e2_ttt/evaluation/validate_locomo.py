"""小型 E²-TTT MLP／SwiGLU 对照原生 B1 缓存续写。"""

from copy import deepcopy

from e2_ttt import E2TTTMLPConfig, E2TTTMLPForCausalLM, E2TTTSwiGLUConfig, E2TTTSwiGLUForCausalLM
import torch

from benchmarks.locomo.causal import CausalLoCoMo
from reproductions.ttt_locomo import INPUT_FORMAT
from reproductions.e2_ttt.optimized.model import optimize_model
from reproductions.locomo_validation import tokenizer_for_validation, validate_context
from .locomo_state import fork_cache, state_tensors


def main():
    torch.manual_seed(24)
    for config_class, model_class in (
        (E2TTTMLPConfig, E2TTTMLPForCausalLM), (E2TTTSwiGLUConfig, E2TTTSwiGLUForCausalLM),
    ):
        config = config_class(
            vocab_size=64, hidden_size=64, num_heads=4, num_hidden_layers=2,
            intermediate_size=128, chunk_size=16, ttt_chunk_size=8,
            max_position_embeddings=512, eos_token_id=1, pad_token_id=0,
            fuse_norm=False, fuse_swiglu=False, fuse_cross_entropy=False,
        )
        reference = model_class(config).eval().to(device="cuda", dtype=torch.bfloat16)
        model = optimize_model(deepcopy(reference))
        adapter = CausalLoCoMo(model, tokenizer_for_validation(), fork_cache, INPUT_FORMAT)
        validate_context(adapter, reference, 8, state_tensors)


if __name__ == "__main__":
    main()
