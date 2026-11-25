"""小型原生 Qwen3 TTT 对照；无需下载权重或运行完整 benchmark。"""

import os

from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
import torch

from benchmarks.locomo.causal import CausalLoCoMo
from reproductions.ttt_locomo import INPUT_FORMAT
from reproductions.in_place_ttt.optimized.model import OptimizedQwen3ForCausalLM
from reproductions.locomo_validation import tokenizer_for_validation, validate_context
from .locomo_state import fork_cache, state_tensors


def main():
    torch.set_num_threads(1)
    torch.manual_seed(24)
    device = torch.device(os.environ.get("LOCOMO_TEST_DEVICE", "cpu"))
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=2048, eos_token_id=1, pad_token_id=0,
        ttt_mode=True, ttt_layers=[0, 1], ttt_chunk=8, attn_implementation="sdpa",
    )
    reference = Qwen3ForCausalLM(config).eval().to(device)
    with torch.no_grad():
        for block in reference.model.layers:
            block.mlp.ttt_conv.weight.normal_(0, 0.1)
    model = OptimizedQwen3ForCausalLM(config).eval().to(device)
    model.load_state_dict(reference.state_dict())
    model.set_optimization(True, 8)
    adapter = CausalLoCoMo(model, tokenizer_for_validation(), fork_cache, INPUT_FORMAT,
                          max_positions=2048)
    validate_context(adapter, reference, 8, state_tensors)


if __name__ == "__main__":
    main()
