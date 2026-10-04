import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from latent_working_memory.v4.config import ModelConfig
from latent_working_memory.v4.model import StreamingMemoryLM


@pytest.fixture
def model_factory():
    def build(model_type="llama", **overrides):
        torch.manual_seed(73)
        config_class, model_class = {
            "llama": (LlamaConfig, LlamaForCausalLM),
            "qwen2": (Qwen2Config, Qwen2ForCausalLM),
            "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        }[model_type]
        base_config = config_class(
            vocab_size=41,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=128,
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        base_config._attn_implementation = "eager"
        backbone = model_class(base_config)
        settings = {
            "model_name_or_path": "local-tiny-model",
            "pending_size": 3,
            "recent_size": 2,
            "num_slots": 3,
            "memory_dim": 8,
            "query_dim": 4,
            "value_dim": 6,
            "num_probes": 4,
            "inner_steps": 2,
            "inner_lr": 0.4,
            "correction_scale": 0.2,
        }
        settings.update(overrides)
        return StreamingMemoryLM(ModelConfig(**settings), backbone)

    return build
