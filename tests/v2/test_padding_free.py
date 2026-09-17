"""Real CUDA varlen kernels: sequence isolation, recurrent gradients and dense generation."""

from dataclasses import replace

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from latent_working_memory.v2.memory_codec import CodecConfig, MemoryCodec
from latent_working_memory.v2.pretrain.config import TrainingConfig
from latent_working_memory.v2.pretrain.data import Trajectory
from latent_working_memory.v2.pretrain.engine import initialize_device, precision_context
from latent_working_memory.v2.pretrain.objective import ReconstructionTask


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA varlen attention")
@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
def test_padding_free_matches_dense_and_isolates_sequences(tiny_base, tmp_path, architecture):
    device = initialize_device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(tiny_base)
    path = tiny_base
    if architecture == "qwen3":
        path = tmp_path / "qwen3"
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=128,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        ).save_pretrained(path)
        tokenizer.save_pretrained(path)
    config = CodecConfig(
        str(path),
        lora_rank=2,
        lora_alpha=4,
        compression="weighted",
        attention_implementation="sdpa",
        gradient_checkpointing=True,
    )
    torch.manual_seed(42)
    dense = ReconstructionTask(
        MemoryCodec(config, torch.bfloat16),
        tokenizer,
        TrainingConfig(objective="ae_lm", lm_ratio=0.5),
    ).to(device)
    packed = ReconstructionTask(
        MemoryCodec(replace(config, padding_free=True), torch.bfloat16), tokenizer, dense.config
    ).to(device)
    packed.load_state_dict(dense.state_dict())
    for task in (dense, packed):
        task.ae_prompt = torch.tensor([4, 5], device=device)
        task.lm_prompt = torch.tensor([4, 5, 6, 7, 4], device=device)
    rows = [
        Trajectory("a", 0, torch.tensor([4, 5, 6, 7] * 8), (7, 17, 27), 3),
        Trajectory("b", 0, torch.tensor([7, 6, 5, 4] * 5), (3, 5, 8, 12), 3),
    ]
    shapes = []
    handle = packed.codec.encoder.get_base_model().model.register_forward_pre_hook(
        lambda module, args, kwargs: shapes.append(tuple(kwargs["inputs_embeds"].shape)),
        with_kwargs=True,
    )
    with precision_context(device):
        expected = dense(rows, read_task=["ae", "lm"])
        actual = packed(rows, read_task=["ae", "lm"])
    handle.remove()
    assert shapes[0][0:2] == (1, 10)  # 7+3 positions, not 2*7 padding.
    assert actual["batch_sizes"] == [2, 2, 2, 1]
    torch.testing.assert_close(
        actual["sample_losses"], expected["sample_losses"], rtol=0.02, atol=0.03
    )
    expected["loss"].backward()
    actual["loss"].backward()
    a, b = [], []
    for (name, p), (_, q) in zip(packed.named_parameters(), dense.named_parameters(), strict=True):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
            a.append(p.grad.flatten().float())
            b.append(q.grad.flatten().float())
    assert torch.nn.functional.cosine_similarity(torch.cat(a), torch.cat(b), dim=0) > 0.99
    packed.eval()
    with torch.no_grad(), precision_context(device):
        original = packed(rows, read_task=["ae", "lm"])["sample_losses"]
        changed = [replace(rows[0], token_ids=torch.full_like(rows[0].token_ids, 6)), rows[1]]
        isolated = packed(changed, read_task=["ae", "lm"])["sample_losses"]
        torch.testing.assert_close(original[1], isolated[1], rtol=0, atol=0)
        memory = packed.codec.write(None, rows[0].token_ids[:7].to(device), 3)
        generated = packed.codec.generate(
            memory, packed.ae_prompt, 3, tokenizer.eos_token_id, tokenizer.pad_token_id
        )
        assert 1 <= len(generated) <= 3
