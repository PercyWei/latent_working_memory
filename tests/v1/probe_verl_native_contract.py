"""CPU probes of verl's native LM input and microbatch contracts; no FSDP training."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from transformers import Qwen2Config, Qwen2ForCausalLM
from verl.utils.tensordict_utils import assign_non_tensor
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine.utils import prepare_micro_batches
from verl.workers.utils.losses import sft_loss


def token_batch():
    rows = [torch.arange(3, 3 + length) for length in (4, 8, 6, 10)]
    batch = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(rows, layout=torch.jagged),
            "position_ids": torch.nested.as_nested_tensor(
                [torch.arange(len(row)) for row in rows], layout=torch.jagged
            ),
            "sample_id": torch.arange(len(rows)),
        },
        batch_size=[len(rows)],
    )
    assign_non_tensor(batch, temperature=1.0, use_fused_kernels=False)
    return batch


def probe_inputs():
    # Exercise only the real input adapter. Model/FSDP initialization needs CUDA
    # and is deliberately outside this CPU contract probe.
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    data = token_batch()
    memory = torch.randn(4, 2, 16, requires_grad=True)
    data["memory"] = memory
    data["inputs_embeds"] = torch.randn(4, 10, 16, requires_grad=True)
    cases = []
    for remove_padding in (False, True):
        assign_non_tensor(data, use_remove_padding=remove_padding)
        model_inputs, _ = engine.prepare_model_inputs(data)
        assert "memory" not in model_inputs and "inputs_embeds" not in model_inputs
        cases.append({"remove_padding": remove_padding, "model_input_keys": sorted(model_inputs)})

    # HF Qwen itself accepts continuous inputs and differentiates through them.
    # The mismatch above is in the native engine adapter, not in Qwen.
    model = (
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=32,
                attention_dropout=0.0,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    ids = torch.tensor([[1, 4, 5]])
    embeddings = model.get_input_embeddings()(ids)
    continuous_input = torch.cat((embeddings[:, :1], memory[:1], embeddings[:, 1:]), dim=1)
    logits = model(inputs_embeds=continuous_input, use_cache=False).logits
    loss = torch.nn.functional.cross_entropy(logits[:, -1].float(), torch.tensor([6]))
    loss.backward()
    grad_norm = memory.grad.norm().item()
    assert grad_norm > 0
    return {"native_input_cases": cases, "hf_continuous_memory_gradient_norm": grad_norm}


def probe_microbatches():
    data = token_batch()
    assign_non_tensor(data, use_dynamic_bsz=True, max_token_len_per_gpu=14)
    batches, indices = prepare_micro_batches(data, same_micro_num_in_dp=False)
    ids = [int(i) for batch in batches for i in batch["sample_id"]]
    assert sorted(ids) == list(range(4))
    selected = [batch["sample_id"].tolist() for batch in batches]
    assert selected == indices
    return {
        "input_lengths": [4, 8, 6, 10],
        "max_token_len_per_gpu": 14,
        "sample_indices": indices,
        "batch_token_counts": [int(batch["input_ids"].offsets().diff().sum()) for batch in batches],
    }


def probe_loss_weighting():
    # Two targets versus four targets, including EOS in each target count.
    data = TensorDict(
        {
            "loss_mask": torch.nested.as_nested_tensor(
                [torch.tensor([0.0, 1.0, 1.0]), torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])],
                layout=torch.jagged,
            )
        },
        batch_size=[2],
    )
    assign_non_tensor(data, dp_size=1, batch_num_tokens=6)
    outputs = {
        "log_probs": torch.nested.as_nested_tensor(
            [torch.full((3,), -1.0), torch.full((5,), -3.0)], layout=torch.jagged
        )
    }
    native, _ = sft_loss(config=None, model_output=outputs, data=data)
    sample_mean = (1.0 + 3.0) / 2
    torch.testing.assert_close(native, torch.tensor(14.0 / 6))
    assert native.item() != sample_mean
    return {"verl_default_sft_loss": native.item(), "v1_equal_sample_mean": sample_mean}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    result = {
        "scope": "CPU input preparation and token batching; no distributed execution or speed measurement",
        "software": {name: version(name) for name in ("verl", "torch", "transformers")},
        "inputs": probe_inputs(),
        "microbatches": probe_microbatches(),
        "loss_weighting": probe_loss_weighting(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
