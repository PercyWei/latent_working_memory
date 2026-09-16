"""Opt-in numeric comparison: GMSA_UPSTREAM=/path/to/pinned/checkout pytest this file."""

import importlib.util
import os
from pathlib import Path
import sys

from peft import LoraConfig
import pytest
import torch


@pytest.mark.skipif(
    not os.environ.get("GMSA_UPSTREAM"), reason="requires pinned GMSA source checkout"
)
def test_static_single_sample_matches_upstream(model, tiny_base):
    path = Path(os.environ["GMSA_UPSTREAM"]) / "modeling_gmsa.py"
    spec = importlib.util.spec_from_file_location("gmsa_reference", path)
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    try:
        spec.loader.exec_module(upstream)
        reference = (
            upstream.GMSA(
                upstream.ModelArguments(
                    model_name_or_path=str(tiny_base),
                    encoder_layers=2,
                    num_mem_fusion_layers=1,
                    merge_size=2,
                    is_random=False,
                    train=False,
                    lora_r=2,
                    lora_alpha=4,
                    lora_dropout=0,
                ),
                upstream.TrainingArguments(output_dir="unused", use_cpu=True, bf16=False),
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    lora_dropout=0,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules="all-linear",
                ),
            )
            .float()
            .eval()
        )
        reference.encoder.load_state_dict(model.encoder.state_dict(), strict=True)
        reference.decoder.load_state_dict(model.decoder.state_dict(), strict=True)
        alignment_state = reference.memory_fusion_layer.model.state_dict()
        alignment_state.update(model.alignment.state_dict())
        reference.memory_fusion_layer.model.load_state_dict(alignment_state, strict=True)
        model.eval()
        for ratio in (2, 4):
            reference.merge_size = ratio
            for length in (3, 4, 5):
                context = torch.arange(4, 4 + length)[None]
                prompt = torch.tensor([[10, 11]])
                ids = torch.cat((context, prompt), 1)
                prompt_mask = torch.cat((torch.zeros_like(context), torch.ones_like(prompt)), 1)
                labels = torch.tensor([[4, 5, 2]])
                with torch.no_grad():
                    original = reference(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids, dtype=torch.bool),
                        prompt_mask=prompt_mask,
                        labels=labels,
                    )
                    migrated = model(
                        context,
                        torch.ones_like(context, dtype=torch.bool),
                        prompt,
                        torch.ones_like(prompt, dtype=torch.bool),
                        labels,
                        ratio,
                    )
                torch.testing.assert_close(original.logits, migrated.logits, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(original.loss, migrated.loss, rtol=1e-5, atol=1e-6)
    finally:
        sys.modules.pop(spec.name, None)
