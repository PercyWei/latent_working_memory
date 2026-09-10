from __future__ import annotations

import torch
from latent_working_memory.v1.sampling import PretrainExample, read_tokens
from latent_working_memory.v1.training import PretrainTrainer, pretrain_forward


def test_mixed_ae_only_objective_and_microbatch_gradients(
    tmp_path, tiny_config, tokenizer, source_records, components, semantic_examples
):
    episodes = semantic_examples(source_records[0], tokenizer, tiny_config)
    continuation = next(e for e in episodes if e.reads[0].task == "continuation")
    only = next(e for e in episodes if e.reads[0].task == "ae")
    ae, _ = read_tokens(only, tokenizer)
    _, lm = read_tokens(continuation, tokenizer)
    examples = [
        PretrainExample(continuation, None, lm, 4),
        PretrainExample(only, ae, None, 4),
        PretrainExample(only, ae, None, 4),
    ]
    backbone, writer = components
    full = pretrain_forward(tiny_config, backbone, writer, examples)
    assert full.lm[1:] == [None, None]
    expected = (
        tiny_config.ae_weight * torch.stack([r.mean_nll for r in full.ae if r is not None]).mean()
        + tiny_config.lm_weight * full.lm[0].mean_nll
    )
    torch.testing.assert_close(full.loss, expected)
    full.loss.backward()
    expected_gradients = {
        name: p.grad.clone() for name, p in writer.named_parameters() if p.grad is not None
    }
    backbone.zero_grad(set_to_none=True)
    writer.zero_grad(set_to_none=True)
    for batch in (examples[:1], examples[1:]):
        pretrain_forward(tiny_config, backbone, writer, batch, (2, 1)).loss.backward()
    for name, parameter in writer.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(
                parameter.grad, expected_gradients[name], atol=1e-6, rtol=1e-4
            )
    trainer = PretrainTrainer(tiny_config, backbone, writer, torch.device("cpu"))
    result = trainer.step(examples[1:])
    assert all(
        row["lm_nll"] is None and row["continuation_tokens"] == 0 for row in result["samples"]
    )
    assert result["target_tokens"] == 2 * len(ae.target_ids)
