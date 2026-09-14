from __future__ import annotations

import torch
from latent_working_memory.v1.pretrain.sampling import PretrainExample, read_tokens
from latent_working_memory.v1.pretrain.training import PretrainTrainer, pretrain_forward


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
        torch.stack([r.mean_nll for r in full.ae if r is not None]).sum() + full.lm[0].mean_nll
    )
    torch.testing.assert_close(full.loss, expected / 3)
    full.loss.backward()
    expected_gradients = {
        name: p.grad.clone() for name, p in writer.named_parameters() if p.grad is not None
    }
    backbone.zero_grad(set_to_none=True)
    writer.zero_grad(set_to_none=True)
    for batch in (examples[:1], examples[1:]):
        pretrain_forward(tiny_config, backbone, writer, batch, 3).loss.backward()
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


def test_capacity_mean_weights_samples_equally_across_microbatches(
    tiny_config, tokenizer, source_records, components, semantic_examples
):
    episodes = semantic_examples(source_records[0], tokenizer, tiny_config)
    ae_episode = next(e for e in episodes if e.reads[0].task == "ae")
    lm_episode = next(e for e in episodes if e.reads[0].task == "continuation")
    ae, _ = read_tokens(ae_episode, tokenizer)
    _, lm = read_tokens(lm_episode, tokenizer)
    examples = [
        PretrainExample(ae_episode, ae, None, 2, 0.25),
        PretrainExample(ae_episode, ae, None, 4, 0.75),
        PretrainExample(lm_episode, None, lm, 4),
    ]
    backbone, writer = components
    full = pretrain_forward(tiny_config, backbone, writer, examples)
    expected = (0.25 * full.ae[0].mean_nll + 0.75 * full.ae[1].mean_nll + full.lm[2].mean_nll) / 2
    torch.testing.assert_close(full.loss, expected)
    full.loss.backward()
    gradients = {n: p.grad.clone() for n, p in writer.named_parameters() if p.grad is not None}
    backbone.zero_grad(set_to_none=True)
    writer.zero_grad(set_to_none=True)
    for example in examples:
        pretrain_forward(tiny_config, backbone, writer, [example], 2).loss.backward()
    for name, parameter in writer.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-6, rtol=1e-4)
