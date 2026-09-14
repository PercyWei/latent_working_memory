"""Original dense-logit reader retained only for numerical and resource comparisons."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from latent_working_memory.v1.objectives import gold_token_nll


@dataclass
class DenseReaderOutput:
    target_logits: torch.Tensor
    token_nll: torch.Tensor

    @property
    def mean_nll(self):
        return self.token_nll.mean()


def dense_read_batch(backbone, memories, tokens, text_contexts=None, use_reader_lora=True):
    if text_contexts is None:
        text_contexts = [()] * len(tokens)
    rows, contexts = [], []
    device = backbone._model_device
    embedding = backbone.language_model.get_input_embeddings()
    for memory, task, text_context in zip(memories, tokens, text_contexts, strict=True):
        ids = torch.tensor(
            (backbone.bos_token_id, *text_context, *task.prompt_ids, *task.target_ids),
            device=device,
        )
        base = embedding(ids)
        projected = backbone.memory_projection(memory.to(backbone.memory_projection.weight.dtype))
        rows.append(torch.cat((base[:1], projected.to(base.dtype), base[1:])))
        contexts.append(1 + len(memory) + len(text_context) + len(task.prompt_ids))
    inputs_embeds = pad_sequence(rows, batch_first=True)
    positions = torch.arange(inputs_embeds.shape[1], device=device)[None, :]
    mask = positions < torch.tensor([len(row) for row in rows], device=device)[:, None]
    position_ids = positions.expand_as(mask).masked_fill(~mask, 0)
    with nullcontext() if use_reader_lora else backbone._frozen_base():
        output = backbone.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
    results = []
    for row, context, task in zip(output.logits, contexts, tokens, strict=True):
        logits = row[context - 1 : context - 1 + len(task.target_ids)]
        target = torch.tensor(task.target_ids, device=device)
        results.append(DenseReaderOutput(logits, gold_token_nll(logits, target)))
    return results
