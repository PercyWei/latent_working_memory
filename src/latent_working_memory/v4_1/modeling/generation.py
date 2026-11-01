"""异长样本的批量生成；正文压缩完成后固定 memory，只保留读取状态。"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .attention import decoder_forward


@dataclass
class DecodeState:
    memory_kv: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    memory_mask: torch.Tensor
    raw_mask: torch.Tensor
    positions: torch.Tensor
    last_hidden: torch.Tensor

    def select(self, indices):
        rows = torch.tensor(indices, device=self.positions.device, dtype=torch.long)
        return DecodeState(
            memory_kv=[tuple(value.index_select(0, rows) for value in kv)
                       for kv in self.memory_kv],
            keys=[value.index_select(0, rows) for value in self.keys],
            values=[value.index_select(0, rows) for value in self.values],
            memory_mask=self.memory_mask.index_select(0, rows),
            raw_mask=self.raw_mask.index_select(0, rows),
            positions=self.positions.index_select(0, rows),
            last_hidden=self.last_hidden.index_select(0, rows),
        )


def _pack(states):
    device = states[0].last_hidden.device
    memory_lengths = torch.tensor([state.memory_kv[0][0].shape[2] for state in states], device=device)
    raw_lengths = torch.tensor([state.keys[0].shape[2] for state in states], device=device)
    memory_length = max(state.memory_kv[0][0].shape[2] for state in states)
    raw_length = max(state.keys[0].shape[2] for state in states)
    memories, keys, values = [], [], []
    for layer in range(len(states[0].keys)):
        layer_memory = []
        for component in range(3):
            parts = []
            for state in states:
                value = state.memory_kv[layer][component]
                padding = memory_length - value.shape[2]
                pad = (0, padding) if component == 2 else (0, 0, 0, padding)
                parts.append(F.pad(value, pad, value=float("-inf") if component == 2 else 0))
            layer_memory.append(torch.cat(parts))
        memories.append(tuple(layer_memory))
        keys.append(torch.cat([F.pad(state.keys[layer], (0, 0, 0, raw_length - state.keys[layer].shape[2]))
                               for state in states]))
        values.append(torch.cat([F.pad(state.values[layer], (0, 0, 0, raw_length - state.values[layer].shape[2]))
                                 for state in states]))
    return DecodeState(
        memories, keys, values,
        torch.arange(memory_length, device=device)[None] < memory_lengths[:, None],
        torch.arange(raw_length, device=device)[None] < raw_lengths[:, None],
        torch.tensor([state.position for state in states], device=device),
        torch.cat([state.last_hidden for state in states]),
    )


def _advance(model, sequences, state):
    """问题及生成 token 不触发压缩；padding KV 永久保持不可见。"""
    width = max(len(sequence) for sequence in sequences)
    if width == 0:
        return
    lengths = torch.tensor([len(sequence) for sequence in sequences], device=model.device)
    ids = torch.tensor([list(sequence) + [0] * (width - len(sequence)) for sequence in sequences],
                       device=model.device, dtype=torch.long)
    hidden = model.backbone.model.embed_tokens(ids)
    positions = state.positions[:, None] + torch.arange(width, device=model.device)[None]
    cosine, sine = model.backbone.model.rotary_emb(hidden, positions)
    raw_mask = torch.cat((state.raw_mask, torch.arange(width, device=model.device)[None]
                          < lengths[:, None]), dim=1)
    key_mask = torch.cat((state.memory_mask, raw_mask), dim=1)
    for index, layer in enumerate(model.backbone.model.layers):
        hidden, keys, values, _ = decoder_forward(
            layer, hidden, state.keys[index], state.values[index],
            *state.memory_kv[index], cosine, sine, key_mask,
        )
        state.keys[index], state.values[index] = keys, values
    hidden = model.backbone.model.norm(hidden)
    last = hidden[torch.arange(len(sequences), device=model.device), lengths.clamp_min(1) - 1]
    state.last_hidden = torch.where(lengths[:, None] > 0, last, state.last_hidden)
    state.raw_mask = raw_mask
    state.positions += lengths


def generate_batch(model, contexts, questions, max_new_tokens, eos_token_id, on_complete):
    if torch.is_inference_mode_enabled():
        raise RuntimeError("online TTT needs autograd; use no_grad(), not inference_mode()")
    count = len(contexts)
    if not count or len(questions) != count or len(max_new_tokens) != count:
        raise ValueError("contexts, questions and generation budgets must have the same nonzero batch size")
    window = model.backbone.config.max_position_embeddings
    for context, question, budget in zip(contexts, questions, max_new_tokens):
        if not context or type(budget) is not int or budget < 1:
            raise ValueError("generation requires nonempty contexts and positive integer token budgets")
        if len(context) + len(question) + budget > window:
            raise ValueError("prompt plus generation budget exceeds the model position limit")
    states, _ = model._ingest_batch(contexts, None, None, differentiable=False)
    diagnostics = [model._diagnostics(state) for state in states]
    state = _pack(states)
    del states
    question_length = max(len(question) for question in questions)
    for start in range(0, question_length, model.config.prefill_chunk_size):
        _advance(model, [question[start:start + model.config.prefill_chunk_size]
                         for question in questions], state)
    eos = model.backbone.generation_config.eos_token_id if eos_token_id is None else eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
    active = list(range(count))
    tokens, results = [[] for _ in contexts], [None] * count
    while active:
        predicted = model.backbone.lm_head(state.last_hidden).argmax(dim=-1).tolist()
        continuing, next_tokens = [], []
        for row, (index, token) in enumerate(zip(active, predicted)):
            tokens[index].append(token)
            reason = "eos" if token in eos else (
                "max_new_tokens" if len(tokens[index]) == max_new_tokens[index] else None
            )
            if reason is None:
                continuing.append(row)
                next_tokens.append((token,))
            else:
                result = {"token_ids": tokens[index], "stop_reason": reason, **diagnostics[index]}
                results[index] = result
                if on_complete is not None:
                    on_complete(index, result)
        if not continuing:
            break
        if len(continuing) != len(active):
            state = state.select(continuing)
            active = [active[row] for row in continuing]
        _advance(model, next_tokens, state)
    return results
