"""因果硬分段：提交 pending，保留新单元；末尾提交不参与策略学习。"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .modeling.streaming import SourceStream


@dataclass(frozen=True)
class Rollout:
    actions: tuple[bool, ...]
    old_log_probs: torch.Tensor
    qa_nll: float
    block_lengths: tuple[int, ...]


def execute(model, input_ids, unit_tokens, actions=None, generator=None):
    """actions=None 时采样（无 generator 则确定性判断），否则回放既定硬动作。"""
    units = tuple(input_ids[i:i + unit_tokens] for i in range(0, len(input_ids), unit_tokens))
    if not units:
        raise ValueError("stream input must be nonempty")
    if actions is not None and len(actions) != len(units) - 1:
        raise ValueError("recorded actions must match the fixed decision positions")
    stream = SourceStream(model, units[0])
    chosen, log_probs, entropies, lengths = [], [], [], []
    for index, unit in enumerate(units[1:]):
        logit = stream.logit(unit)
        if actions is not None:
            commit = actions[index]
        elif generator is None:
            commit = bool(logit.detach() >= 0)
        else:
            commit = bool(torch.rand((), generator=generator) < logit.detach().sigmoid().cpu())
        chosen.append(commit)
        log_probs.append(F.logsigmoid(logit if commit else -logit))
        entropies.append(F.softplus(logit) - logit * logit.sigmoid())
        if commit:
            lengths.append(len(stream.pending_ids))
        stream.advance(unit, commit)
    lengths.append(len(stream.pending_ids))
    blocks = stream.finish()
    zero = blocks[0].new_empty((0,), dtype=torch.float32)
    return (
        blocks, tuple(chosen),
        torch.stack(log_probs) if log_probs else zero,
        torch.stack(entropies) if entropies else zero,
        tuple(lengths),
    )


@torch.no_grad()
def execute_batch(model, contexts, unit_tokens):
    """独立评估的确定性批量分段；各文档维护自己的有效 KV 与不可变记忆块。"""
    units = [tuple(context[start:start + unit_tokens]
                   for start in range(0, len(context), unit_tokens)) for context in contexts]
    if not units or any(not sequence for sequence in units):
        raise ValueError("a stream batch must contain nonempty inputs")
    blocks = [[] for _ in units]
    pending = [len(sequence[0]) for sequence in units]
    chosen, log_probs, entropies, lengths = ([[] for _ in units] for _ in range(4))
    caches = model.prefill_source_batch(blocks, [sequence[0] for sequence in units])

    def flush(indices):
        if not indices:
            return
        final = model.compress_source_batch([caches[index] for index in indices])
        for index, memory in zip(indices, final, strict=True):
            blocks[index].append(memory)
            lengths[index].append(pending[index])
            caches[index] = None

    flush([index for index, sequence in enumerate(units) if len(sequence) == 1])
    for step in range(1, max(map(len, units))):
        active = [index for index, sequence in enumerate(units) if step < len(sequence)]
        previous = [caches[index] for index in active]
        arriving = [units[index][step] for index in active]
        logits, observed = model.observe_source_batch(previous, arriving)
        commits = (logits >= 0).tolist()
        probabilities = F.logsigmoid(torch.where(logits >= 0, logits, -logits))
        uncertainty = F.softplus(logits) - logits * logits.sigmoid()
        committed = []
        for row, index in enumerate(active):
            chosen[index].append(commits[row])
            log_probs[index].append(probabilities[row])
            entropies[index].append(uncertainty[row])
            caches[index] = observed[row]
            if commits[row]:
                # 观察 a+b 的临时 KV 不再使用，及时释放后再压缩 a。
                caches[index] = observed[row] = None
                lengths[index].append(pending[index])
                pending[index] = len(arriving[row])
                committed.append(row)
            else:
                previous[row] = None
                pending[index] += len(arriving[row])
        if committed:
            # previous 尚未包含仅用于边界判断的 b；只提交此前的 a。
            memories = model.compress_source_batch([previous[row] for row in committed])
            previous.clear()
            for row, memory in zip(committed, memories, strict=True):
                blocks[active[row]].append(memory)
            # 提交后以新记忆块和 b 重建有效上下文，不能沿用 a+b 的 KV。
            refreshed = model.prefill_source_batch(
                [blocks[active[row]] for row in committed],
                [arriving[row] for row in committed],
            )
            for row, cache in zip(committed, refreshed, strict=True):
                caches[active[row]] = cache
            del cache, refreshed
        del previous, observed
        flush([index for index in active if step + 1 == len(units[index])])
    results = []
    for index, memories in enumerate(blocks):
        zero = memories[0].new_empty((0,), dtype=torch.float32)
        results.append((
            tuple(blocks[index]), tuple(chosen[index]),
            torch.stack(log_probs[index]) if log_probs[index] else zero,
            torch.stack(entropies[index]) if entropies[index] else zero,
            tuple(lengths[index]),
        ))
    return results


def qa_nll(model, blocks, document):
    """每题先平均答案 token，再对固定 QA 集合平均；各路径使用相同分母。"""
    return torch.stack([
        model.answer_nll(blocks, qa.prompt_ids, qa.target_ids) for qa in document.qas
    ]).mean()


@torch.no_grad()
def sample_group(model, document, objective, generator):
    paths = []
    for _ in range(objective["group_size"]):
        blocks, actions, log_probs, _, lengths = execute(
            model, document.input_ids, objective["unit_tokens"], generator=generator,
        )
        loss = qa_nll(model, blocks, document)
        paths.append(Rollout(actions, log_probs.detach(), float(loss), lengths))
    return tuple(paths)
