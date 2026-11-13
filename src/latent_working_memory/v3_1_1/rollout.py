"""因果硬分段：提交 pending，保留新单元；末尾提交不参与策略学习。"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .modeling.streaming import SourceStream


@dataclass(frozen=True)
class Rollout:
    actions: tuple[bool, ...]
    old_log_probs: torch.Tensor
    qa_nll: float
    block_lengths: tuple[int, ...]


def _compress_segment(model, pending, *memory_blocks):
    """压缩重算显式接收历史 memory，不捕获可变 stream 状态。"""
    return model.compress(pending, memory_blocks)


def _boundary_segment(model, visible, positions, *memory_blocks):
    return model.boundary_logits(memory_blocks, visible, positions)


def _replay_fixed(model, units, actions, bptt_steps):
    """同一提交段的边界共享因果前向，压缩时排除仅用于判断的新单元。"""
    blocks, log_probs, entropies, lengths = [], [], [], []
    history = []
    ends = [index + 1 for index, commit in enumerate(actions) if commit] + [len(units)]
    start = 0
    for end in ends:
        pending = tuple(token for unit in units[start:end] for token in unit)
        visible_end = end + (end < len(units))
        positions, offset = [], len(units[start]) - 1
        for unit in units[start + 1:visible_end]:
            offset += len(unit)
            positions.append(offset)
        # 短前缀只保留层级重算；长前缀也按段重算，限制跨段常驻的激活。
        visible = pending + (tuple(units[end]) if end < len(units) else ())
        logits = None
        if positions:
            if sum(map(len, history)) + len(visible) <= 4096:
                logits = model.boundary_logits(history, visible, positions)
            else:
                logits = checkpoint(
                    _boundary_segment, model, visible, tuple(positions), *history,
                    use_reentrant=False,
                )
        memory = checkpoint(
            _compress_segment, model, pending, *history,
            use_reentrant=False,
        )
        if positions:
            # 较早位置不能读取同段中较晚的原文；lookahead 只参与本段最后的判断。
            chosen = torch.tensor(actions[start:visible_end - 1], device=logits.device, dtype=torch.bool)
            log_probs.append(F.logsigmoid(torch.where(chosen, logits, -logits)))
            entropies.append(F.softplus(logits) - logits * logits.sigmoid())
        blocks.append(memory)
        history.append(memory)
        if bptt_steps is not None and len(blocks) % bptt_steps == 0:
            # 只截断后续上下文；原输出仍接收 Reader 的直接 QA 梯度。
            history = [block.detach() for block in history]
        lengths.append(len(pending))
        start = end
    zero = blocks[0].new_empty((0,), dtype=torch.float32)
    return (
        tuple(blocks), tuple(actions),
        torch.cat(log_probs) if log_probs else zero,
        torch.cat(entropies) if entropies else zero,
        tuple(lengths),
    )


def execute(model, input_ids, unit_tokens, actions=None, generator=None, bptt_steps=None):
    """actions=None 时采样（无 generator 则确定性判断），否则回放既定硬动作。"""
    units = tuple(input_ids[i:i + unit_tokens] for i in range(0, len(input_ids), unit_tokens))
    if not units:
        raise ValueError("stream input must be nonempty")
    if actions is not None and len(actions) != len(units) - 1:
        raise ValueError("recorded actions must match the fixed decision positions")
    if actions is not None and torch.is_grad_enabled():
        return _replay_fixed(model, units, actions, bptt_steps)
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


def execute_replay_batch(model, contexts, unit_tokens, actions, bptt_steps=None):
    """按实际写入轮次批量回放，各行独立维护分段、历史及反传跨度。"""
    plans = []
    for context, decisions in zip(contexts, actions, strict=True):
        units = tuple(context[start:start + unit_tokens]
                      for start in range(0, len(context), unit_tokens))
        if not units or len(decisions) != len(units) - 1:
            raise ValueError("recorded actions must match nonempty source decision positions")
        ends = [index + 1 for index, commit in enumerate(decisions) if commit] + [len(units)]
        segments, start = [], 0
        for end in ends:
            pending = tuple(token for unit in units[start:end] for token in unit)
            visible_end = end + (end < len(units))
            positions, offset = [], len(units[start]) - 1
            for unit in units[start + 1:visible_end]:
                offset += len(unit)
                positions.append(offset)
            visible = pending + (tuple(units[end]) if end < len(units) else ())
            segments.append((pending, visible, tuple(positions), decisions[start:visible_end - 1]))
            start = end
        plans.append(segments)
    blocks, history, log_probs, entropies, lengths = ([[] for _ in plans] for _ in range(5))
    for step in range(max(map(len, plans))):
        active = [index for index, plan in enumerate(plans) if step < len(plan)]
        boundary = [index for index in active if plans[index][step][2]]
        if boundary:
            memories = tuple(tuple(history[index]) for index in boundary)
            visible = tuple(plans[index][step][1] for index in boundary)
            positions = tuple(plans[index][step][2] for index in boundary)
            arguments = (memories, visible, positions)
            logits = (checkpoint(model.boundary_logits_batch, *arguments, use_reentrant=False)
                      if any(sum(map(len, past)) + len(tokens) > 4096
                             for past, tokens in zip(memories, visible, strict=True))
                      else model.boundary_logits_batch(*arguments))
            for index, values in zip(boundary, logits, strict=True):
                chosen = torch.tensor(plans[index][step][3], device=values.device, dtype=torch.bool)
                log_probs[index].append(F.logsigmoid(torch.where(chosen, values, -values)))
                entropies[index].append(F.softplus(values) - values * values.sigmoid())
        pending = tuple(plans[index][step][0] for index in active)
        memories = tuple(tuple(history[index]) for index in active)
        compressed = checkpoint(model.compress_batch, pending, memories, use_reentrant=False)
        for row, index in enumerate(active):
            memory = compressed[row]
            blocks[index].append(memory)
            history[index].append(memory)
            if bptt_steps is not None and len(blocks[index]) % bptt_steps == 0:
                history[index] = [block.detach() for block in history[index]]
            lengths[index].append(len(pending[row]))
    results = []
    for index, memories in enumerate(blocks):
        zero = memories[0].new_empty((0,), dtype=torch.float32)
        results.append((
            tuple(memories), tuple(actions[index]),
            torch.cat(log_probs[index]) if log_probs[index] else zero,
            torch.cat(entropies[index]) if entropies[index] else zero,
            tuple(lengths[index]),
        ))
    return results


@torch.no_grad()
def execute_batch(model, contexts, unit_tokens, uniforms=None):
    """批量分段；uniforms 给出各行随机判断量，省略时确定性判断。"""
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
        if uniforms is None:
            commits = (logits >= 0).tolist()
        else:
            commits = [uniforms[index][step - 1] < probability for index, probability in
                       zip(active, logits.sigmoid().cpu().tolist(), strict=True)]
        chosen_tensor = torch.tensor(commits, device=logits.device, dtype=torch.bool)
        probabilities = F.logsigmoid(torch.where(chosen_tensor, logits, -logits))
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
            for row, memory in zip(committed, memories, strict=True):
                # H 内容与位置未变，只保留其 KV，追加新 memory 与 b。
                previous[row].crop(sum(map(len, blocks[active[row]])))
                blocks[active[row]].append(memory)
            refreshed = model.prefill_source_batch(
                [(memory,) for memory in memories],
                [arriving[row] for row in committed],
                [previous[row] for row in committed],
            )
            previous.clear()
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
def sample_group(model, document, rollout, generator):
    paths = []
    for _ in range(rollout["group_size"]):
        blocks, actions, log_probs, _, lengths = execute(
            model, document.input_ids, rollout["unit_tokens"], generator=generator,
        )
        loss = qa_nll(model, blocks, document)
        paths.append(Rollout(actions, log_probs.detach(), float(loss), lengths))
    return tuple(paths)


@torch.no_grad()
def sample_group_batch(model, documents, rollout, generator):
    """按文档／路径原顺序分配随机数，每轮批量采样各文档的一条路径。"""
    group_size, unit_tokens = rollout["group_size"], rollout["unit_tokens"]
    uniforms = (None if generator is None else [
        torch.rand((group_size, (len(document.input_ids) - 1) // unit_tokens),
                   generator=generator).tolist()
        for document in documents
    ])
    paths = [[] for _ in documents]
    scores = [{} for _ in documents]
    for path_index in range(group_size):
        results = execute_batch(
            model, [document.input_ids for document in documents], unit_tokens,
            None if uniforms is None else [values[path_index] for values in uniforms],
        )
        losses = [[] for _ in documents]
        for question in range(max(len(document.qas) for document in documents)):
            active = [index for index, document in enumerate(documents)
                      if question < len(document.qas)]
            values = model.answer_nll_batch(
                [results[index][0] for index in active],
                [documents[index].qas[question].prompt_ids for index in active],
                [documents[index].qas[question].target_ids for index in active],
            )
            for index, loss in zip(active, values, strict=True):
                losses[index].append(loss)
        for index, result in enumerate(results):
            # 同篇相同动作是同一确定性轨迹，避免批量舍入噪声产生伪优势。
            score = scores[index].setdefault(result[1], float(torch.stack(losses[index]).mean()))
            paths[index].append(Rollout(
                result[1], result[2].detach(), score, result[4],
            ))
        del results, result
    return [tuple(group) for group in paths]
