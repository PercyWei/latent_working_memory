"""因果硬分段、确定性回放与逐位置奖励账本。"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .modeling.streaming import SourceStream, split_recent


class ExecutionResult(NamedTuple):
    blocks: tuple[torch.Tensor, ...]
    actions: tuple[bool, ...]
    log_probs: torch.Tensor
    entropy: torch.Tensor
    block_lengths: tuple[int, ...]
    recent_ids: tuple[int, ...]
    decision_mask: tuple[bool, ...]
    prefix_retained_counts: tuple[int, ...]
    pending_ids: tuple[int, ...] = ()

    @property
    def retained_ids(self):
        return self.pending_ids + self.recent_ids


@dataclass(frozen=True)
class Rollout:
    actions: tuple[bool, ...]
    old_log_probs: torch.Tensor
    qa_nll: float
    block_lengths: tuple[int, ...]
    recent_ids: tuple[int, ...]
    decision_mask: tuple[bool, ...]
    prefix_retained_counts: tuple[int, ...]
    pending_ids: tuple[int, ...] = ()

    @property
    def retained_ids(self):
        return self.pending_ids + self.recent_ids


def _compress_segment(model, pending, *memory_blocks):
    """压缩重算显式接收历史 memory，不捕获可变 stream 状态。"""
    return model.compress(pending, memory_blocks)


def _boundary_segment(model, visible, positions, *memory_blocks):
    return model.boundary_logits(memory_blocks, visible, positions)


def _source_units(input_ids, pending_stride, recent_size):
    prefix, recent = split_recent(input_ids, recent_size)
    units = tuple(prefix[start:start + pending_stride] for start in range(0, len(prefix), pending_stride))
    return units, recent


def _path_metadata(units, actions, slots, pending_size):
    """固定前缀读取位置的 memory 与 pending 占用，排除固定 recent。"""
    pending, block_count = len(units[0]) if units else 0, 0
    retained = [0, pending]
    mask, lengths = [], []
    for unit, commit in zip(units[1:], actions, strict=True):
        allowed = pending >= pending_size
        if commit and not allowed:
            raise ValueError("recorded commit is shorter than pending_size")
        mask.append(allowed)
        if commit:
            lengths.append(pending)
            block_count += 1
            pending = len(unit)
        else:
            pending += len(unit)
        retained.append(block_count * slots + pending)
    if pending >= pending_size:
        lengths.append(pending)
        block_count += 1
        pending = 0
    retained.append(block_count * slots + pending)
    return tuple(mask), tuple(lengths), tuple(retained)


def _segments(units, actions, pending_size):
    if not units:
        return [], ()
    ends = [index + 1 for index, commit in enumerate(actions) if commit] + [len(units)]
    segments, start, retained_pending = [], 0, ()
    for end in ends:
        pending = tuple(token for unit in units[start:end] for token in unit)
        visible_end = end + (end < len(units))
        positions, offset = [], len(units[start]) - 1
        for unit in units[start + 1:visible_end]:
            offset += len(unit)
            positions.append(offset)
        visible = pending + (tuple(units[end]) if end < len(units) else ())
        compressed = pending
        if end == len(units) and len(pending) < pending_size:
            compressed, retained_pending = (), pending
        segments.append((compressed, visible, tuple(positions), start, visible_end - 1))
        start = end
    return segments, retained_pending


def _replay_fixed(model, units, actions, bptt_steps, temperature, pending_size, recent):
    """同一提交段的边界共享因果前向，收尾压缩排除 recent。"""
    mask, lengths, retained = _path_metadata(units, actions, model.memory_slots, pending_size)
    segments, retained_pending = _segments(units, actions, pending_size)
    blocks, log_probs, entropies, history = [], [], [], []
    for pending, visible, positions, start, end in segments:
        if positions:
            if sum(map(len, history)) + len(visible) <= 4096:
                logits = model.boundary_logits(history, visible, positions)
            else:
                logits = checkpoint(
                    _boundary_segment, model, visible, positions, *history, use_reentrant=False,
                )
            logits = logits / temperature
            chosen = torch.tensor(actions[start:end], device=logits.device, dtype=torch.bool)
            allowed = torch.tensor(mask[start:end], device=logits.device, dtype=torch.bool)
            log_probs.append(torch.where(
                allowed, F.logsigmoid(torch.where(chosen, logits, -logits)), 0.,
            ))
            entropies.append(torch.where(allowed, F.softplus(logits) - logits * logits.sigmoid(), 0.))
        if pending:
            memory = checkpoint(_compress_segment, model, pending, *history, use_reentrant=False)
            blocks.append(memory)
            history.append(memory)
            if bptt_steps is not None and len(blocks) % bptt_steps == 0:
                history = [block.detach() for block in history]
    zero = model.memory_embeddings.new_empty((0,), dtype=torch.float32)
    return ExecutionResult(
        tuple(blocks), tuple(actions), torch.cat(log_probs) if log_probs else zero,
        torch.cat(entropies) if entropies else zero, lengths, recent, mask, retained, retained_pending,
    )


def execute(model, input_ids, pending_stride, actions=None, generator=None, bptt_steps=None,
            temperature=1.0, pending_size=None, recent_size=0):
    """先保留 recent，再采样或回放前缀的硬动作；不足阈值的最终 pending 保留原文。"""
    pending_size = model.memory_slots if pending_size is None else pending_size
    units, recent = _source_units(input_ids, pending_stride, recent_size)
    if not input_ids:
        raise ValueError("stream input must be nonempty")
    if actions is not None and len(actions) != max(0, len(units) - 1):
        raise ValueError("recorded actions must match the fixed prefix decision positions")
    if actions is not None and torch.is_grad_enabled():
        return _replay_fixed(model, units, actions, bptt_steps, temperature, pending_size, recent)
    zero = model.memory_embeddings.new_empty((0,), dtype=torch.float32)
    if not units:
        return ExecutionResult((), (), zero, zero, (), recent, (), (0, 0, 0))
    stream = SourceStream(model, units[0])
    chosen, log_probs, entropies = [], [], []
    for index, unit in enumerate(units[1:]):
        logit = stream.logit(unit) / temperature
        allowed = len(stream.pending_ids) >= pending_size
        if actions is not None:
            commit = actions[index]
            if commit and not allowed:
                raise ValueError("recorded commit is shorter than pending_size")
        elif generator is None:
            commit = allowed and bool(logit.detach() >= 0)
        else:
            # 包括强制等待在内，按固定前缀位置消耗随机数，保持单例／批量映射一致。
            uniform = torch.rand((), generator=generator)
            commit = allowed and bool(uniform < logit.detach().sigmoid().cpu())
        chosen.append(commit)
        log_probs.append(F.logsigmoid(logit if commit else -logit) if allowed else logit.new_zeros(()))
        entropies.append(F.softplus(logit) - logit * logit.sigmoid() if allowed else logit.new_zeros(()))
        stream.advance(unit, commit)
    blocks, pending, _ = stream.finish(pending_size)
    mask, lengths, retained = _path_metadata(units, chosen, model.memory_slots, pending_size)
    return ExecutionResult(
        blocks, tuple(chosen), torch.stack(log_probs) if log_probs else zero,
        torch.stack(entropies) if entropies else zero, lengths, recent, mask, retained, pending,
    )


def execute_replay_batch(model, contexts, pending_stride, actions, bptt_steps=None, temperature=1.0,
                         pending_size=None, recent_size=0):
    """按实际写入轮次批量回放，各行独立维护分段、历史与强制动作掩码。"""
    pending_size = model.memory_slots if pending_size is None else pending_size
    plans, tails, pending_tails, metadata = [], [], [], []
    for context, decisions in zip(contexts, actions, strict=True):
        units, recent = _source_units(context, pending_stride, recent_size)
        if not context or len(decisions) != max(0, len(units) - 1):
            raise ValueError("recorded actions must match nonempty source prefix decision positions")
        metadata.append(_path_metadata(units, decisions, model.memory_slots, pending_size))
        segments, pending = _segments(units, decisions, pending_size)
        plans.append(segments)
        tails.append(recent)
        pending_tails.append(pending)
    blocks, history, log_probs, entropies = ([[] for _ in plans] for _ in range(4))
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
                values = values / temperature
                start, end = plans[index][step][3:]
                chosen = torch.tensor(actions[index][start:end], device=values.device, dtype=torch.bool)
                allowed = torch.tensor(metadata[index][0][start:end], device=values.device, dtype=torch.bool)
                log_probs[index].append(torch.where(
                    allowed, F.logsigmoid(torch.where(chosen, values, -values)), 0.,
                ))
                entropies[index].append(torch.where(
                    allowed, F.softplus(values) - values * values.sigmoid(), 0.,
                ))
        compressed_indices = [index for index in active if plans[index][step][0]]
        if compressed_indices:
            pending = tuple(plans[index][step][0] for index in compressed_indices)
            memories = tuple(tuple(history[index]) for index in compressed_indices)
            compressed = checkpoint(model.compress_batch, pending, memories, use_reentrant=False)
            for row, index in enumerate(compressed_indices):
                memory = compressed[row]
                blocks[index].append(memory)
                history[index].append(memory)
                if bptt_steps is not None and len(blocks[index]) % bptt_steps == 0:
                    history[index] = [block.detach() for block in history[index]]
    zero = model.memory_embeddings.new_empty((0,), dtype=torch.float32)
    return [ExecutionResult(
        tuple(blocks[index]), tuple(actions[index]),
        torch.cat(log_probs[index]) if log_probs[index] else zero,
        torch.cat(entropies[index]) if entropies[index] else zero,
        metadata[index][1], tails[index], metadata[index][0], metadata[index][2], pending_tails[index],
    ) for index in range(len(plans))]


@torch.no_grad()
def execute_batch(model, contexts, pending_stride, uniforms=None, temperature=1.0,
                  pending_size=None, recent_size=0):
    """批量分段；uniforms 按正文固定位置索引，强制等待不产生策略概率。"""
    pending_size = model.memory_slots if pending_size is None else pending_size
    sources = [_source_units(context, pending_stride, recent_size) for context in contexts]
    units = [source[0] for source in sources]
    recent = [source[1] for source in sources]
    if not contexts or any(not context for context in contexts):
        raise ValueError("a stream batch must contain nonempty inputs")
    prefixes = [tuple(token for unit in sequence for token in unit) for sequence in units]
    blocks = [[] for _ in units]
    pending = [len(sequence[0]) if sequence else 0 for sequence in units]
    chosen, log_probs, entropies = ([[] for _ in units] for _ in range(3))
    pending_tails = [() for _ in units]
    caches = [None] * len(units)
    initial = [index for index, sequence in enumerate(units) if sequence]
    if initial:
        values = model.prefill_source_batch([blocks[index] for index in initial],
                                           [units[index][0] for index in initial])
        for index, value in zip(initial, values, strict=True):
            caches[index] = value

    def flush(indices):
        compressed_indices = []
        for index in indices:
            if pending[index] >= pending_size:
                compressed_indices.append(index)
            else:
                pending_tails[index] = prefixes[index][-pending[index]:] if pending[index] else ()
                caches[index] = None
        if compressed_indices:
            final = model.compress_source_batch([caches[index] for index in compressed_indices])
            for index, memory in zip(compressed_indices, final, strict=True):
                blocks[index].append(memory)
                caches[index] = None

    flush([index for index, sequence in enumerate(units) if len(sequence) <= 1])
    for step in range(1, max(map(len, units))):
        active = [index for index, sequence in enumerate(units) if step < len(sequence)]
        previous = [caches[index] for index in active]
        arriving = [units[index][step] for index in active]
        logits, observed = model.observe_source_batch(previous, arriving)
        policy_logits = logits / temperature
        allowed = [pending[index] >= pending_size for index in active]
        if uniforms is None:
            commits = [valid and bool(value) for valid, value in zip(allowed, (logits >= 0).tolist(), strict=True)]
        else:
            commits = [allowed[row] and uniforms[index][step - 1] < probability
                       for row, (index, probability) in enumerate(zip(
                           active, policy_logits.sigmoid().cpu().tolist(), strict=True,
                       ))]
        chosen_tensor = torch.tensor(commits, device=logits.device, dtype=torch.bool)
        allowed_tensor = torch.tensor(allowed, device=logits.device, dtype=torch.bool)
        probabilities = torch.where(
            allowed_tensor, F.logsigmoid(torch.where(chosen_tensor, policy_logits, -policy_logits)), 0.,
        )
        uncertainty = torch.where(
            allowed_tensor, F.softplus(policy_logits) - policy_logits * policy_logits.sigmoid(), 0.,
        )
        committed = []
        for row, index in enumerate(active):
            chosen[index].append(commits[row])
            log_probs[index].append(probabilities[row])
            entropies[index].append(uncertainty[row])
            caches[index] = observed[row]
            if commits[row]:
                caches[index] = observed[row] = None
                pending[index] = len(arriving[row])
                committed.append(row)
            else:
                previous[row] = None
                pending[index] += len(arriving[row])
        if committed:
            memories = model.compress_source_batch([previous[row] for row in committed])
            for row, memory in zip(committed, memories, strict=True):
                previous[row].crop(sum(map(len, blocks[active[row]])))
                blocks[active[row]].append(memory)
            refreshed = model.prefill_source_batch(
                [(memory,) for memory in memories], [arriving[row] for row in committed],
                [previous[row] for row in committed],
            )
            previous.clear()
            for row, cache in zip(committed, refreshed, strict=True):
                caches[active[row]] = cache
            del cache, refreshed
        del previous, observed
        flush([index for index in active if step + 1 == len(units[index])])
    results = []
    zero = model.memory_embeddings.new_empty((0,), dtype=torch.float32)
    for index, memories in enumerate(blocks):
        mask, lengths, retained = _path_metadata(
            units[index], chosen[index], model.memory_slots, pending_size,
        )
        results.append(ExecutionResult(
            tuple(memories), tuple(chosen[index]),
            torch.stack(log_probs[index]) if log_probs[index] else zero,
            torch.stack(entropies[index]) if entropies[index] else zero,
            lengths, recent[index], mask, retained, pending_tails[index],
        ))
    return results


def qa_nll(model, blocks, document, retained_ids=()):
    """答案 token 平均，再按文档 QA 平均；pending 和 recent 按原文顺序进入 Reader。"""
    return torch.stack([
        model.answer_nll(blocks, (*retained_ids, *qa.prompt_ids), qa.target_ids) for qa in document.qas
    ]).mean()


@torch.no_grad()
def sample_paths(model, document, pending_stride, num_samples, generator, temperature=1.0,
                 pending_size=None, recent_size=0):
    paths = []
    for _ in range(num_samples):
        result = execute(model, document.input_ids, pending_stride, generator=generator,
                         temperature=temperature, pending_size=pending_size, recent_size=recent_size)
        loss = qa_nll(model, result.blocks, document, result.retained_ids)
        paths.append(Rollout(result.actions, result.log_probs.detach(), float(loss),
                             result.block_lengths, result.recent_ids, result.decision_mask,
                             result.prefix_retained_counts, result.pending_ids))
    return tuple(paths)


@torch.no_grad()
def sample_paths_batch(model, documents, pending_stride, num_samples, generator, temperature=1.0,
                       score_cache=None, pending_size=None, recent_size=0):
    """按文档／路径分配随机数；同批文档的随机与 greedy 路径共享确定性评分。"""
    uniforms = (None if generator is None else [
        torch.rand((num_samples, max(0, (len(document.input_ids) - recent_size - 1) // pending_stride)), generator=generator).tolist()
        for document in documents
    ])
    paths = [[] for _ in documents]
    scores = [{} for _ in documents] if score_cache is None else score_cache
    for path_index in range(num_samples):
        results = execute_batch(
            model, [document.input_ids for document in documents], pending_stride,
            None if uniforms is None else [values[path_index] for values in uniforms],
            temperature=temperature, pending_size=pending_size, recent_size=recent_size,
        )
        losses = [[] for _ in documents]
        for question in range(max(len(document.qas) for document in documents)):
            active = [index for index, document in enumerate(documents) if question < len(document.qas)]
            values = model.answer_nll_batch(
                [results[index].blocks for index in active],
                [(*results[index].retained_ids, *documents[index].qas[question].prompt_ids) for index in active],
                [documents[index].qas[question].target_ids for index in active],
            )
            for index, loss in zip(active, values, strict=True):
                losses[index].append(loss)
        for index, result in enumerate(results):
            score = scores[index].setdefault(result.actions, float(torch.stack(losses[index]).mean()))
            paths[index].append(Rollout(
                result.actions, result.log_probs.detach(), score, result.block_lengths,
                result.recent_ids, result.decision_mask, result.prefix_retained_counts, result.pending_ids,
            ))
        del results, result
    return [tuple(group) for group in paths]
