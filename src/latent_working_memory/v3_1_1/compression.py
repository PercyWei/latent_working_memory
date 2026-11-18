"""单块、多倍率训练和固定主倍率验证共用的正文分段与压缩。"""

from typing import NamedTuple

import torch


class CompressionResult(NamedTuple):
    blocks: tuple[torch.Tensor, ...]
    recent_ids: tuple[int, ...]
    block_lengths: tuple[int, ...]
    pending_ids: tuple[int, ...] = ()

    @property
    def retained_ids(self):
        return self.pending_ids + self.recent_ids


def segment_lengths(input_length, memory_slots, compression, generator=None, fixed_ratio=None,
                    recent_size=0, pending_size=None):
    """返回完整覆盖正文的压缩段长度及原文尾段长度。"""
    if input_length < 1:
        raise ValueError("compression requires a nonempty document")
    if compression.mode == "policy":
        if fixed_ratio is None:
            raise ValueError("policy compression requires a fixed_ratio outside policy rollout")
        minimum, recent = pending_size, recent_size
    elif compression.mode == "single":
        if fixed_ratio is None:
            return (input_length,), 0
        minimum, recent = memory_slots, 0
    else:
        minimum = min(compression.ratios) * memory_slots
        recent = recent_size
    remaining, lengths = input_length - recent, []
    if remaining < minimum:
        raise ValueError("the compression prefix is shorter than the minimum segment after reserving recent")
    widths = ([fixed_ratio * memory_slots] if fixed_ratio is not None
              else [ratio * memory_slots for ratio in compression.ratios])
    while remaining:
        candidates = [width for width in widths if width <= remaining
                      and (remaining == width or remaining - width >= minimum)]
        if not candidates:
            # 仅合法常规倍率为空时收尾，不将剩余全文作为正常采样候选。
            length = remaining
        elif fixed_ratio is not None:
            length = candidates[0]
        else:
            length = candidates[int(torch.randint(len(candidates), (), generator=generator))]
        lengths.append(length)
        remaining -= length
    return tuple(lengths), recent


def compress_documents_batch(model, contexts, compression, generator=None, fixed_ratio=None,
                             recent_size=0, pending_size=None):
    """逐写入轮次批量压缩；历史 memory 的梯度保留给后续压缩和 Reader。"""
    plans = [
        segment_lengths(
            len(tokens), model.memory_slots, compression, generator, fixed_ratio, recent_size,
            pending_size,
        ) for tokens in contexts
    ]
    memories = [[] for _ in contexts]
    offsets = [0] * len(contexts)
    for step in range(max((len(lengths) for lengths, _ in plans), default=0)):
        active = [index for index, (lengths, _) in enumerate(plans) if step < len(lengths)]
        chunks = [contexts[index][offsets[index]:offsets[index] + plans[index][0][step]]
                  for index in active]
        compressed = model.compress_batch(chunks, [tuple(memories[index]) for index in active])
        for index, memory in zip(active, compressed, strict=True):
            memories[index].append(memory)
            offsets[index] += plans[index][0][step]
    return [CompressionResult(tuple(blocks), tuple(tokens[offset:]), lengths)
            for tokens, blocks, offset, (lengths, _) in zip(
                contexts, memories, offsets, plans, strict=True,
            )]


def compress_document(model, input_ids, compression, generator=None, fixed_ratio=None,
                      recent_size=0, pending_size=None):
    return compress_documents_batch(
        model, [input_ids], compression, generator, fixed_ratio, recent_size, pending_size,
    )[0]
