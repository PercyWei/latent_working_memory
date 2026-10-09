"""按 token 长度均分连续输入；压缩块独立于数据原始段落。"""

import hashlib
import random


def example_rng(seed, epoch, identifier):
    value = hashlib.sha256(f"{seed}:{epoch}:{identifier}".encode()).digest()
    return random.Random(int.from_bytes(value[:8], "big"))


def even_token_chunks(token_ids, num_chunks):
    """顺序均分成固定数量的非空块，块长最多相差一个 token。"""
    if len(token_ids) < num_chunks:
        raise ValueError("input must contain at least as many tokens as sampled ICAE chunks")
    size, remainder = divmod(len(token_ids), num_chunks)
    chunks, offset = [], 0
    for index in range(num_chunks):
        end = offset + size + (index < remainder)
        chunks.append(token_ids[offset:end])
        offset = end
    return chunks


def icae_multi_plan(token_ids, total_slots, min_segments, max_segments, seed, identifier):
    """按样本固定采样块数，均分完整正文及总记忆容量。"""
    num_segments = example_rng(seed, 0, identifier).randint(min_segments, max_segments)
    chunks = even_token_chunks(token_ids, num_segments)
    slots, remainder = divmod(total_slots, num_segments)
    slot_counts = [slots + (index < remainder) for index in range(num_segments)]
    return chunks, slot_counts


def ac_token_chunks(token_ids, min_segment_tokens, max_segment_tokens, rng):
    """AutoCompressors 随机连续分段；首段后至少保留两个 token 的监督。"""
    if len(token_ids) < 3:
        raise ValueError("AutoCompressors has no trainable next-token target in the second segment")
    chunks, offset = [], 0
    while offset < len(token_ids):
        maximum = max_segment_tokens
        if offset == 0:
            maximum = min(maximum, len(token_ids) - 2)
        size = rng.randint(min(min_segment_tokens, maximum), maximum)
        chunks.append(token_ids[offset : offset + size])
        offset += size
    return chunks
