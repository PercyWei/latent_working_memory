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


def ac_plan(token_ids, total_slots, num_segments, bptt_steps, max_positions, rng=None):
    """固定正文段数与总 K；每个 BPTT 子块内部随机切分连续文本。

    每组段数由 bptt_steps 决定，尾组取剩余段数；None 将全部段作为一组。
    先均分正文以固定各组的 token 总数，再让组内段长在平均段长的 2/3–4/3 内变化。
    6144 tokens、4 段、BPTT=2 时得到论文的
    3072-token 子块及 1024–2048-token 段。rng=None 用于确定性均分评估。
    """
    if len(token_ids) < 2 * num_segments:
        raise ValueError("AutoCompressors input must contain at least two tokens per segment")
    if total_slots < num_segments:
        raise ValueError("AutoCompressors num_segments must not exceed total memory slots")
    slots, remainder = divmod(total_slots, num_segments)
    slot_counts = [slots + (index < remainder) for index in range(num_segments)]
    nominal = even_token_chunks(token_ids, num_segments)
    cumulative = 0
    capacities = []
    for count in slot_counts:
        cumulative += count
        capacities.append(max_positions - cumulative)
    chunks, offset = [], 0
    group_steps = num_segments if bptt_steps is None else bptt_steps
    for start in range(0, num_segments, group_steps):
        stop = min(start + group_steps, num_segments)
        window_tokens = sum(len(part) for part in nominal[start:stop])
        count = stop - start
        minimum = max(2, (2 * window_tokens + 3 * count - 1) // (3 * count))
        maximum = 4 * window_tokens // (3 * count)
        upper = [min(maximum, capacity) for capacity in capacities[start:stop]]
        if any(limit < minimum for limit in upper) or not (
            count * minimum <= window_tokens <= sum(upper)
        ):
            raise ValueError("AutoCompressors segments and cumulative memory exceed model window")
        remaining = window_tokens
        for index in range(count):
            if rng is None:
                size = len(nominal[start + index])
                if size > upper[index]:
                    raise ValueError(
                        "AutoCompressors segments and cumulative memory exceed model window"
                    )
            else:
                low = max(minimum, remaining - sum(upper[index + 1 :]))
                high = min(upper[index], remaining - minimum * (count - index - 1))
                size = rng.randint(low, high)
            chunks.append(token_ids[offset : offset + size])
            offset += size
            remaining -= size
    return chunks, slot_counts
