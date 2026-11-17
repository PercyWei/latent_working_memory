"""按评估长度设置 batch，并按真实 token 数排序后相邻组批。"""

from array import array
from collections import Counter


BATCHING = "Descending actual token length across tasks within each RULER length; adjacent batches; left padding to batch maximum"


def execution_parameters(batch_size, values, lengths, inference_mode):
    if batch_size < 1:
        raise ValueError("--batch-size 必须为正整数")
    mapping = {}
    for value in values:
        try:
            length_text, size_text = value.split("=", 1)
            length, size = int(length_text), int(size_text)
        except ValueError as error:
            raise ValueError("--batch-size-by-length 使用 LENGTH=BATCH，例如 16384=4") from error
        if length not in lengths:
            raise ValueError(f"batch 长度不在本次评估中：{length}")
        if size < 1:
            raise ValueError("按长度设置的 batch size 必须为正整数")
        if str(length) in mapping:
            raise ValueError(f"batch 长度重复：{length}")
        mapping[str(length)] = size
    effective = {str(length): mapping.get(str(length), batch_size) for length in lengths}
    if inference_mode == "original" and any(size != 1 for size in effective.values()):
        raise ValueError("original 推理要求所有评估长度的有效 batch size 为 1")
    return {
        "batch_size": batch_size,
        "batch_size_by_length": mapping,
        "effective_batch_size_by_length": effective,
        "batching": BATCHING if inference_mode == "optimized" else "Original single-sample order",
    }


def prepare_batches(assigned, tokenizer, parameters, inference_mode, generation_budget, progress):
    group_totals = Counter(sample[0] for sample in assigned.values())
    if inference_mode == "original":
        current_length = None
        for sample_id, sample in assigned.items():
            if current_length != sample[0]:
                current_length = sample[0]
                progress.start_group("ruler", current_length, group_totals[current_length], 1)
            yield sample[0], [(sample_id, sample, None)]
        return
    for length in dict.fromkeys(sample[0] for sample in assigned.values()):
        progress.start_group(
            "ruler", length, group_totals[length],
            parameters["effective_batch_size_by_length"][str(length)],
        )
        pending = []
        for sample_id, sample in assigned.items():
            if sample[0] != length:
                continue
            row = sample[4]
            token_ids = tokenizer.encode(
                row["input"] + row["answer_prefix"], add_special_tokens=True, truncation=False
            )
            if len(token_ids) + generation_budget > length:
                raise ValueError(
                    f"{sample_id}: prompt ({len(token_ids)}) + generation budget "
                    f"({generation_budget}) exceeds {length}"
                )
            pending.append((sample_id, sample, array("I", token_ids)))
        pending.sort(key=lambda item: -len(item[2]))
        size = parameters["effective_batch_size_by_length"][str(length)]
        for start in range(0, len(pending), size):
            yield length, [
                (sample_id, sample, token_ids.tolist())
                for sample_id, sample, token_ids in pending[start : start + size]
            ]
