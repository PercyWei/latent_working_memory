"""E²-TTT 按 RULER 评估长度设置每卡 batch 上限。"""


def execution_parameters(batch_size, values, ruler_lengths, benchmarks, inference_mode):
    if batch_size < 1:
        raise ValueError("--batch-size 必须为正整数")
    mapping = {}
    for value in values:
        try:
            length_text, size_text = value.split("=", 1)
            length, size = int(length_text), int(size_text)
        except ValueError as error:
            raise ValueError("--batch-size-by-length 使用 LENGTH=BATCH，例如 16384=4") from error
        if length not in ruler_lengths:
            raise ValueError(f"batch 长度不在原 RULER 评估计划中：{length}")
        if size < 1:
            raise ValueError("按长度设置的 batch size 必须为正整数")
        if str(length) in mapping:
            raise ValueError(f"batch 长度重复：{length}")
        mapping[str(length)] = size
    effective = {
        str(length): mapping.get(str(length), batch_size) for length in ruler_lengths
    } if "ruler" in benchmarks else {}
    if inference_mode == "original" and (
        any(size != 1 for size in effective.values())
        or (any(name != "ruler" for name in benchmarks) and batch_size != 1)
    ):
        raise ValueError("original 推理要求所有选中 benchmark 的有效 batch size 为 1")
    return {
        "batch_size": batch_size,
        "batch_size_by_length": mapping,
        "effective_batch_size_by_length": effective,
    }
