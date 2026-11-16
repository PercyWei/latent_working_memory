"""LaCT 按评估长度设置 batch 上限。"""


def parse_batch_size_by_length(values, lengths):
    sizes = {}
    for value in values:
        try:
            length_text, size_text = value.split("=", 1)
            length, size = int(length_text), int(size_text)
        except ValueError as error:
            raise ValueError("--batch-size-by-length 使用 LENGTH=BATCH，例如 32768=8") from error
        if length not in lengths:
            raise ValueError(f"batch 长度不在本次评估中：{length}")
        if size < 1:
            raise ValueError("按长度设置的 batch size 必须为正整数")
        if str(length) in sizes:
            raise ValueError(f"batch 长度重复：{length}")
        sizes[str(length)] = size
    return sizes
