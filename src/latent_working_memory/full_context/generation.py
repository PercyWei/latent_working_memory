"""完整上下文批量推理共用的左填充和生成边界。"""

import torch
from torch.nn.utils.rnn import pad_sequence


def left_pad(rows, pad_token_id, device):
    """左侧补齐 token 行，按真实长度构建 mask，支持 pad 与 EOS 共用 ID。"""
    inputs = pad_sequence(
        [torch.tensor(row, dtype=torch.long) for row in rows],
        batch_first=True, padding_value=pad_token_id, padding_side="left",
    ).to(device)
    width = inputs.shape[1]
    lengths = torch.tensor([len(row) for row in rows], device=device)
    mask = torch.arange(width, device=device) >= (width - lengths[:, None])
    return inputs, mask.long()


def generated_ids(row, eos_token_id):
    """保留首个 EOS，去掉生成器在该行结束后追加的填充。"""
    if eos_token_id in row:
        return row[:row.index(eos_token_id) + 1]
    return row
