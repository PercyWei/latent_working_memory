"""按真实 token 进度分组执行 In-Place TTT，保留逐样本 fast weights。"""

from opt_einsum import contract
import torch


def grouped_mlp_forward(mlp, x, t=None, past_w=None):
    """等长组的 MLP；TTT 权重形状为 [组内样本数, 输出维度, 中间维度]。"""
    batch, seq_len, _ = x.shape
    ttt = hasattr(mlp, "ttt_conv")
    current_w = None
    if ttt:
        current_w = (
            mlp.down_proj.weight.unsqueeze(0).expand(batch, -1, -1).clone()
            if past_w is None else past_w
        )
    block_size = mlp.mlp_chunk_size or seq_len

    if not ttt or t is None or seq_len < mlp.ttt_chunk:
        output = None
        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            part = x[:, start:end]
            h = mlp.act_fn(mlp.gate_proj(part)) * mlp.up_proj(part)
            y = (
                contract("g d h, g c h -> g c d", current_w, h)
                if ttt else mlp.down_proj(h)
            )
            if output is None:
                output = y.new_empty((batch, seq_len, mlp.hidden_size))
            output[:, start:end] = y
        return (output, current_w) if ttt else output

    # 卷积按原 TTT chunk 分别零填充，MLP 分块不改变卷积的边界。
    t_padded = mlp.padding(t)
    _, chunk_num, chunk_size, _ = t_padded.shape
    t_conv = (
        mlp.ttt_conv(t_padded.transpose(-1, -2).reshape(batch * chunk_num, -1, chunk_size))
        .transpose(-1, -2)
        .reshape(batch, chunk_num, chunk_size, -1)
    )
    output = torch.zeros_like(t_conv)
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        part = x[:, start:end]
        h = mlp.act_fn(mlp.gate_proj(part)) * mlp.up_proj(part)
        h_padded = mlp.padding(h)
        first_chunk = start // mlp.ttt_chunk
        for offset in range(h_padded.shape[1]):
            index = first_chunk + offset
            current_h = h_padded[:, offset]
            current_t = t_conv[:, index]
            output[:, index] = contract("g d h, g c h -> g c d", current_w, current_h)
            # 满 chunk 在本次读出后立即更新；最后一个不满 chunk 仅保留尾段。
            if seq_len % mlp.ttt_chunk == 0 or index != chunk_num - 1:
                if mlp.ttt_proj is None:
                    dw = contract("g c h, g c d -> g d h", current_h, current_t) * mlp.ttt_lr
                else:
                    dw = contract(
                        "g c h, g c d, d e -> g e h",
                        current_h, current_t, mlp.ttt_proj.weight,
                    ) * mlp.ttt_lr
                current_w = current_w + dw
    return output.flatten(1, 2)[:, :seq_len], current_w


def ttt_forward(mlp, hidden_states, target_states, states, lengths):
    """真实新 token 位于每行末尾；状态为 (normalized x 尾段、target 尾段、W)。"""
    _, width, _ = hidden_states.shape
    output = torch.zeros_like(hidden_states)
    new_states = list(states)
    groups = {}
    for row, length in enumerate(lengths):
        if length == 0:
            continue
        past_x, past_t, past_w = states[row]
        x = hidden_states[row : row + 1, width - length :]
        t = target_states[row : row + 1, width - length :]
        combined_x = torch.cat((past_x, x), dim=1) if past_x is not None else x
        combined_t = torch.cat((past_t, t), dim=1) if past_t is not None else t
        updating = combined_x.shape[1] >= mlp.ttt_chunk
        # 不满一个 chunk 时，原 decoder 只投影本次新增 x；满 chunk 后重算尾段及新增 x。
        read_x = combined_x if updating else x
        key = (
            read_x.shape[1], updating, past_w is None,
            None if past_w is None else past_w.dtype,
        )
        groups.setdefault(key, []).append((row, length, read_x, combined_x, combined_t, past_w))

    for (_, updating, initial, _), rows in groups.items():
        if len(rows) == 1:
            x = rows[0][2]
            t = rows[0][4] if updating else None
        else:
            x = torch.cat([item[2] for item in rows], dim=0)
            t = torch.cat([item[4] for item in rows], dim=0) if updating else None
        if initial:
            past_w = None
        elif len(rows) == 1:
            past_w = rows[0][5].unsqueeze(0)
        else:
            past_w = torch.stack([item[5] for item in rows], dim=0)
        values, weights = grouped_mlp_forward(mlp, x, t, past_w)
        for index, (row, length, _, combined_x, combined_t, previous_w) in enumerate(rows):
            output[row, width - length :] = values[index, -length:]
            tail_length = combined_x.shape[1] % mlp.ttt_chunk
            if tail_length:
                tail_x = combined_x[:, -tail_length:].clone()
                tail_t = combined_t[:, -tail_length:].clone()
            else:
                tail_x, tail_t = None, None
            # 无更新时直接保留原权重张量；读取分组的临时堆叠不进入长期状态。
            current_w = previous_w if previous_w is not None and not updating else weights[index]
            new_states[row] = tail_x, tail_t, current_w
    return output, new_states
