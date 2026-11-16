"""左 padding 的有效 attention 与逐样本原生 TTT 状态。"""

from dataclasses import dataclass

from einops import rearrange
from flash_attn import flash_attn_varlen_func
import torch
import torch.nn.functional as F

from . import mlp, swiglu


@dataclass
class PaddingLayout:
    mask: torch.Tensor
    positions: tuple
    lengths: tuple
    active: tuple
    indices: torch.Tensor
    cu_seqlens: torch.Tensor


def padding_layout(mask):
    positions = tuple(row.nonzero(as_tuple=False).flatten() for row in mask)
    lengths = tuple(position.numel() for position in positions)
    active = tuple(index for index, length in enumerate(lengths) if length)
    indices = mask.flatten().nonzero(as_tuple=False).flatten()
    counts = torch.tensor([lengths[index] for index in active], device=mask.device, dtype=torch.int32)
    cu_seqlens = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
    return PaddingLayout(mask, positions, lengths, active, indices, cu_seqlens)


def padding_forward(layer, hidden_states, layout, past_key_values, previous_seen):
    batch, width, _ = hidden_states.shape
    last = (
        past_key_values[layer.layer_idx]
        if past_key_values is not None and len(past_key_values) > layer.layer_idx else None
    )
    state = (
        last["padding_state"] if last is not None else {
            "seen": (0,) * batch,
            "rows": [{"attn_state": None, "recurrent_state": None} for _ in range(batch)],
        }
    )
    if len(state["rows"]) != batch:
        raise ValueError("Padding cache batch size differs from the input batch")
    seen = list(state["seen"])
    rows = list(state["rows"])

    q, k = layer.q_proj(hidden_states), layer.k_proj(hidden_states)
    v = F.silu(layer.v_proj(hidden_states))
    q, k = (rearrange(value, "b t (h d) -> b t h d", d=layer.head_k_dim) for value in (q, k))
    v = rearrange(v, "b t (h d) -> b t h d", d=layer.head_v_dim)
    attention_output = torch.zeros_like(v)
    if layout.active:
        packed_q = q.flatten(0, 1).index_select(0, layout.indices).unsqueeze(0)
        packed_k = k.flatten(0, 1).index_select(0, layout.indices).unsqueeze(0)
        if layer.use_local_pos_encoding:
            # FLA 0.4.1 advances its global counter in layer 0. Later cached
            # layers therefore use n+1 in the native B1 path; preserve that
            # behavior rather than changing the original RoPE computation.
            native_advance = (
                past_key_values.get_seq_length(layer.layer_idx) - previous_seen
                if past_key_values is not None else 0
            )
            offsets = torch.tensor(
                [seen[index] + native_advance for index in layout.active], device=q.device, dtype=torch.long,
            )
            maximum = max(
                packed_q.shape[1],
                max(seen[index] + native_advance + layout.lengths[index] for index in layout.active),
            )
            if layer.max_position_embeddings is not None:
                maximum = max(maximum, layer.max_position_embeddings)
            packed_q, packed_k = layer.rotary(
                packed_q, packed_k, seqlen_offset=offsets,
                cu_seqlens=layout.cu_seqlens, max_seqlen=maximum,
            )
        packed_q, packed_k = packed_q.squeeze(0), packed_k.squeeze(0)
        packed_v = v.flatten(0, 1).index_select(0, layout.indices)
        keys, values, key_lengths = [], [], []
        cursor = 0
        for sample in layout.active:
            length = layout.lengths[sample]
            current_k, current_v = packed_k[cursor : cursor + length], packed_v[cursor : cursor + length]
            old = rows[sample]["attn_state"]
            if old is not None:
                old_k = old[0].reshape(-1, q.shape[2], layer.head_k_dim)
                old_v = old[1].reshape(-1, v.shape[2], layer.head_v_dim)
                current_k = torch.cat((old_k, current_k), dim=0)
                current_v = torch.cat((old_v, current_v), dim=0)
            keys.append(current_k)
            values.append(current_v)
            key_lengths.append(current_k.shape[0])
            keep = current_k.shape[0] if layer.chunk_size is None else min(current_k.shape[0], layer.chunk_size)
            rows[sample] = {
                **rows[sample], "attn_state": (
                    current_k[-keep:].reshape(1, keep, -1).clone(),
                    current_v[-keep:].reshape(1, keep, -1).clone(),
                ),
            }
            cursor += length
        if not layer._swiglu or layer.use_window_attention:
            counts = torch.tensor(key_lengths, device=q.device, dtype=torch.int32)
            cu_keys = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
            packed_output = flash_attn_varlen_func(
                packed_q.contiguous(), torch.cat(keys).contiguous(), torch.cat(values).contiguous(),
                cu_seqlens_q=layout.cu_seqlens, cu_seqlens_k=cu_keys,
                max_seqlen_q=max(layout.lengths), max_seqlen_k=max(key_lengths),
                causal=True,
                window_size=(-1, -1) if layer.chunk_size is None else (layer.chunk_size - 1, 0),
            )
            attention_output.flatten(0, 1).index_copy_(0, layout.indices, packed_output)

    q, k = F.silu(q), F.silu(k)
    q, k = (F.normalize(value, p=2, dim=-1, eps=1e-8) for value in (q, k))
    lr = F.sigmoid(layer.lr_proj(hidden_states).float()).unsqueeze(-1) * layer.base_lr
    if layer.use_ttt_decay:
        decay = F.sigmoid(layer.decay_proj(hidden_states).float()) * layer.base_decay
        log_decay = torch.log1p(-lr * decay.unsqueeze(-1))
    else:
        log_decay = torch.zeros_like(lr).float()
    log_momentum = F.logsigmoid(layer.momentum_proj(hidden_states).float()).unsqueeze(-1) / 16
    lr = lr * (1 / layer.ttt_chunk_size)
    output = None
    kernels = swiglu if layer._swiglu else mlp
    for sample in layout.active:
        positions = layout.positions[sample]
        inputs = [value[sample : sample + 1].index_select(1, positions) for value in (q, k, v, lr, log_decay, log_momentum)]
        recurrent = rows[sample]["recurrent_state"]
        options = {"use_post_norm": layer.use_post_norm} if layer._swiglu else {"use_grad_clip": layer.use_grad_clip}
        if recurrent is None:
            recurrent = (layer.w0, layer.w1, layer.w2) if layer._swiglu else (layer.w0, layer.w1)
            function = kernels.chunk_ttt_momentum
            options["is_training"] = False
        else:
            function = kernels.chunk_ttt_momentum_with_cache
        current_output, recurrent = function(
            *inputs, recurrent, layer.ttt_norm_weight, layer.ttt_norm_bias,
            chunk_size=layer.ttt_chunk_size, verbose=False,
            use_closed_form=layer.use_closed_form, **options,
        )
        if output is None:
            output = current_output.new_zeros(v.shape)
        output[sample].index_copy_(0, positions, current_output[0])
        rows[sample] = {**rows[sample], "recurrent_state": recurrent}
        seen[sample] += layout.lengths[sample]

    if output is None:
        # Both native kernels return the normalized query dtype, which may be
        # FP32 under autocast even though the value projection is BF16.
        output = q.new_zeros(v.shape)

    if layer.use_memory_gate:
        gate = rearrange(layer.memory_gate_proj(hidden_states), "b t (h d) -> b t h d", d=layer.head_v_dim).sigmoid()
        output = gate * output + (1 - gate) * attention_output
    elif not layer._swiglu or layer.use_window_attention:
        output = output + attention_output
    if past_key_values is not None:
        cached = past_key_values.update(layer_idx=layer.layer_idx, offset=width)
        cached["padding_state"] = {"seen": tuple(seen), "rows": rows}
    if layer.use_gate:
        gate = rearrange(layer.g_proj(hidden_states), "b t (h d) -> b t h d", d=layer.head_v_dim)
        output = layer.o_norm(output, gate)
    else:
        output = layer.o_norm(output)
    return layer.o_proj(rearrange(output, "b t h d -> b t (h d)"))
