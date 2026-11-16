"""左 padding 的真实位置、独立 KV 与原始 TTT 状态调度。"""

from dataclasses import dataclass

from einops import rearrange
from flash_attn import flash_attn_varlen_func
from lact_model.ttt_operation import l2_norm
import torch
import torch.nn.functional as F

from .ttt import TTTState


class PaddedLayerCache:
    def __init__(self, batch_size):
        self.seen = [0] * batch_size
        self.keys = [None] * batch_size
        self.values = [None] * batch_size
        self.ttt = [None] * batch_size


@dataclass
class PaddingStep:
    mask: torch.Tensor
    lengths: list[int]
    active: list[int]
    query_offsets: torch.Tensor
    position_offsets: torch.Tensor
    max_position: int


class PaddedLaCTCache:
    """物理列数仅用于 HF 输入切片；各层 seen 只计真实 token。"""

    def __init__(self, num_layers, batch_size):
        self.layers = [PaddedLayerCache(batch_size) for _ in range(num_layers)]
        self.batch_size = batch_size
        self.physical_seen = 0

    def get_seq_length(self, layer_idx=0):
        return self.physical_seen

    def prepare_step(self, input_ids, attention_mask):
        batch_size, width = input_ids.shape
        if batch_size != self.batch_size:
            raise ValueError("Input batch size differs from the existing cache")
        if attention_mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            if attention_mask.shape != (batch_size, self.physical_seen + width):
                raise ValueError("Padded attention mask must cover the complete physical prefix")
            mask = attention_mask[:, -width:].bool()
        lengths = mask.sum(-1).tolist()
        sizes = torch.tensor(lengths, device=input_ids.device, dtype=torch.long)
        expected = torch.arange(width, device=input_ids.device)[None] >= width - sizes[:, None]
        if not torch.equal(mask, expected):
            raise ValueError("Each new input must use contiguous left padding")
        if self.physical_seen == 0 and any(length == 0 for length in lengths):
            raise ValueError("Every prefill row must contain a real token")
        active = [row for row, length in enumerate(lengths) if length]
        offsets = [0]
        for row in active:
            offsets.append(offsets[-1] + lengths[row])
        seen = self.layers[0].seen
        # 左pad位置对应负索引，旋转核不读取；所有有效位置从各行真实seen开始。
        position_offsets = torch.tensor(
            [previous - (width - length) for previous, length in zip(seen, lengths, strict=True)],
            device=input_ids.device, dtype=torch.long,
        )
        return PaddingStep(
            mask=mask, lengths=lengths, active=active,
            query_offsets=torch.tensor(offsets, device=input_ids.device, dtype=torch.int32),
            position_offsets=position_offsets,
            max_position=max(width, max(previous + length for previous, length in zip(seen, lengths, strict=True))),
        )


def attention_forward_padded(layer, hidden_states, cache, step):
    batch_size, width, _ = hidden_states.shape
    q, k, v = layer.qkv(hidden_states).chunk(3, dim=-1)
    if layer.attn_qk_norm:
        q, k = layer.q_norm(q), layer.k_norm(k)
    fast_q, fast_k = layer._rescale_qk(q, k)
    fast_v = v
    q = rearrange(q, "b s (h d) -> b s h d", h=layer.num_heads)
    k = rearrange(k, "b s (h d) -> b s h d", h=layer.num_heads)
    v = rearrange(v, "b s (h d) -> b s h d", h=layer.num_heads)
    max_seqlen = max(step.max_position, layer.max_position_embeddings or 0)
    q, k = layer.rotary(q, k, seqlen_offset=step.position_offsets, max_seqlen=max_seqlen)
    attention = torch.zeros_like(v)
    keys, values, key_lengths = [], [], []
    key_offsets = [0]
    for row in step.active:
        selected_k = k[row:row + 1, step.mask[row]]
        selected_v = v[row:row + 1, step.mask[row]]
        if cache.keys[row] is not None:
            selected_k = torch.cat((cache.keys[row], selected_k), dim=1)
            selected_v = torch.cat((cache.values[row], selected_v), dim=1)
        key_length = selected_k.shape[1]
        keys.append(selected_k[0])
        values.append(selected_v[0])
        key_lengths.append(key_length)
        key_offsets.append(key_offsets[-1] + key_length)
        keep = key_length if layer.window_size is None else min(key_length, layer.window_size - 1)
        cache.keys[row] = selected_k[:, key_length - keep:].clone()
        cache.values[row] = selected_v[:, key_length - keep:].clone()
    if step.active:
        attention[step.mask] = flash_attn_varlen_func(
            q[step.mask], torch.cat(keys), torch.cat(values),
            cu_seqlens_q=step.query_offsets,
            cu_seqlens_k=torch.tensor(key_offsets, device=q.device, dtype=torch.int32),
            max_seqlen_q=max(step.lengths), max_seqlen_k=max(key_lengths),
            causal=True,
            window_size=(-1, -1) if layer.window_size is None else (layer.window_size - 1, 0),
        )
    fast_q = rearrange(fast_q, "b s (h d) -> (b h) s d", h=layer.num_fw_heads)
    fast_k = rearrange(fast_k, "b s (h d) -> (b h) s d", h=layer.num_fw_heads)
    fast_v = rearrange(fast_v, "b s (h d) -> (b h) s d", h=layer.num_fw_heads)
    if layer.qkv_silu:
        fast_q, fast_k = F.silu(fast_q), F.silu(fast_k)
        if not layer.no_v_silu:
            fast_v = F.silu(fast_v)
    fast_q, fast_k = l2_norm(fast_q), l2_norm(fast_k)
    if not layer.ttt_nope:
        fast_q = rearrange(fast_q, "(b h) s d -> b s h d", b=batch_size, h=layer.num_fw_heads)
        fast_k = rearrange(fast_k, "(b h) s d -> b s h d", b=batch_size, h=layer.num_fw_heads)
        fast_q = fast_q.reshape(batch_size, width, layer.num_heads, layer.head_dim)
        fast_k = fast_k.reshape(batch_size, width, layer.num_heads, layer.head_dim)
        fast_q, fast_k = layer.rotary(
            fast_q, fast_k, seqlen_offset=step.position_offsets, max_seqlen=max_seqlen,
        )
        fast_q = rearrange(fast_q, "b s h d -> b s (h d)")
        fast_k = rearrange(fast_k, "b s h d -> b s (h d)")
        fast_q = rearrange(fast_q, "b s (h d) -> (b h) s d", h=layer.num_fw_heads)
        fast_k = rearrange(fast_k, "b s (h d) -> (b h) s d", h=layer.num_fw_heads)
    fast_q = rearrange(fast_q, "(b h) s d -> b s h d", b=batch_size, h=layer.num_fw_heads)
    fast_k = rearrange(fast_k, "(b h) s d -> b s h d", b=batch_size, h=layer.num_fw_heads)
    fast_v = rearrange(fast_v, "(b h) s d -> b s h d", b=batch_size, h=layer.num_fw_heads)
    if layer.lr_parameterization != "mamba":
        raise ValueError("Optimized inference requires the checkpoint's mamba LR rule")
    learning_rates = F.softplus(layer.lr_proj(hidden_states).float() + layer.base_lr_inv)
    learning_rates = rearrange(learning_rates, "b s (h d) -> b s h d", h=layer.num_fw_heads)
    lr0, lr1, lr2 = learning_rates.chunk(3, dim=-1)
    momentum = layer.momentum_proj(hidden_states).float() if layer.use_momentum else None
    fast_output = torch.zeros_like(fast_v)
    if any(cache.ttt[row] is None for row in step.active):
        w0 = layer.w0() if layer.w0_w2_low_rank > 0 else layer.w0
        w2 = layer.w2() if layer.w0_w2_low_rank > 0 else layer.w2
        w1 = layer.w1
        if layer.fp32_states:
            w0, w1, w2 = w0.float(), w1.float(), w2.float()
    for row in step.active:
        if cache.ttt[row] is None:
            cache.ttt[row] = TTTState(
                w0, w1, w2, layer.lact_chunk_size, layer.ttt_prenorm, layer.use_muon, layer.use_momentum,
            )
        valid = step.mask[row]
        result = cache.ttt[row].forward(
            fast_q[row, valid].transpose(0, 1),
            fast_k[row, valid].transpose(0, 1),
            fast_v[row, valid].transpose(0, 1),
            lr0[row, valid].transpose(0, 1),
            lr1[row, valid].transpose(0, 1),
            lr2[row, valid].transpose(0, 1),
            momentum[row, valid].transpose(0, 1).unsqueeze(-1) if momentum is not None else None,
        )
        fast_output[row, valid] = result.transpose(0, 1)
        cache.seen[row] += step.lengths[row]
    fast_output = rearrange(fast_output, "b s h d -> (b h) s d")
    fast_output = layer.ttt_norm(fast_output)
    fast_output = rearrange(fast_output, "(b h) s d -> b s h d", b=batch_size, h=layer.num_fw_heads)
    if layer.learnable_ttt_scale:
        scale = F.silu(layer.ttt_scale_proj(hidden_states), inplace=False)
        fast_output = fast_output * scale.unsqueeze(-1)
    return layer.o_proj(attention.reshape(batch_size, width, -1) + fast_output.reshape(batch_size, width, -1))
