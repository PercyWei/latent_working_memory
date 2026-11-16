"""左 padding 的真实位置、独立 KV 与原始 TTT 状态调度。"""

from dataclasses import dataclass

from einops import rearrange
from flash_attn import flash_attn_varlen_func
from lact_model.ttt_operation import l2_norm
import torch
import torch.nn.functional as F

from .ttt import BatchedTTTState


class PaddedLayerCache:
    def __init__(self, batch_size):
        self.seen = [0] * batch_size
        self.keys = [None] * batch_size
        self.values = [None] * batch_size
        self.ttt = None
        self.key_lengths = None
        self.key_offsets = None


@dataclass
class PaddingStep:
    width: int
    lengths: list[int]
    length_tensor: torch.Tensor | None
    token_indices: torch.Tensor | None
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
        self._next_positions = None
        self._layout = None
        self._generated_mask = None
        self._generated_mask_version = None

    def get_seq_length(self, layer_idx=0):
        return self.physical_seen

    def mark_generation_mask(self, mask, num_new_tokens):
        # HF 的标准生成在此处追加一列 1。只信任未被原地改写的同一个张量。
        self._generated_mask = None
        self._generated_mask_version = None
        if (mask is not None and num_new_tokens == 1 and not torch.is_inference(mask)
                and mask.shape == (self.batch_size, self.physical_seen + 1)):
            self._generated_mask = mask
            self._generated_mask_version = mask._version

    def prepare_step(self, input_ids, attention_mask):
        batch_size, width = input_ids.shape
        if batch_size != self.batch_size:
            raise ValueError("Input batch size differs from the existing cache")
        if attention_mask is None:
            lengths = [width] * batch_size
        else:
            if attention_mask.shape != (batch_size, self.physical_seen + width):
                raise ValueError("Padded attention mask must cover the complete physical prefix")
            trusted_decode = (width == 1 and attention_mask is self._generated_mask
                              and attention_mask._version == self._generated_mask_version)
            if trusted_decode:
                lengths = [1] * batch_size
            else:
                # 通用forward仍尊重外部mask；一次CPU解析后，各层只用整数索引。
                mask = attention_mask[:, -width:].detach().to(device="cpu", dtype=torch.bool)
                lengths = mask.sum(-1).tolist()
                expected = torch.arange(width, device="cpu")[None] >= width - torch.tensor(lengths, device="cpu")[:, None]
                if not torch.equal(mask, expected):
                    raise ValueError("Each new input must use contiguous left padding")
        self._generated_mask = None
        self._generated_mask_version = None
        if self.physical_seen == 0 and any(length == 0 for length in lengths):
            raise ValueError("Every prefill row must contain a real token")
        signature = (width, tuple(lengths))
        if self._layout is None or self._layout[0] != signature:
            active = [row for row, length in enumerate(lengths) if length]
            offsets = [0]
            for row in active:
                offsets.append(offsets[-1] + lengths[row])
            full = all(length == width for length in lengths)
            indices = None if full else torch.cat([
                torch.arange(row * width + width - lengths[row], (row + 1) * width, device="cpu")
                for row in active
            ]) if active else torch.empty(0, dtype=torch.long, device="cpu")
            if indices is not None:
                indices = indices.to(input_ids.device, non_blocking=True)
            query_offsets = torch.tensor(offsets, dtype=torch.int32, device="cpu").to(input_ids.device, non_blocking=True)
            length_tensor = None if full else torch.tensor(lengths, device="cpu").to(input_ids.device, non_blocking=True)
            self._layout = (signature, active, indices, query_offsets, length_tensor)
        _, active, indices, query_offsets, length_tensor = self._layout
        if self._next_positions is None:
            self._next_positions = torch.zeros(batch_size, device=input_ids.device, dtype=torch.long)
        position_offsets = self._next_positions if length_tensor is None else self._next_positions - (width - length_tensor)
        seen = self.layers[0].seen
        return PaddingStep(
            width=width, lengths=lengths, length_tensor=length_tensor,
            token_indices=indices, active=active, query_offsets=query_offsets,
            position_offsets=position_offsets,
            max_position=max(width, max(previous + length for previous, length in zip(seen, lengths, strict=True))),
        )

    def advance(self, step):
        self.physical_seen += step.width
        self._next_positions.add_(step.width if step.length_tensor is None else step.length_tensor)


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
    keys, values, key_lengths = [], [], []
    key_offsets = [0]
    for row in step.active:
        start = width - step.lengths[row]
        selected_k = k[row:row + 1, start:]
        selected_v = v[row:row + 1, start:]
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
    attention = v.new_zeros(v.shape)
    if step.active:
        signature = tuple(key_lengths)
        if cache.key_lengths != signature:
            cache.key_lengths = signature
            cache.key_offsets = torch.tensor(key_offsets, dtype=torch.int32).to(q.device, non_blocking=True)
        queries = q.reshape(batch_size * width, layer.num_heads, layer.head_dim)
        if step.token_indices is not None:
            queries = queries.index_select(0, step.token_indices)
        packed = flash_attn_varlen_func(
            queries, torch.cat(keys), torch.cat(values),
            cu_seqlens_q=step.query_offsets, cu_seqlens_k=cache.key_offsets,
            max_seqlen_q=max(step.lengths), max_seqlen_k=max(key_lengths),
            causal=True,
            window_size=(-1, -1) if layer.window_size is None else (layer.window_size - 1, 0),
        )
        if step.token_indices is None:
            attention = packed.reshape_as(v)
        else:
            attention.reshape(batch_size * width, layer.num_heads, layer.head_dim).index_copy_(0, step.token_indices, packed)
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
    if cache.ttt is None:
        w0 = layer.w0() if layer.w0_w2_low_rank > 0 else layer.w0
        w2 = layer.w2() if layer.w0_w2_low_rank > 0 else layer.w2
        w1 = layer.w1
        if layer.fp32_states:
            w0, w1, w2 = w0.float(), w1.float(), w2.float()
        cache.ttt = BatchedTTTState(
            w0, w1, w2, batch_size, layer.lact_chunk_size,
            layer.ttt_prenorm, layer.use_muon, layer.use_momentum,
        )
        cache.seen = cache.ttt.seen
    fast_output = cache.ttt.forward(fast_q, fast_k, fast_v, lr0, lr1, lr2, momentum, step.lengths)
    fast_output = rearrange(fast_output, "b s h d -> (b h) s d")
    fast_output = layer.ttt_norm(fast_output)
    fast_output = rearrange(fast_output, "(b h) s d -> b s h d", b=batch_size, h=layer.num_fw_heads)
    if layer.learnable_ttt_scale:
        scale = F.silu(layer.ttt_scale_proj(hidden_states), inplace=False)
        fast_output = fast_output * scale.unsqueeze(-1)
    return layer.o_proj(attention.reshape(batch_size, width, -1) + fast_output.reshape(batch_size, width, -1))
