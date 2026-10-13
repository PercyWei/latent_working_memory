"""冻结 causal LM 上的逐层 Q/O 修正与按边界写入。

position_ids 保持绝对位置，cache_position 使用当前物理缓存索引；删除前缀后
不重新编码 retained KV。训练与推理共享同一分块及写入路径。
"""

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, DynamicCache

from latent_working_memory.v4.memory import LayerMemory


@dataclass
class StreamState:
    memories: list[torch.Tensor]
    sources: list[torch.Tensor | None]
    cache: DynamicCache
    position: int = 0
    write_events: int = 0
    write_loss_sum: torch.Tensor | float = 0.0
    peak_live_tokens: int = 0


class StreamingMemoryLM(nn.Module):
    def __init__(self, config, backbone=None):
        super().__init__()
        self.config = config
        self.backbone = (
            backbone
            if backbone is not None
            else AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                revision=config.revision,
                dtype=getattr(torch, config.backbone_dtype),
                attn_implementation=config.attention_implementation,
            )
        )
        base = self.backbone.config
        if base.model_type not in {"llama", "qwen2", "qwen3"}:
            raise ValueError("v4 supports full-attention Llama, Qwen2 and Qwen3 causal LMs")
        if getattr(base, "use_sliding_window", False) or any(
            kind != "full_attention" for kind in (getattr(base, "layer_types", None) or [])
        ):
            raise ValueError("native sliding/hybrid attention is not supported by the v4 cache")
        if config.pending_size + config.recent_size >= base.max_position_embeddings:
            raise ValueError("the first write must occur before the backbone position limit")
        self.backbone.set_attn_implementation(config.attention_implementation)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.memory_layers = nn.ModuleList(
            LayerMemory(base.hidden_size, layer.self_attn.q_proj.out_features, config)
            for layer in self.backbone.model.layers
        )
        self._active_state = None
        self._source_chunks = {}
        self._corrections = {}
        for index, layer in enumerate(self.backbone.model.layers):
            attention = layer.self_attn
            attention.attention_dropout = 0.0
            attention.register_forward_pre_hook(
                partial(self._before_attention, index), with_kwargs=True
            )
            attention.q_proj.register_forward_hook(partial(self._correct_query, index))
            attention.o_proj.register_forward_hook(partial(self._correct_output, index))

    def train(self, mode=True):
        super().train(mode)
        # Frozen backbone still participates in autograd, but its dropout stays disabled.
        self.backbone.eval()
        return self

    def _before_attention(self, index, module, args, kwargs):
        if self._active_state is None:
            return
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        memory = self.memory_layers[index]
        self._source_chunks[index] = memory.source(hidden)
        if self._active_state.write_events:
            self._corrections[index] = memory.corrections(
                hidden, self._active_state.memories[index]
            )

    def _correct_query(self, index, module, args, output):
        # q_proj precedes Qwen3 q_norm and the backbone's own RoPE application.
        if index in self._corrections:
            return output + self._corrections[index][0]
        return output

    def _correct_output(self, index, module, args, output):
        # o_proj is after attention/W_O and before the decoder residual addition.
        if index in self._corrections:
            return output + self._corrections[index][1]
        return output

    def initial_state(self, differentiable=False, batch_size=1):
        return StreamState(
            memories=[m.initial_state(batch_size, differentiable) for m in self.memory_layers],
            sources=[None] * len(self.memory_layers),
            cache=DynamicCache(),
        )

    def _write(self, state, differentiable, active_rows=None):
        count = self.config.pending_size
        rows = (
            None
            if active_rows is None
            else torch.tensor(active_rows, dtype=torch.long, device=state.memories[0].device)
        )
        num_events = state.memories[0].shape[0] if rows is None else len(active_rows)
        losses = []
        for index, memory in enumerate(self.memory_layers):
            source = state.sources[index]
            updated, metrics = memory.write(
                state.memories[index] if rows is None else state.memories[index][rows],
                source[:, :count] if rows is None else source[rows, :count],
                differentiable,
            )
            state.memories[index] = (
                updated if rows is None else state.memories[index].index_copy(0, rows, updated)
            )
            # clone also owns storage when a batch-one slice is already contiguous.
            state.sources[index] = source[:, count:].clone(memory_format=torch.contiguous_format)
            losses.append(metrics["write_loss_after"])
        for layer in state.cache.layers:
            layer.keys = layer.keys[..., count:, :].clone(memory_format=torch.contiguous_format)
            layer.values = layer.values[..., count:, :].clone(memory_format=torch.contiguous_format)
        state.write_events += num_events
        state.write_loss_sum += sum(losses) / len(losses) * num_events

    def _chunks(self, input_ids, state, differentiable, lengths):
        """Advance right-padded rows together; lengths exclude padding in this call."""
        batch_size, total_length = input_ids.shape
        end_positions = [state.position + length for length in lengths]
        ends = torch.tensor(end_positions, device=input_ids.device)
        if state.position + total_length > self.backbone.config.max_position_embeddings:
            raise ValueError("stream exceeds the backbone absolute position limit")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("TTT requires autograd; use torch.no_grad(), not inference_mode()")
        start = 0
        while start < total_length:
            live = state.cache.get_seq_length()
            length = min(
                total_length - start,
                self.config.pending_size + self.config.recent_size - live,
            )
            was_written = state.write_events > 0
            positions = torch.arange(
                state.position, state.position + length, device=input_ids.device
            )
            self._active_state = state
            self._source_chunks = {}
            self._corrections = {}
            try:
                with torch.set_grad_enabled(differentiable):
                    output = self.backbone.model(
                        input_ids=input_ids[:, start : start + length],
                        position_ids=positions.unsqueeze(0).expand(batch_size, -1),
                        cache_position=torch.arange(live, live + length, device=input_ids.device),
                        attention_mask=(
                            torch.arange(
                                state.position - live,
                                state.position + length,
                                device=input_ids.device,
                            )[None]
                            < ends[:, None]
                        ),
                        past_key_values=state.cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    for index, source in self._source_chunks.items():
                        old = state.sources[index]
                        state.sources[index] = (
                            source if old is None else torch.cat((old, source), dim=1)
                        )
            finally:
                self._active_state = None
                self._source_chunks = {}
                self._corrections = {}
            state.position += length
            state.peak_live_tokens = max(state.peak_live_tokens, live + length)
            if live + length == self.config.pending_size + self.config.recent_size:
                active_rows = [i for i, end in enumerate(end_positions) if end >= state.position]
                if len(active_rows) == batch_size:
                    self._write(state, differentiable)
                else:
                    self._write(state, differentiable, active_rows)
            yield start, output.last_hidden_state, was_written
            start += length

    def consume(self, input_ids, state=None, differentiable=False, last_only=False):
        """推进一条流；last_only 只计算最后位置的 logits，用于生成。"""
        if input_ids.ndim != 1 or input_ids.dtype != torch.long or input_ids.numel() == 0:
            raise ValueError("input_ids must be a nonempty one-dimensional torch.long tensor")
        if state is None:
            state = self.initial_state(differentiable)
        pieces = []
        for _, hidden, _ in self._chunks(
            input_ids.unsqueeze(0), state, differentiable, [len(input_ids)]
        ):
            if not last_only:
                with torch.set_grad_enabled(differentiable):
                    pieces.append(self.backbone.get_output_embeddings()(hidden[0]))
        with torch.set_grad_enabled(differentiable):
            logits = (
                self.backbone.get_output_embeddings()(hidden[0, -1:])
                if last_only
                else torch.cat(pieces)
            )
        return logits, state

    def _token_loss(self, hidden, targets):
        logits = self.backbone.get_output_embeddings()(hidden)
        scores = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
        return F.cross_entropy(scores, targets, reduction="sum")

    def forward(self, input_ids, differentiable=True, target_starts=None):
        """真实 microbatch；监督起点与写入后有效位置相交，按 target tokens 平均。"""
        rows = [input_ids] if isinstance(input_ids, torch.Tensor) else input_ids
        if not isinstance(rows, list) or not rows:
            raise ValueError("input_ids must be a tensor or a nonempty list of tensors")
        minimum = self.config.pending_size + self.config.recent_size + 2
        if target_starts is None:
            target_starts = [0] * len(rows)
        if len(target_starts) != len(rows):
            raise ValueError("target_starts must contain one supervision start per input row")
        for row, target_start in zip(rows, target_starts, strict=True):
            if not isinstance(row, torch.Tensor) or row.ndim != 1 or row.dtype != torch.long:
                raise ValueError("each input row must be a one-dimensional torch.long tensor")
            if len(row) < minimum:
                raise ValueError(f"a training episode needs at least {minimum} tokens")
            if type(target_start) is not int or not 0 <= target_start < len(row):
                raise ValueError("target_start must be a token index within its episode")
        padded = pad_sequence(rows, batch_first=True, padding_value=0)
        lengths = [len(row) - 1 for row in rows]
        valid_lengths = torch.tensor(lengths, device=padded.device)
        supervision_starts = torch.tensor(target_starts, device=padded.device)
        state = self.initial_state(differentiable, len(rows))
        losses = []
        target_tokens = 0
        for start, hidden, was_written in self._chunks(
            padded[:, :-1], state, differentiable, lengths
        ):
            if was_written:
                length = hidden.shape[1]
                valid = (
                    torch.arange(start, start + length, device=padded.device)[None]
                    < valid_lengths[:, None]
                )
                valid &= (
                    torch.arange(start + 1, start + 1 + length, device=padded.device)[None]
                    >= supervision_starts[:, None]
                )
                with torch.set_grad_enabled(differentiable):
                    selected = hidden[valid]
                    targets = padded[:, start + 1 : start + 1 + length][valid]
                    for offset in range(0, len(targets), self.config.loss_chunk_size):
                        chunk = selected[offset : offset + self.config.loss_chunk_size]
                        target = targets[offset : offset + self.config.loss_chunk_size]
                        losses.append(
                            checkpoint(self._token_loss, chunk, target, use_reentrant=False)
                            if differentiable and chunk.requires_grad
                            else self._token_loss(chunk, target)
                        )
                target_tokens += len(targets)
        with torch.set_grad_enabled(differentiable):
            loss = torch.stack(losses).sum() / target_tokens
        return {
            "loss": loss,
            "target_tokens": target_tokens,
            "write_events": state.write_events,
            "write_loss": float(state.write_loss_sum / state.write_events),
            "peak_live_tokens": state.peak_live_tokens,
        }

    def generate(self, input_ids, max_new_tokens, eos_token_id=None):
        """Greedy 生成；每条调用从独立 memory 开始，在线仍执行局部 TTT。"""
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if len(input_ids) + max_new_tokens > self.backbone.config.max_position_embeddings:
            raise ValueError("generation exceeds the backbone absolute position limit")
        logits, state = self.consume(input_ids, last_only=True)
        generated = []
        for step in range(max_new_tokens):
            token = logits[-1].argmax().reshape(1)
            generated.append(token)
            if token.item() == eos_token_id or step + 1 == max_new_tokens:
                break
            logits, state = self.consume(token, state, last_only=True)
        return torch.cat((input_ids, *generated))

    def trainable_state_dict(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.memory_layers.state_dict().items()
        }

    def load_trainable_state_dict(self, state):
        self.memory_layers.load_state_dict(state, strict=True)
