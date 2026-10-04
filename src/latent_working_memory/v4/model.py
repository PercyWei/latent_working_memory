"""冻结 causal LM 上的逐层 Q/O 修正与按边界写入。

position_ids 保持绝对位置，cache_position 使用当前物理缓存索引；删除前缀后
不重新编码 retained KV。训练与推理共享同一分块及写入路径。
"""

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
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
                attn_implementation="eager",
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
        self.backbone.set_attn_implementation("eager")
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

    def initial_state(self, differentiable=False):
        return StreamState(
            memories=[m.initial_state(1, differentiable) for m in self.memory_layers],
            sources=[None] * len(self.memory_layers),
            cache=DynamicCache(),
        )

    def _write(self, state, differentiable):
        count = self.config.pending_size
        losses = []
        for index, memory in enumerate(self.memory_layers):
            source = state.sources[index]
            updated, metrics = memory.write(
                state.memories[index], source[:, :count], differentiable
            )
            state.memories[index] = updated
            # Own the retained storage so inference can release the evicted prefix.
            state.sources[index] = source[:, count:].contiguous()
            losses.append(metrics["write_loss_after"])
        for layer in state.cache.layers:
            layer.keys = layer.keys[..., count:, :].contiguous()
            layer.values = layer.values[..., count:, :].contiguous()
        state.write_events += 1
        state.write_loss_sum += sum(losses) / len(losses)

    def _chunks(self, input_ids, state, differentiable):
        if input_ids.ndim != 1 or input_ids.dtype != torch.long or input_ids.numel() == 0:
            raise ValueError("input_ids must be a nonempty one-dimensional torch.long tensor")
        if state.position + len(input_ids) > self.backbone.config.max_position_embeddings:
            raise ValueError("stream exceeds the backbone absolute position limit")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("TTT requires autograd; use torch.no_grad(), not inference_mode()")
        start = 0
        while start < len(input_ids):
            live = state.cache.get_seq_length()
            length = min(
                len(input_ids) - start,
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
                    output = self.backbone(
                        input_ids=input_ids[start : start + length].unsqueeze(0),
                        position_ids=positions.unsqueeze(0),
                        cache_position=torch.arange(live, live + length, device=input_ids.device),
                        attention_mask=torch.ones(
                            (1, live + length), dtype=torch.long, device=input_ids.device
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
                self._write(state, differentiable)
            yield start, output.logits[0], was_written
            start += length

    def consume(self, input_ids, state=None, differentiable=False):
        """推进一条流；调用者可在下一次调用继续传入返回的 state。"""
        if state is None:
            state = self.initial_state(differentiable)
        logits = [value for _, value, _ in self._chunks(input_ids, state, differentiable)]
        return torch.cat(logits), state

    def forward(self, input_ids, differentiable=True):
        """外层 LM 目标只使用写入生效后产生的 logits；不加写入损失。"""
        minimum = self.config.pending_size + self.config.recent_size + 2
        if len(input_ids) < minimum:
            raise ValueError(f"a training episode needs at least {minimum} tokens")
        state = self.initial_state(differentiable)
        losses = []
        target_tokens = 0
        for start, logits, was_written in self._chunks(input_ids[:-1], state, differentiable):
            if was_written:
                targets = input_ids[start + 1 : start + 1 + len(logits)]
                with torch.set_grad_enabled(differentiable):
                    scores = (
                        logits.float()
                        if logits.dtype in {torch.float16, torch.bfloat16}
                        else logits
                    )
                    losses.append(F.cross_entropy(scores, targets, reduction="sum"))
                target_tokens += len(targets)
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
        logits, state = self.consume(input_ids)
        generated = []
        for step in range(max_new_tokens):
            token = logits[-1].argmax().reshape(1)
            generated.append(token)
            if token.item() == eos_token_id or step + 1 == max_new_tokens:
                break
            logits, state = self.consume(token, state)
        return torch.cat((input_ids, *generated))

    def trainable_state_dict(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.memory_layers.state_dict().items()
        }

    def load_trainable_state_dict(self, state):
        self.memory_layers.load_state_dict(state, strict=True)
