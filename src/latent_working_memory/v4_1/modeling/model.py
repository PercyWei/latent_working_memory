"""Qwen3 上的在线 TTT：正文触发写入，QA 阶段固定 memory。"""

from dataclasses import dataclass, field
from functools import partial

from peft import LoraConfig, inject_adapter_in_model
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

from .am import local_read, matching_loss
from .attention import decoder_forward
from .memory import LayerMemory


@dataclass
class StreamState:
    memories: list[torch.Tensor]
    memory_kv: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    recent_queries: list[torch.Tensor]
    position: int = 0
    compressed_tokens: int = 0
    write_events: int = 0
    last_hidden: torch.Tensor | None = None
    inner_before: list[torch.Tensor] = field(default_factory=list)
    inner_after: list[torch.Tensor] = field(default_factory=list)


class OnlineMemoryLM(nn.Module):
    def __init__(self, config, backbone=None):
        super().__init__()
        self.config = config
        self.backbone = backbone if backbone is not None else AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, dtype=getattr(torch, config.backbone_dtype),
            attn_implementation="eager",
        )
        base = self.backbone.config
        if base.model_type != "qwen3" or any(
            kind != "full_attention" for kind in base.layer_types
        ):
            raise ValueError("v4.1 currently supports full-attention Qwen3 only")
        rope_type = (base.rope_scaling or {}).get("rope_type", "default")
        if rope_type != "default":
            raise ValueError("v4.1 uses fixed default RoPE; scaled/dynamic RoPE is not supported")
        if config.pending_size + config.recent_size >= base.max_position_embeddings:
            raise ValueError("first compression boundary must precede the model position limit")
        self.backbone.requires_grad_(False)
        self.backbone = inject_adapter_in_model(
            LoraConfig(
                r=config.lora_rank, lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout, target_modules=["q_proj", "k_proj", "v_proj"],
                bias="none",
            ),
            self.backbone,
        )
        for parameter in self.backbone.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.memory_layers = nn.ModuleList(
            LayerMemory(config, base.num_key_value_heads, layer.self_attn.head_dim)
            for layer in self.backbone.model.layers
        )
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # 冻结基座和 LoRA 的随机 dropout，但允许梯度穿过完整前向。
        self.backbone.eval()
        return self

    @property
    def device(self):
        return self.backbone.model.embed_tokens.weight.device

    def trainable_state_dict(self):
        return {name: parameter.detach().cpu().clone() for name, parameter in self.named_parameters()
                if parameter.requires_grad}

    def load_trainable_state_dict(self, state):
        parameters = {name: parameter for name, parameter in self.named_parameters()
                      if parameter.requires_grad}
        if parameters.keys() != state.keys():
            raise ValueError("checkpoint trainable parameter names differ from the model")
        with torch.no_grad():
            for name, parameter in parameters.items():
                if parameter.shape != state[name].shape:
                    raise ValueError(f"checkpoint shape mismatch: {name}")
                parameter.copy_(state[name])

    def initial_state(self, differentiable=False):
        dtype = self.backbone.model.embed_tokens.weight.dtype
        memories, memory_kv, keys, values, queries = [], [], [], [], []
        for memory, layer in zip(self.memory_layers, self.backbone.model.layers):
            memories.append(memory.initial_slots if differentiable else memory.initial_slots.detach())
            empty = torch.empty(
                (memory.num_kv_heads, 0, memory.head_dim), device=self.device, dtype=dtype
            )
            memory_kv.append((empty, empty, empty.new_empty((memory.num_kv_heads, 0))))
            keys.append(empty.unsqueeze(0))
            values.append(empty.unsqueeze(0))
            queries.append(empty.new_empty((1, layer.self_attn.q_proj.out_features // memory.head_dim,
                                            0, memory.head_dim)))
        return StreamState(memories, memory_kv, keys, values, queries)

    def _advance(self, token_ids, state, differentiable):
        """纯 layer checkpoint 不读写 StreamState，避免重算时重复追加缓存。"""
        with torch.set_grad_enabled(differentiable):
            ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)[None]
            hidden = self.backbone.model.embed_tokens(ids)
            positions = torch.arange(state.position, state.position + len(token_ids),
                                     device=self.device)[None]
            cosine, sine = self.backbone.model.rotary_emb(hidden, positions)
            for index, layer in enumerate(self.backbone.model.layers):
                args = (hidden, state.keys[index], state.values[index],
                        *state.memory_kv[index], cosine, sine)
                function = partial(decoder_forward, layer)
                if differentiable and self.config.gradient_checkpointing:
                    hidden, keys, values, query = checkpoint(function, *args, use_reentrant=False)
                else:
                    hidden, keys, values, query = function(*args)
                state.keys[index], state.values[index] = keys, values
                state.recent_queries[index] = torch.cat(
                    (state.recent_queries[index], query), dim=2
                )[:, :, -self.config.recent_size:].clone()
            hidden = self.backbone.model.norm(hidden)
            state.position += len(token_ids)
            state.last_hidden = hidden[:, -1]
            return hidden

    def _write(self, state, differentiable):
        count = self.config.pending_size
        compressed = state.compressed_tokens + count
        rotary = self.backbone.model.rotary_emb
        before, after = [], []
        with torch.set_grad_enabled(differentiable):
            for index, memory in enumerate(self.memory_layers):
                old_key, old_value, old_bias = state.memory_kv[index]
                pending_key = state.keys[index][0, :, :count].float()
                pending_value = state.values[index][0, :, :count].float()
                teacher = (
                    torch.cat((old_key.float(), pending_key), dim=1),
                    torch.cat((old_value.float(), pending_value), dim=1),
                    torch.cat((old_bias.float(), old_bias.new_zeros((old_bias.shape[0], count))), -1),
                )
                query = state.recent_queries[index][0].transpose(0, 1).float()
                updated, metrics = memory.write(
                    state.memories[index], teacher, query, compressed_tokens=compressed,
                    rotary=rotary, differentiable=differentiable,
                )
                state.memories[index] = updated
                state.memory_kv[index] = memory.decode(updated, compressed, rotary)
                # clone 释放被删除 pending 的存储；recent KV 保留原值及全局位置。
                state.keys[index] = state.keys[index][:, :, count:].clone()
                state.values[index] = state.values[index][:, :, count:].clone()
                before.append(metrics[0])
                after.append(metrics[1])
        state.compressed_tokens = compressed
        state.write_events += 1
        state.inner_before.append(torch.stack(before).mean())
        state.inner_after.append(torch.stack(after).mean())

    def _outer_am(self, state, teacher, generator):
        reference = teacher.reference(
            state.write_events, state.position, state.compressed_tokens, generator
        )
        losses = []
        for index, kv in enumerate(state.memory_kv):
            outputs, mass = local_read(reference["queries"][index], *kv)
            losses.append(matching_loss(
                outputs, mass, reference["outputs"][index], reference["log_mass"][index],
                reference["scales_o"][index], reference["scales_z"][index],
            ))
        return torch.stack(losses).mean()

    def consume(self, token_ids, state, differentiable=False, write=True, on_write=None):
        """返回按 chunk 排列的 hidden states；压缩仅发生在完整前向之后。"""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("online TTT needs autograd; use no_grad(), not inference_mode()")
        if state.position + len(token_ids) > self.backbone.config.max_position_embeddings:
            raise ValueError("input exceeds the backbone global position limit")
        start = 0
        while start < len(token_ids):
            length = min(self.config.prefill_chunk_size, len(token_ids) - start)
            if write:
                capacity = self.config.pending_size + self.config.recent_size
                length = min(length, capacity - state.keys[0].shape[2])
            hidden = self._advance(token_ids[start:start + length], state, differentiable)
            start += length
            if write and state.keys[0].shape[2] == capacity:
                self._write(state, differentiable)
                if on_write is not None:
                    on_write(state)
            yield hidden

    def _ingest(self, context_ids, differentiable, on_write=None):
        state = self.initial_state(differentiable)
        for _ in self.consume(context_ids, state, differentiable, on_write=on_write):
            pass
        return state

    def _diagnostics(self, state):
        return {
            "write_events": state.write_events,
            "inner_before": float(torch.stack(state.inner_before).mean()) if state.inner_before else 0.0,
            "inner_after": float(torch.stack(state.inner_after).mean()) if state.inner_after else 0.0,
        }

    def forward(self, episode, teacher=None, lambda_am=1.0,
                generator=None, differentiable=True):
        if not episode.context_ids or not episode.target_ids:
            raise ValueError("a QA episode must have nonempty context and target")
        with torch.set_grad_enabled(differentiable):
            outer_losses = []

            def supervise(state):
                if teacher is not None:
                    outer_losses.append(self._outer_am(state, teacher, generator))

            state = self._ingest(episode.context_ids, differentiable, supervise)
            for _ in self.consume(episode.question_ids, state, differentiable, write=False):
                pass
            # target[0] 由问题末尾预测；其后每个 target token 预测下一个 token。
            predictors = [state.last_hidden[:, None, :]]
            predictors.extend(self.consume(episode.target_ids[:-1], state, differentiable, write=False))
            targets = torch.tensor(episode.target_ids, device=self.device, dtype=torch.long)
            losses, offset = [], 0
            for hidden in predictors:
                for chunk in hidden.split(self.config.loss_chunk_size, dim=1):
                    labels = targets[offset:offset + chunk.shape[1]]
                    if differentiable and self.config.gradient_checkpointing:
                        loss = checkpoint(self._answer_loss, chunk, labels, use_reentrant=False)
                    else:
                        loss = self._answer_loss(chunk, labels)
                    losses.append(loss)
                    offset += chunk.shape[1]
            qa_loss = torch.stack(losses).sum() / len(episode.target_ids)
            outer_am = torch.stack(outer_losses).mean() if outer_losses else qa_loss.new_zeros(())
            return {
                "loss": qa_loss + lambda_am * outer_am,
                "qa_loss": qa_loss, "outer_am": outer_am,
                "target_tokens": len(episode.target_ids), **self._diagnostics(state),
            }

    def _answer_loss(self, hidden, labels):
        logits = self.backbone.lm_head(hidden).float().squeeze(0)
        return F.cross_entropy(logits, labels, reduction="sum")

    @torch.no_grad()
    def generate(self, context_ids, question_ids, max_new_tokens, eos_token_id=None):
        if not context_ids or max_new_tokens < 1:
            raise ValueError("generation requires nonempty context and a positive token budget")
        if len(context_ids) + len(question_ids) + max_new_tokens > self.backbone.config.max_position_embeddings:
            raise ValueError("prompt plus generation budget exceeds the model position limit")
        state = self._ingest(context_ids, differentiable=False)
        for _ in self.consume(question_ids, state, write=False):
            pass
        eos = self.backbone.generation_config.eos_token_id if eos_token_id is None else eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
        tokens, stop_reason = [], "max_new_tokens"
        for index in range(max_new_tokens):
            token = int(self.backbone.lm_head(state.last_hidden).argmax(dim=-1).item())
            tokens.append(token)
            if token in eos:
                stop_reason = "eos"
                break
            if index + 1 < max_new_tokens:
                for _ in self.consume((token,), state, write=False):
                    pass
        return {"token_ids": tokens, "stop_reason": stop_reason, **self._diagnostics(state)}
