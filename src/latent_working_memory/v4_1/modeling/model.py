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

    def select(self, indices):
        """提取仍带完整计算图的样本；同一活跃组共享流式位置与写入轮次。"""
        rows = torch.tensor(indices, device=self.keys[0].device, dtype=torch.long)
        return StreamState(
            memories=[value.index_select(0, rows) for value in self.memories],
            memory_kv=[tuple(value.index_select(0, rows) for value in kv) for kv in self.memory_kv],
            keys=[value.index_select(0, rows) for value in self.keys],
            values=[value.index_select(0, rows) for value in self.values],
            recent_queries=[value.index_select(0, rows) for value in self.recent_queries],
            position=self.position, compressed_tokens=self.compressed_tokens,
            write_events=self.write_events,
            last_hidden=None if self.last_hidden is None else self.last_hidden.index_select(0, rows),
            inner_before=[value.index_select(0, rows) for value in self.inner_before],
            inner_after=[value.index_select(0, rows) for value in self.inner_after],
        )


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

    def initial_state(self, differentiable=False, batch_size=1):
        dtype = self.backbone.model.embed_tokens.weight.dtype
        memories, memory_kv, keys, values, queries = [], [], [], [], []
        for memory, layer in zip(self.memory_layers, self.backbone.model.layers):
            slots = memory.initial_slots if differentiable else memory.initial_slots.detach()
            memories.append(slots.unsqueeze(0).expand(batch_size, -1, -1))
            empty = torch.empty(
                (batch_size, memory.num_kv_heads, 0, memory.head_dim), device=self.device, dtype=dtype
            )
            memory_kv.append((empty, empty, empty.new_empty((batch_size, memory.num_kv_heads, 0))))
            keys.append(empty)
            values.append(empty)
            queries.append(empty.new_empty((batch_size, layer.self_attn.q_proj.out_features // memory.head_dim,
                                            0, memory.head_dim)))
        return StreamState(memories, memory_kv, keys, values, queries)

    def _advance(self, token_ids, state, differentiable):
        """纯 layer checkpoint 不读写 StreamState，避免重算时重复追加缓存。"""
        with torch.set_grad_enabled(differentiable):
            ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
            hidden = self.backbone.model.embed_tokens(ids)
            positions = torch.arange(state.position, state.position + len(token_ids[0]),
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
            state.position += len(token_ids[0])
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
                pending_key = state.keys[index][:, :, :count].float()
                pending_value = state.values[index][:, :, :count].float()
                teacher = (
                    torch.cat((old_key.float(), pending_key), dim=2),
                    torch.cat((old_value.float(), pending_value), dim=2),
                    torch.cat((old_bias.float(), old_bias.new_zeros((*old_bias.shape[:-1], count))), -1),
                )
                query = state.recent_queries[index].transpose(1, 2).float()
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
        state.inner_before.append(torch.stack(before).mean(dim=0))
        state.inner_after.append(torch.stack(after).mean(dim=0))

    def _outer_am(self, state, teachers, generator):
        """返回逐样本的 AM 总损失，以及停止梯度的 output、log-mass 分项。"""
        references = [teacher.reference(
            state.write_events, state.position, state.compressed_tokens, generator
        ) for teacher in teachers]
        losses, output_losses, mass_losses = [], [], []
        for index, kv in enumerate(state.memory_kv):
            reference = {name: torch.stack([record[name][index] for record in references])
                         for name in ("queries", "outputs", "log_mass", "scales_o", "scales_z")}
            outputs, mass = local_read(reference["queries"], *kv)
            output_loss, mass_loss = matching_loss(
                outputs, mass, reference["outputs"], reference["log_mass"],
                reference["scales_o"], reference["scales_z"],
            )
            losses.append(output_loss + mass_loss)
            output_losses.append(output_loss.detach())
            mass_losses.append(mass_loss.detach())
        return tuple(torch.stack(values).mean(dim=0)
                     for values in (losses, output_losses, mass_losses))

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
            hidden = self._advance([token_ids[start:start + length]], state, differentiable)
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

    def _ingest_batch(self, episodes, teachers, generator, differentiable):
        """正文按活跃样本批量推进；正文结束后提取该样本的最终状态。"""
        active = list(range(len(episodes)))
        state = self.initial_state(differentiable, batch_size=len(episodes))
        final_states = [None] * len(episodes)
        outer_losses = [[] for _ in episodes]
        capacity = self.config.pending_size + self.config.recent_size
        while active:
            length = min(
                self.config.prefill_chunk_size,
                capacity - state.keys[0].shape[2],
                min(len(episodes[index].context_ids) - state.position for index in active),
            )
            self._advance([
                episodes[index].context_ids[state.position:state.position + length] for index in active
            ], state, differentiable)
            if state.keys[0].shape[2] == capacity:
                self._write(state, differentiable)
                if teachers is not None:
                    losses = self._outer_am(state, [teachers[index] for index in active], generator)
                    for row, index in enumerate(active):
                        outer_losses[index].append(tuple(loss[row] for loss in losses))
            continuing = []
            for row, index in enumerate(active):
                if state.position == len(episodes[index].context_ids):
                    final_states[index] = state.select([row])
                else:
                    continuing.append(row)
            if len(continuing) != len(active):
                if continuing:
                    state = state.select(continuing)
                active = [active[row] for row in continuing]
        return final_states, outer_losses

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
                    losses = self._outer_am(state, [teacher], generator)
                    outer_losses.append(tuple(loss[0] for loss in losses))

            state = self._ingest(episode.context_ids, differentiable, supervise)
            return self._sample_output(episode, state, outer_losses, lambda_am, differentiable)

    def forward_batch(self, episodes, teachers=None, lambda_am=1.0,
                      generator=None, differentiable=True):
        """正文及写入真正批量计算；QA 分别读取各自状态，返回逐样本损失。"""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("online TTT needs autograd; use no_grad(), not inference_mode()")
        if not episodes or any(not episode.context_ids or not episode.target_ids for episode in episodes):
            raise ValueError("a QA batch must contain nonempty contexts and targets")
        if max(len(episode.context_ids) for episode in episodes) > self.backbone.config.max_position_embeddings:
            raise ValueError("input exceeds the backbone global position limit")
        with torch.set_grad_enabled(differentiable):
            states, outer_losses = self._ingest_batch(episodes, teachers, generator, differentiable)
            return [self._sample_output(episode, state, losses, lambda_am, differentiable)
                    for episode, state, losses in zip(episodes, states, outer_losses)]

    def _sample_output(self, episode, state, outer_losses, lambda_am, differentiable):
        qa_loss = self._qa_loss(episode, state, differentiable)
        if outer_losses:
            outer_am, outer_am_output, outer_am_mass = (
                torch.stack(values).mean() for values in zip(*outer_losses)
            )
        else:
            outer_am, outer_am_output, outer_am_mass = (qa_loss.new_zeros(()) for _ in range(3))
        return {
            "loss": qa_loss + lambda_am * outer_am,
            "qa_loss": qa_loss, "outer_am": outer_am,
            "outer_am_output": outer_am_output, "outer_am_mass": outer_am_mass,
            "target_tokens": len(episode.target_ids), **self._diagnostics(state),
        }

    def _qa_loss(self, episode, state, differentiable):
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
        return torch.stack(losses).sum() / len(episode.target_ids)

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
