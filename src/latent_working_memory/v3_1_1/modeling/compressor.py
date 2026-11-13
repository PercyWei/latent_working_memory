"""同源 LoRA 写入器与冻结读取器；连续 memory 路径保留梯度。"""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from pathlib import Path

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .batching import pack_caches, pad_inputs, unpack_caches
from .boundary import BoundaryHead


class MemoryModel(nn.Module):
    def __init__(self, model_name_or_path, model_settings, compute_settings, device):
        super().__init__()
        structure, lora = model_settings["structure"], model_settings["lora"]
        self.mixed_precision = compute_settings["mixed_precision"]
        self.save_on_cpu = compute_settings["save_on_cpu"]
        source = str(Path(model_name_or_path).expanduser())
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("memory training requires tokenizer.eos_token_id")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=getattr(torch, compute_settings["dtype"]),
            attn_implementation=compute_settings["attention_implementation"],
        ).to(device)
        if base.config.model_type not in {"llama", "qwen2", "qwen3"}:
            raise ValueError("the memory writer supports Llama, Qwen2 and Qwen3 causal LMs")
        layer_types = getattr(base.config, "layer_types", None)
        if layer_types is not None and any(attention != "full_attention" for attention in layer_types):
            raise ValueError("the streaming writer requires full-attention layers")
        if base.config.rope_scaling is not None and base.config.rope_scaling["rope_type"] != "default":
            raise ValueError("the streaming writer requires fixed default RoPE")
        self.memory_slots = structure["memory_slots"]
        self.max_positions = base.config.max_position_embeddings
        self.width = base.config.hidden_size
        base.requires_grad_(False)
        base.config.use_cache = False
        base.config.attention_dropout = 0.0
        for layer in base.model.layers:
            layer.self_attn.attention_dropout = 0.0
        for module in base.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        # 模块状态各自独立，只复用始终冻结的基座参数。
        self.decoder = deepcopy(base, {id(parameter): parameter for parameter in base.parameters()})
        self.writer = get_peft_model(
            base,
            LoraConfig(
                r=lora["lora_rank"],
                lora_alpha=lora["lora_alpha"],
                target_modules=lora["lora_target_modules"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        for parameter in self.writer.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        for backbone in (self.writer.get_base_model(), self.decoder):
            if compute_settings["gradient_checkpointing"]:
                backbone.gradient_checkpointing_enable({"use_reentrant": False})
            else:
                backbone.gradient_checkpointing_disable()
        self.memory_embeddings = nn.Parameter(
            torch.empty(self.memory_slots, self.width, device=device, dtype=torch.float32)
        )
        nn.init.normal_(self.memory_embeddings, mean=0.0, std=0.02)
        boundary_hidden_size = structure["boundary_hidden_size"]
        self.boundary = (None if boundary_hidden_size is None
                         else BoundaryHead(self.width, boundary_hidden_size, device))
        self.train()

    @contextmanager
    def forward_context(self):
        with ExitStack() as stack:
            if self.mixed_precision:
                stack.enter_context(torch.autocast(
                    device_type=self.memory_embeddings.device.type, dtype=torch.bfloat16,
                ))
            if self.save_on_cpu and torch.is_grad_enabled():
                stack.enter_context(torch.autograd.graph.save_on_cpu(pin_memory=True))
            yield

    def _tokens(self, values):
        return torch.tensor(values, dtype=torch.long, device=self.memory_embeddings.device)

    def _check_length(self, length):
        if length > self.max_positions:
            raise ValueError(f"sequence length {length} exceeds model window {self.max_positions}")

    def _inputs(self, backbone, memory_blocks, input_ids):
        tokens = self._tokens(input_ids)
        text = backbone.get_input_embeddings()(tokens)
        return torch.cat([*(block.to(text.dtype) for block in memory_blocks), text])

    @staticmethod
    def _hidden(backbone, inputs):
        positions = torch.arange(len(inputs), device=inputs.device)[None]
        return backbone.model(
            inputs_embeds=inputs[None],
            position_ids=positions,
            attention_mask=None,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[0]

    def _hidden_batch(self, backbone, rows):
        for row in rows:
            self._check_length(len(row))
        inputs, _, mask = pad_inputs(rows)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        return backbone.model(
            inputs_embeds=inputs, attention_mask=mask, position_ids=positions,
            use_cache=False, return_dict=True,
        ).last_hidden_state

    def compress(self, input_ids, memory_blocks=()):
        if not input_ids:
            raise ValueError("cannot compress an empty text block")
        backbone = self.writer.get_base_model()
        source = self._inputs(backbone, memory_blocks, input_ids)
        inputs = torch.cat((source, self.memory_embeddings.to(source.dtype)))
        self._check_length(len(inputs))
        return self._hidden(backbone, inputs)[-self.memory_slots :]

    def compress_batch(self, input_ids):
        """批量压缩独立正文，保留各行 memory 到 Writer 的梯度。"""
        backbone = self.writer.get_base_model()
        rows = []
        for tokens in input_ids:
            if not tokens:
                raise ValueError("cannot compress an empty text block")
            source = self._inputs(backbone, (), tokens)
            rows.append(torch.cat((source, self.memory_embeddings.to(source.dtype))))
        return self._hidden_batch(backbone, rows)[:, -self.memory_slots:]

    @torch.no_grad()
    def _source_hidden(self, inputs, cache=None):
        """仅用于无梯度采样；可变 KV 不进入 checkpoint 的重计算图。"""
        backbone = self.writer.get_base_model()
        offset = 0 if cache is None else cache.get_seq_length()
        self._check_length(offset + len(inputs))
        positions = torch.arange(offset, offset + len(inputs), device=inputs.device)
        was_training = backbone.training
        # 原生 gradient checkpoint 在 train 模式禁用 cache；无 dropout 时 eval
        # 与回放的 train 前向使用同一计算，只关闭重计算机制。
        backbone.eval()
        try:
            output = backbone.model(
                inputs_embeds=inputs[None],
                position_ids=positions[None],
                cache_position=positions,
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        finally:
            backbone.train(was_training)
        return output.last_hidden_state[0], output.past_key_values

    @torch.no_grad()
    def prefill_source(self, memory_blocks, input_ids, cache=None):
        backbone = self.writer.get_base_model()
        inputs = self._inputs(backbone, memory_blocks, input_ids)
        return self._source_hidden(inputs, cache)[1]

    @torch.no_grad()
    def observe_source(self, cache, new_unit_ids):
        embed = self.writer.get_input_embeddings()
        hidden, cache = self._source_hidden(embed(self._tokens(new_unit_ids)), cache)
        return self.boundary(hidden[-1]), cache

    @torch.no_grad()
    def compress_source(self, cache):
        dtype = self.writer.get_input_embeddings().weight.dtype
        hidden, _ = self._source_hidden(self.memory_embeddings.to(dtype), cache)
        return hidden

    @torch.no_grad()
    def _source_hidden_batch(self, rows, caches=None, keep_cache=True):
        """无梯度批量源前向；padding 使用物理 cache 槽位和真实 RoPE 位置。"""
        inputs, new_lengths, new_mask = pad_inputs(rows)
        batch, new_width = inputs.shape[:2]
        if caches is None:
            packed, past_lengths, past_width = None, [0] * batch, 0
            past_mask = new_mask[:, :0]
        else:
            if len(caches) != batch:
                raise ValueError("source caches and new inputs must have the same batch size")
            packed, past_lengths, past_width, past_mask = pack_caches(caches)
        for past, new in zip(past_lengths, new_lengths, strict=True):
            self._check_length(past + new)
        mask = torch.cat((past_mask, new_mask), dim=-1)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)[:, -new_width:]
        cache_positions = torch.arange(past_width, past_width + new_width, device=inputs.device)
        backbone = self.writer.get_base_model()
        was_training = backbone.training
        backbone.eval()
        try:
            output = backbone.model(
                inputs_embeds=inputs, attention_mask=mask, position_ids=positions,
                cache_position=cache_positions, past_key_values=packed,
                use_cache=True, return_dict=True,
            )
        finally:
            backbone.train(was_training)
        split = (unpack_caches(output.past_key_values, past_lengths, new_lengths, past_width, new_width)
                 if keep_cache else None)
        return output.last_hidden_state, split

    @torch.no_grad()
    def prefill_source_batch(self, memories, inputs):
        backbone = self.writer.get_base_model()
        rows = [self._inputs(backbone, blocks, tokens)
                for blocks, tokens in zip(memories, inputs, strict=True)]
        return self._source_hidden_batch(rows)[1]

    @torch.no_grad()
    def observe_source_batch(self, caches, new_units):
        embed = self.writer.get_input_embeddings()
        rows = [embed(self._tokens(tokens)) for tokens in new_units]
        hidden, updated = self._source_hidden_batch(rows, caches)
        return self.boundary(hidden[:, -1]), updated

    @torch.no_grad()
    def compress_source_batch(self, caches):
        dtype = self.writer.get_input_embeddings().weight.dtype
        rows = [self.memory_embeddings.to(dtype) for _ in caches]
        hidden, _ = self._source_hidden_batch(rows, caches, keep_cache=False)
        return list(hidden.unbind(0))

    def boundary_logit(self, memory_blocks, pending_ids, new_unit_ids):
        if not pending_ids or not new_unit_ids:
            raise ValueError("a boundary decision requires pending text and a new unit")
        backbone = self.writer.get_base_model()
        inputs = self._inputs(backbone, memory_blocks, (*pending_ids, *new_unit_ids))
        self._check_length(len(inputs))
        # 只读取此刻已到达的最后一个原文位置；不计算记忆 suffix 或任务输入。
        # 每次重建当前因果前缀，避免将提交前的 KV 用到已经改变的上下文。
        source_hidden = self._hidden(backbone, inputs)[-1]
        return self.boundary(source_hidden)

    def boundary_logits(self, memory_blocks, input_ids, positions):
        """一次因果前向读取多个边界；positions 为正文中的零起始位置。"""
        backbone = self.writer.get_base_model()
        inputs = self._inputs(backbone, memory_blocks, input_ids)
        self._check_length(len(inputs))
        indices = self._tokens(positions) + sum(len(block) for block in memory_blocks)
        return self.boundary(self._hidden(backbone, inputs)[indices])

    def answer_nll(self, memory_blocks, prompt_ids, target_ids):
        if not memory_blocks or not prompt_ids or not target_ids:
            raise ValueError("answer scoring requires memory, a prompt and nonempty targets")
        inputs = self._inputs(self.decoder, memory_blocks, (*prompt_ids, *target_ids[:-1]))
        self._check_length(len(inputs))
        offset = sum(len(block) for block in memory_blocks) + len(prompt_ids) - 1
        # Decoder 参数冻结，但不能 no_grad：目标通过 memory 回传到写入器。
        selected = self._hidden(self.decoder, inputs)[offset:]
        targets = self._tokens(target_ids)

        def token_loss(hidden, token_targets):
            logits = self.decoder.get_output_embeddings()(hidden)
            return F.cross_entropy(logits.float(), token_targets, reduction="sum")

        losses = []
        for start in range(0, len(targets), 256):
            hidden, token_targets = selected[start : start + 256], targets[start : start + 256]
            losses.append(
                checkpoint(token_loss, hidden, token_targets, use_reentrant=False)
                if torch.is_grad_enabled() and hidden.requires_grad
                else token_loss(hidden, token_targets)
            )
        return torch.stack(losses).sum() / len(targets)

    def answer_nll_batch(self, memories, prompt_ids, target_ids):
        """批量 Reader 前向；各行损失只平均自身的实际 target tokens。"""
        if not memories or not len(memories) == len(prompt_ids) == len(target_ids):
            raise ValueError("memories, prompts and targets must have the same nonzero batch size")
        if any(not targets for targets in target_ids):
            raise ValueError("NLL targets must be nonempty")
        rows = [self._inputs(self.decoder, blocks, (*prompt, *target[:-1]))
                for blocks, prompt, target in zip(memories, prompt_ids, target_ids, strict=True)]
        hidden = self._hidden_batch(self.decoder, rows)
        # 左 padding 后每行最后 len(target) 个隐藏状态恰好预测该行的目标。
        lengths = [len(target) for target in target_ids]
        selected = torch.cat([hidden[index, -length:] for index, length in enumerate(lengths)])
        targets = torch.cat([self._tokens(target) for target in target_ids])

        def token_loss(hidden, token_targets):
            logits = self.decoder.get_output_embeddings()(hidden)
            return F.cross_entropy(logits.float(), token_targets, reduction="none")

        losses = []
        for start in range(0, len(targets), 256):
            hidden, token_targets = selected[start:start + 256], targets[start:start + 256]
            losses.append(
                checkpoint(token_loss, hidden, token_targets, use_reentrant=False)
                if torch.is_grad_enabled() and hidden.requires_grad
                else token_loss(hidden, token_targets)
            )
        return torch.stack([values.mean() for values in torch.cat(losses).split(lengths)])

    @torch.no_grad()
    def generate(self, memory_blocks, prompt_ids, max_new_tokens):
        inputs = self._inputs(self.decoder, memory_blocks, prompt_ids)
        self._check_length(len(inputs) + max_new_tokens)
        was_training = self.decoder.training
        self.decoder.eval()
        try:
            tokens = self.decoder.generate(
                inputs_embeds=inputs[None],
                attention_mask=torch.ones((1, len(inputs)), device=inputs.device, dtype=torch.long),
                generation_config=GenerationConfig(
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                ),
                use_model_defaults=False,
            )[0]
        finally:
            self.decoder.train(was_training)
        return tuple(tokens.tolist())

    @torch.no_grad()
    def generate_batch(self, memories, prompt_ids, budgets, on_complete=None):
        """真实批量贪心解码；每行独立 EOS／预算，完成后移出活跃批次。"""
        count = len(memories)
        if not count or len(prompt_ids) != count or len(budgets) != count:
            raise ValueError("memories, prompts and budgets must have the same nonzero batch size")
        rows = [self._inputs(self.decoder, blocks, prompt)
                for blocks, prompt in zip(memories, prompt_ids, strict=True)]
        for length, budget in zip(map(len, rows), budgets, strict=True):
            if type(budget) is not int or budget < 1:
                raise ValueError("generation budgets must be positive integers")
            self._check_length(length + budget)
        inputs, lengths, mask = pad_inputs(rows)
        device = inputs.device
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        was_training = self.decoder.training
        self.decoder.eval()
        try:
            output = self.decoder.model(
                inputs_embeds=inputs, attention_mask=mask, position_ids=positions,
                cache_position=torch.arange(inputs.shape[1], device=device),
                use_cache=True, return_dict=True,
            )
            cache = output.past_key_values
            # 单行时末位置切片也可能被视为 contiguous，clone 确保不保留整段 hidden。
            last_hidden = output.last_hidden_state[:, -1].clone()
            valid_lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            del output, rows, inputs, positions, lengths
            active = list(range(count))
            tokens, results = [[] for _ in range(count)], [None] * count
            while active:
                predicted = self.decoder.get_output_embeddings()(last_hidden).argmax(-1).tolist()
                continuing, next_tokens = [], []
                for row, (index, token) in enumerate(zip(active, predicted, strict=True)):
                    tokens[index].append(token)
                    reason = ("eos" if token == self.tokenizer.eos_token_id else
                              "max_new_tokens" if len(tokens[index]) == budgets[index] else None)
                    if reason is None:
                        continuing.append(row)
                        next_tokens.append(token)
                    else:
                        result = {"token_ids": tokens[index], "stop_reason": reason}
                        results[index] = result
                        if on_complete is not None:
                            on_complete(index, result)
                if not continuing:
                    break
                if len(continuing) != len(active):
                    selected = torch.tensor(continuing, device=device, dtype=torch.long)
                    cache.batch_select_indices(selected)
                    mask = mask.index_select(0, selected)
                    valid_lengths = valid_lengths.index_select(0, selected)
                    active = [active[row] for row in continuing]
                mask = torch.cat((mask, mask.new_ones((len(active), 1))), dim=-1)
                token_ids = torch.tensor(next_tokens, device=device, dtype=torch.long)[:, None]
                output = self.decoder.model(
                    inputs_embeds=self.decoder.get_input_embeddings()(token_ids),
                    attention_mask=mask, position_ids=valid_lengths[:, None],
                    cache_position=torch.tensor([cache.get_seq_length()], device=device),
                    past_key_values=cache, use_cache=True, return_dict=True,
                )
                cache, last_hidden = output.past_key_values, output.last_hidden_state[:, -1]
                del output
                valid_lengths += 1
        finally:
            self.decoder.train(was_training)
        return results

    def compressor_parameters(self):
        yield self.memory_embeddings
        yield from (parameter for parameter in self.writer.parameters() if parameter.requires_grad)

    def boundary_parameters(self):
        yield from self.boundary.parameters()

    def trainable_state_dict(self):
        return {
            "memory_embeddings": self.memory_embeddings.detach().cpu().clone(),
            "adapter": {
                key: value.detach().cpu().clone()
                for key, value in get_peft_model_state_dict(
                    self.writer, save_embedding_layers=False
                ).items()
            },
            "boundary": None if self.boundary is None else {
                key: value.detach().cpu().clone() for key, value in self.boundary.state_dict().items()
            },
        }

    def load_compressor_state_dict(self, state):
        if set(state) != {"memory_embeddings", "adapter", "boundary"}:
            raise ValueError("model state must contain exactly memory_embeddings, adapter and boundary")
        if state["memory_embeddings"].shape != self.memory_embeddings.shape:
            raise ValueError("checkpoint memory embedding shape differs from this model")
        expected = get_peft_model_state_dict(self.writer, save_embedding_layers=False)
        adapter = state["adapter"]
        if set(adapter) != set(expected) or any(
            adapter[key].shape != expected[key].shape for key in expected
        ):
            raise ValueError("checkpoint LoRA keys or shapes differ from this model")
        set_peft_model_state_dict(self.writer, adapter)
        with torch.no_grad():
            self.memory_embeddings.copy_(state["memory_embeddings"])

    def load_trainable_state_dict(self, state):
        if (self.boundary is None) != (state["boundary"] is None):
            raise ValueError("checkpoint boundary state differs from this model's training stage")
        self.load_compressor_state_dict(state)
        if self.boundary is not None:
            self.boundary.load_state_dict(state["boundary"], strict=True)
