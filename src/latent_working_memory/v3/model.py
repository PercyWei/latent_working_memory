"""同源独立编码器／解码器：gist 写入、冻结读取与逐层激活重计算。"""

from copy import deepcopy
from contextlib import contextmanager, nullcontext

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint


class GistMemoryModel(nn.Module):
    def __init__(
        self,
        base_model,
        memory_slots,
        lora_rank,
        lora_alpha,
        lora_target_modules,
        lora_dropout=0.0,
        gradient_checkpointing=True,
        write_slots=None,
        writer_mode="local",
        tag_tokens=3,
    ):
        super().__init__()
        if base_model.config.model_type not in {"llama", "qwen2", "qwen3"}:
            raise ValueError("gist memory supports Llama, Qwen2 and Qwen3 causal LMs")
        if type(gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")
        if type(memory_slots) is not int or memory_slots < 1:
            raise ValueError("memory_slots must be a positive integer")
        write_slots = memory_slots if write_slots is None else write_slots
        if type(write_slots) is not int or not 1 <= write_slots <= memory_slots:
            raise ValueError("write_slots must be a positive integer no greater than memory_slots")
        if lora_dropout != 0.0:
            raise ValueError("the initial gist writer requires lora_dropout=0")
        if writer_mode not in {"local", "tag", "mask", "dual_lora"}:
            raise ValueError("writer_mode must be local, tag, mask or dual_lora")
        if type(tag_tokens) is not int or tag_tokens < 1:
            raise ValueError("tag_tokens must be a positive integer")
        self.width = base_model.config.hidden_size
        self.max_positions = base_model.config.max_position_embeddings
        self.writer_mode = writer_mode
        # 总容量 K 与单次写入可用的 gist embeddings 数可以不同。
        self.memory_slots = memory_slots
        self.write_slots = write_slots
        base_model.requires_grad_(False)
        base_model.config.use_cache = False
        # 保持原有 dropout 关闭的语义，同时允许 train 模式触发原生逐层重算。
        base_model.config.attention_dropout = 0.0
        for layer in base_model.model.layers:
            layer.self_attn.attention_dropout = 0.0
        for module in base_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        # 在安装 LoRA 前复制冻结基座；两个对象不共享 adapter 状态或权重存储。
        self.decoder = deepcopy(base_model)
        self.language_model = get_peft_model(
            base_model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        if writer_mode == "dual_lora":
            self.language_model.add_adapter(
                "append", deepcopy(self.language_model.peft_config["default"])
            )
            self._activate_adapter("default")
        for backbone in (self.language_model.get_base_model(), self.decoder):
            if gradient_checkpointing:
                backbone.gradient_checkpointing_enable({"use_reentrant": False})
            else:
                backbone.gradient_checkpointing_disable()
        if writer_mode == "dual_lora" and gradient_checkpointing:
            # 每次层重算捕获其原 forward 的 adapter，不读取后来写入动作的状态。
            self.language_model.get_base_model()._set_gradient_checkpointing(
                enable=True, gradient_checkpointing_func=self._writer_checkpoint
            )
        embedding = self.language_model.get_input_embeddings().weight
        self.memory_embeddings = nn.Parameter(
            torch.empty(write_slots, self.width, device=embedding.device, dtype=torch.float32)
        )
        nn.init.normal_(self.memory_embeddings, mean=0.0, std=0.02)
        if writer_mode == "tag":
            # 两组分别表示 <APPEND> 与 <REWRITE>，不扩充冻结基座的词表。
            self.tag_embeddings = nn.Parameter(
                torch.empty(2, tag_tokens, self.width, device=embedding.device, dtype=torch.float32)
            )
            nn.init.normal_(self.tag_embeddings, mean=0.0, std=0.02)
        self.train()

    def _check_tokens(self, token_ids):
        if token_ids.ndim != 1 or token_ids.dtype != torch.long or len(token_ids) == 0:
            raise ValueError("token IDs must be nonempty one-dimensional torch.long tensors")
        if token_ids.device != self.memory_embeddings.device:
            raise ValueError("token IDs and model must be on the same device")

    def _check_memory(self, memory):
        if memory.ndim != 2 or memory.shape[1] != self.width or not memory.is_floating_point():
            raise ValueError(f"memory must be a floating tensor with shape [slots, {self.width}]")
        if memory.device != self.memory_embeddings.device:
            raise ValueError("memory and model must be on the same device")

    def _check_length(self, length):
        if length > self.max_positions:
            raise ValueError(f"sequence length {length} exceeds model window {self.max_positions}")

    def _activate_adapter(self, adapter):
        self.language_model.set_adapter(adapter)
        # PEFT 默认冻结 inactive adapter；两套参数必须一直留在 optimizer/DDP 中。
        for name, parameter in self.language_model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)

    @contextmanager
    def _adapter_context(self, adapter):
        previous = self.language_model.active_adapter
        self._activate_adapter(adapter)
        try:
            yield
        finally:
            self._activate_adapter(previous)

    def _writer_checkpoint(self, function, *args):
        adapter = self.language_model.active_adapter
        return checkpoint(
            function,
            *args,
            use_reentrant=False,
            context_fn=lambda: (nullcontext(), self._adapter_context(adapter)),
        )

    def compress(self, text_ids, memory_blocks=None, output_slots=None, action=None):
        """压缩新文本及指定历史块；调用方决定传入末块还是累计历史。"""
        histories = None if memory_blocks is None else [memory_blocks]
        slots = None if output_slots is None else [output_slots]
        actions = None if action is None else [action]
        return self.compress_batch([text_ids], histories, slots, actions)[0]

    def compress_batch(self, text_ids, memory_blocks=None, output_slots=None, actions=None):
        """一次 forward 写入独立样本，按各样本指定的 slots 数放置 gist tokens。"""
        if not text_ids:
            raise ValueError("writer text batch must be nonempty")
        histories = [[] for _ in text_ids] if memory_blocks is None else memory_blocks
        if len(histories) != len(text_ids):
            raise ValueError("writer texts and memory histories must align")
        slots = [self.write_slots] * len(text_ids) if output_slots is None else output_slots
        if len(slots) != len(text_ids):
            raise ValueError("writer texts and output_slots must align")
        actions = [None] * len(text_ids) if actions is None else actions
        if len(actions) != len(text_ids) or any(
            action not in {None, "initial", "append", "overwrite"} for action in actions
        ):
            raise ValueError("writer actions must align and be initial, append or overwrite")
        if self.writer_mode == "dual_lora":
            results = [None] * len(text_ids)
            for adapter in ("default", "append"):
                indices = [
                    i
                    for i, action in enumerate(actions)
                    if ("append" if action in {"append", "initial"} else "default") == adapter
                ]
                if not indices:
                    continue
                with self._adapter_context(adapter):
                    written = self._compress_batch(
                        [text_ids[i] for i in indices],
                        [histories[i] for i in indices],
                        [slots[i] for i in indices],
                        [actions[i] for i in indices],
                    )
                for i, memory in zip(indices, written, strict=True):
                    results[i] = memory
            return results
        return self._compress_batch(text_ids, histories, slots, actions)

    def _compress_batch(self, text_ids, histories, slots, actions):
        embed = self.language_model.get_input_embeddings()
        rows, blocked_histories = [], []
        for tokens, blocks, count, action in zip(text_ids, histories, slots, actions, strict=True):
            if type(count) is not int or not 1 <= count <= self.write_slots:
                raise ValueError("output_slots must be integers between 1 and write_slots")
            self._check_tokens(tokens)
            for memory in blocks:
                self._check_memory(memory)
                if not 1 <= len(memory) <= self.memory_slots:
                    raise ValueError("writer memory blocks must contain 1 to memory_slots vectors")
            text = embed(tokens)
            tag = text[:0]
            if self.writer_mode == "tag" and action in {"append", "overwrite"}:
                index = 0 if action == "append" else 1
                tag = self.tag_embeddings[index].to(text.dtype)
            self._check_length(
                sum(len(memory) for memory in blocks) + len(tag) + len(tokens) + count
            )
            rows.append(
                torch.cat(
                    [
                        *(memory.to(text.dtype) for memory in blocks),
                        tag,
                        text,
                        self.memory_embeddings[:count].to(text.dtype),
                    ]
                )
            )
            blocked_histories.append(
                sum(len(memory) for memory in blocks)
                if action == "append"
                else sum(len(memory) for memory in blocks[:-1])
                if action == "overwrite"
                else 0
            )
        inputs = pad_sequence(rows, batch_first=True)
        positions = torch.arange(inputs.shape[1], device=inputs.device)[None]
        attention_mask = None
        if self.writer_mode == "mask" and any(blocked_histories):
            length = inputs.shape[1]
            attention_mask = (
                torch.full(
                    (length, length),
                    torch.finfo(inputs.dtype).min,
                    dtype=inputs.dtype,
                    device=inputs.device,
                )
                .triu(1)[None, None]
                .repeat(len(rows), 1, 1, 1)
            )
            for i, (row, count, blocked) in enumerate(
                zip(rows, slots, blocked_histories, strict=True)
            ):
                attention_mask[i, :, len(row) - count : len(row), :blocked] = torch.finfo(
                    inputs.dtype
                ).min
        # 只有右侧 padding，因果 attention 下有效前缀不会读到 padding。
        # 仅注意力限制版传四维 mask；其余写入保持 SDPA 的纯 causal 路径。
        hidden = (
            self.language_model.get_base_model()
            .model(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                position_ids=positions.expand(inputs.shape[:2]),
                use_cache=False,
                return_dict=True,
            )
            .last_hidden_state
        )
        return [
            hidden[index, len(row) - count : len(row)]
            for index, (row, count) in enumerate(zip(rows, slots, strict=True))
        ]

    def answer_nll(self, memories, prompt_ids, answer_ids):
        """返回每题答案 token 的平均 NLL；不为输入自动添加 BOS/EOS。"""
        if not memories or not len(memories) == len(prompt_ids) == len(answer_ids):
            raise ValueError("memories, prompts and answers must align and be nonempty")
        embed = self.decoder.get_input_embeddings()
        rows, offsets, lengths = [], [], []
        for memory, prompt, answer in zip(memories, prompt_ids, answer_ids, strict=True):
            self._check_memory(memory)
            self._check_tokens(prompt)
            self._check_tokens(answer)
            length = len(memory) + len(prompt) + len(answer) - 1
            self._check_length(length)
            text = embed(torch.cat((prompt, answer[:-1])))
            rows.append(torch.cat((memory.to(text.dtype), text)))
            offsets.append(len(memory) + len(prompt) - 1)
            lengths.append(len(answer))
        inputs = pad_sequence(rows, batch_first=True)
        positions = torch.arange(inputs.shape[1], device=inputs.device)[None]
        # 冻结参数仍向 memory 反传梯度；不能在此使用 no_grad。
        hidden = self.decoder.model(
            inputs_embeds=inputs,
            attention_mask=None,
            position_ids=positions.expand(inputs.shape[:2]),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        selected = torch.cat(
            [
                row[offset : offset + length]
                for row, offset, length in zip(hidden, offsets, lengths, strict=True)
            ]
        )

        def token_losses(hidden, targets):
            logits = self.decoder.get_output_embeddings()(hidden)
            scores = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
            return F.cross_entropy(scores, targets, reduction="none")

        targets = torch.cat(answer_ids)
        chunk_size = 256
        if len(selected) <= chunk_size:
            losses = token_losses(selected, targets)
        else:
            pieces = []
            for start in range(0, len(selected), chunk_size):
                hidden, target = (
                    selected[start : start + chunk_size],
                    targets[start : start + chunk_size],
                )
                pieces.append(
                    checkpoint(token_losses, hidden, target, use_reentrant=False)
                    if torch.is_grad_enabled() and hidden.requires_grad
                    else token_losses(hidden, target)
                )
            losses = torch.cat(pieces)
        return torch.stack([row.mean() for row in losses.split(lengths)])

    @torch.no_grad()
    def generate(self, memory, prompt_ids, max_new_tokens, eos_token_id, pad_token_id):
        """冻结读取端贪心生成，仅返回新生成的 token。"""
        self._check_memory(memory)
        self._check_tokens(prompt_ids)
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        self._check_length(len(memory) + len(prompt_ids) + max_new_tokens)
        text = self.decoder.get_input_embeddings()(prompt_ids)
        inputs = torch.cat((memory.to(text.dtype), text))[None]
        was_training = self.decoder.training
        self.decoder.eval()
        try:
            return self.decoder.generate(
                inputs_embeds=inputs,
                attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True,
            )[0]
        finally:
            self.decoder.train(was_training)

    def trainable_state_dict(self):
        return {
            "memory_embeddings": self.memory_embeddings.detach().cpu().clone(),
            "adapter": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self._adapter_state_dict().items()
            },
        }

    def _adapter_state_dict(self):
        state = get_peft_model_state_dict(self.language_model, save_embedding_layers=False)
        if self.writer_mode == "tag":
            state["tag_embeddings"] = self.tag_embeddings
        if self.writer_mode == "dual_lora":
            appended = get_peft_model_state_dict(
                self.language_model, adapter_name="append", save_embedding_layers=False
            )
            for name, tensor in appended.items():
                prefix, suffix = name.rsplit(".", 1)
                state[f"{prefix}.append.{suffix}"] = tensor
        return state

    def load_trainable_state_dict(self, state, initialize=False):
        if not isinstance(state, dict) or set(state) != {"memory_embeddings", "adapter"}:
            raise ValueError("trainable state must contain exactly memory_embeddings and adapter")
        if state["memory_embeddings"].shape != self.memory_embeddings.shape:
            raise ValueError("checkpoint memory embedding shape differs from the model")
        expected = self._adapter_state_dict()
        default = get_peft_model_state_dict(self.language_model, save_embedding_layers=False)
        adapter = state["adapter"]
        if initialize and self.writer_mode == "tag" and set(adapter) == set(default):
            # 共享预训练没有动作标记；继承已有权重，保留新初始化的两组向量。
            adapter = {**adapter, "tag_embeddings": self.tag_embeddings.detach()}
        if initialize and self.writer_mode == "dual_lora" and set(adapter) == set(default):
            # 共享预训练的一套 LoRA 显式复制给两种动作；恢复训练不走此路径。
            adapter = dict(adapter)
            for name, tensor in state["adapter"].items():
                prefix, suffix = name.rsplit(".", 1)
                adapter[f"{prefix}.append.{suffix}"] = tensor
        if set(adapter) != set(expected):
            raise ValueError("checkpoint adapter keys differ from the model")
        for name, tensor in adapter.items():
            if tensor.shape != expected[name].shape:
                raise ValueError(f"checkpoint adapter shape differs for {name}")
        with torch.no_grad():
            self.memory_embeddings.copy_(state["memory_embeddings"])
            if self.writer_mode == "tag":
                self.tag_embeddings.copy_(adapter["tag_embeddings"])
        set_peft_model_state_dict(self.language_model, {name: adapter[name] for name in default})
        if self.writer_mode == "dual_lora":
            appended = {}
            for name in default:
                prefix, suffix = name.rsplit(".", 1)
                appended[name] = adapter[f"{prefix}.append.{suffix}"]
            set_peft_model_state_dict(self.language_model, appended, adapter_name="append")
