"""同源独立编码器／解码器：gist 写入、冻结读取与逐层激活重计算。"""

from copy import deepcopy

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
        self.width = base_model.config.hidden_size
        self.max_positions = base_model.config.max_position_embeddings
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
        for backbone in (self.language_model.get_base_model(), self.decoder):
            if gradient_checkpointing:
                backbone.gradient_checkpointing_enable({"use_reentrant": False})
            else:
                backbone.gradient_checkpointing_disable()
        embedding = self.language_model.get_input_embeddings().weight
        self.memory_embeddings = nn.Parameter(
            torch.empty(write_slots, self.width, device=embedding.device, dtype=torch.float32)
        )
        nn.init.normal_(self.memory_embeddings, mean=0.0, std=0.02)
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

    def compress(self, text_ids, memory_blocks=None, output_slots=None):
        """压缩新文本及指定历史块；调用方决定传入末块还是累计历史。"""
        histories = None if memory_blocks is None else [memory_blocks]
        slots = None if output_slots is None else [output_slots]
        return self.compress_batch([text_ids], histories, slots)[0]

    def compress_batch(self, text_ids, memory_blocks=None, output_slots=None):
        """一次 forward 写入独立样本，按各样本指定的 slots 数放置 gist tokens。"""
        if not text_ids:
            raise ValueError("writer text batch must be nonempty")
        histories = [[] for _ in text_ids] if memory_blocks is None else memory_blocks
        if len(histories) != len(text_ids):
            raise ValueError("writer texts and memory histories must align")
        slots = [self.write_slots] * len(text_ids) if output_slots is None else output_slots
        if len(slots) != len(text_ids):
            raise ValueError("writer texts and output_slots must align")
        embed = self.language_model.get_input_embeddings()
        rows = []
        for tokens, blocks, count in zip(text_ids, histories, slots, strict=True):
            if type(count) is not int or not 1 <= count <= self.write_slots:
                raise ValueError("output_slots must be integers between 1 and write_slots")
            self._check_tokens(tokens)
            for memory in blocks:
                self._check_memory(memory)
                if not 1 <= len(memory) <= self.memory_slots:
                    raise ValueError("writer memory blocks must contain 1 to memory_slots vectors")
            self._check_length(sum(len(memory) for memory in blocks) + len(tokens) + count)
            text = embed(tokens)
            rows.append(
                torch.cat(
                    [
                        *(memory.to(text.dtype) for memory in blocks),
                        text,
                        self.memory_embeddings[:count].to(text.dtype),
                    ]
                )
            )
        inputs = pad_sequence(rows, batch_first=True)
        positions = torch.arange(inputs.shape[1], device=inputs.device)[None]
        # 只有右侧 padding，因果 attention 下有效前缀不会读到 padding。
        # 不传四维 mask，也不重置 padding 的位置，保留 SDPA 的纯 causal 路径。
        hidden = (
            self.language_model.get_base_model()
            .model(
                inputs_embeds=inputs,
                attention_mask=None,
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
                for name, tensor in get_peft_model_state_dict(
                    self.language_model, save_embedding_layers=False
                ).items()
            },
        }

    def load_trainable_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"memory_embeddings", "adapter"}:
            raise ValueError("trainable state must contain exactly memory_embeddings and adapter")
        if state["memory_embeddings"].shape != self.memory_embeddings.shape:
            raise ValueError("checkpoint memory embedding shape differs from the model")
        expected = get_peft_model_state_dict(self.language_model, save_embedding_layers=False)
        if set(state["adapter"]) != set(expected):
            raise ValueError("checkpoint adapter keys differ from the model")
        for name, tensor in state["adapter"].items():
            if tensor.shape != expected[name].shape:
                raise ValueError(f"checkpoint adapter shape differs for {name}")
        with torch.no_grad():
            self.memory_embeddings.copy_(state["memory_embeddings"])
        set_peft_model_state_dict(self.language_model, state["adapter"])
