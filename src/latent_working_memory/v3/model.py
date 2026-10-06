"""单基座 gist-token 写入器与无 LoRA 的可微读取器。"""

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
    ):
        super().__init__()
        if base_model.config.model_type not in {"llama", "qwen2", "qwen3"}:
            raise ValueError("gist memory supports Llama, Qwen2 and Qwen3 causal LMs")
        if base_model.is_gradient_checkpointing:
            raise ValueError(
                "native gradient checkpointing is unsupported with shared reader/writer adapters"
            )
        if type(memory_slots) is not int or memory_slots < 1:
            raise ValueError("memory_slots must be a positive integer")
        if lora_dropout != 0.0:
            raise ValueError("the initial gist writer requires lora_dropout=0")
        self.width = base_model.config.hidden_size
        self.max_positions = base_model.config.max_position_embeddings
        self.memory_slots = memory_slots
        base_model.requires_grad_(False)
        base_model.config.use_cache = False
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
        embedding = self.language_model.get_input_embeddings().weight
        self.memory_embeddings = nn.Parameter(
            torch.empty(memory_slots, self.width, device=embedding.device, dtype=torch.float32)
        )
        nn.init.normal_(self.memory_embeddings, mean=0.0, std=0.02)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # 冻结基座的 dropout 始终关闭；LoRA 的梯度不依赖模块 training 标记。
        self.language_model.eval()
        return self

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

    def compress(self, text_ids, memory_blocks=None):
        """压缩新文本及指定历史块；调用方决定传入末块还是累计历史。"""
        self._check_tokens(text_ids)
        blocks = [] if memory_blocks is None else memory_blocks
        for memory in blocks:
            self._check_memory(memory)
            if len(memory) != self.memory_slots:
                raise ValueError("writer memory blocks must each contain memory_slots vectors")
        self._check_length(
            sum(len(memory) for memory in blocks) + len(text_ids) + self.memory_slots
        )
        embed = self.language_model.get_input_embeddings()
        text = embed(text_ids)
        inputs = torch.cat(
            [
                *(memory.to(text.dtype) for memory in blocks),
                text,
                self.memory_embeddings.to(text.dtype),
            ]
        )[None]
        hidden = (
            self.language_model.get_base_model()
            .model(
                inputs_embeds=inputs,
                attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device),
                position_ids=torch.arange(inputs.shape[1], device=inputs.device)[None],
                use_cache=False,
                return_dict=True,
            )
            .last_hidden_state
        )
        return hidden[0, -self.memory_slots :]

    def answer_nll(self, memories, prompt_ids, answer_ids):
        """返回每题答案 token 的平均 NLL；不为输入自动添加 BOS/EOS。"""
        if not memories or not len(memories) == len(prompt_ids) == len(answer_ids):
            raise ValueError("memories, prompts and answers must align and be nonempty")
        embed = self.language_model.get_input_embeddings()
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
        mask = positions < torch.tensor([len(row) for row in rows], device=inputs.device)[:, None]
        with self.language_model.disable_adapter():
            # 只关闭 adapter；不能使用 no_grad，否则读取损失无法回传到记忆。
            base = self.language_model.get_base_model()
            hidden = base.model(
                inputs_embeds=inputs,
                attention_mask=mask,
                position_ids=positions.expand_as(mask).masked_fill(~mask, 0),
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
            # backward 的 checkpoint 重算同样关闭 adapter，不依赖外层上下文仍然存活。
            with self.language_model.disable_adapter():
                logits = self.language_model.get_output_embeddings()(hidden)
                scores = (
                    logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
                )
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
        text = self.language_model.get_input_embeddings()(prompt_ids)
        inputs = torch.cat((memory.to(text.dtype), text))[None]
        with self.language_model.disable_adapter():
            return self.language_model.generate(
                inputs_embeds=inputs,
                attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True,
            )[0]

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
