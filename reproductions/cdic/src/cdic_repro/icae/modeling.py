from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import Tensor, nn
from torch.nn import functional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from cdic_repro.icae.token_layout import prepare_icae_token_layout


@dataclass(frozen=True, slots=True)
class IcaeConfig:

    memory_size: int = 128
    lora_rank: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_memory_head: bool = False

    def __post_init__(self) -> None:
        if self.memory_size < 1:
            raise ValueError("memory_size must be positive")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be positive")
        if self.lora_alpha < 1:
            raise ValueError("lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")


class MemoryHead(nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, hidden_size)
        self.dense_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dense_in.weight.dtype)
        hidden_states = functional.gelu(self.dense_in(hidden_states))
        return self.dense_out(hidden_states).to(input_dtype)


class LlamaICAE(nn.Module):

    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: IcaeConfig,
    ) -> None:
        super().__init__()
        if base_model.config.model_type != "llama":
            raise ValueError("LlamaICAE requires a Llama causal language model")
        base_embedding_count = base_model.get_input_embeddings().num_embeddings
        if base_embedding_count != len(tokenizer):
            raise ValueError(
                "base model input embeddings and tokenizer must have the same vocabulary size"
            )

        self.config = config
        self.tokenizer = tokenizer
        self.token_layout = prepare_icae_token_layout(tokenizer, config.memory_size)
        base_model.resize_token_embeddings(
            self.token_layout.tokenizer_vocabulary_size,
            mean_resizing=False,
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.config.eos_token_id = tokenizer.eos_token_id

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        self.icae: PeftModel = get_peft_model(base_model, lora_config)

        embedding_weight = self.icae.get_input_embeddings().weight
        self.memory_token_embed = nn.Embedding(
            config.memory_size + 3,
            base_model.config.hidden_size,
            device=embedding_weight.device,
            dtype=embedding_weight.dtype,
        )
        self.memory_head = (
            MemoryHead(base_model.config.hidden_size).to(
                device=embedding_weight.device,
                dtype=embedding_weight.dtype,
            )
            if config.use_memory_head
            else None
        )
        self._decode_without_adapter: ContextVar[bool] = ContextVar(
            "icae_decode_without_adapter",
            default=False,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        config: IcaeConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> LlamaICAE:
        tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path))
        base_model = AutoModelForCausalLM.from_pretrained(
            str(model_name_or_path),
            dtype=dtype,
        )
        return cls(base_model=base_model, tokenizer=tokenizer, config=config)

    @property
    def hidden_size(self) -> int:
        return int(self.icae.get_base_model().config.hidden_size)

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        """获取 tokens 的嵌入表示.

        存在三类不同的 token:
        - 普通文本 token
        - memory token
        - 三个控制 token ([AE], [LM], [FT])
        其中, 在嵌入时, 普通文本 token 可以直接使用 base model 的嵌入层, 而 memory token 和控制 token 则需要使用专门的嵌入层 (memory_token_embed).
        """

        if tokens.dtype != torch.long:
            raise TypeError("tokens must use torch.long")
        if tokens.numel() > 0:
            minimum = int(tokens.min().item())
            maximum = int(tokens.max().item())
            if minimum < 0 or maximum >= self.token_layout.token_id_upper_bound:
                raise ValueError("tokens contain values outside the ICAE token layout")

        special_mask = tokens >= self.token_layout.memory_token_start
        base_tokens = tokens.masked_fill(special_mask, self.tokenizer.pad_token_id)
        embeddings = self.icae.get_input_embeddings()(base_tokens)
        if special_mask.any():
            special_indices = tokens[special_mask] - self.token_layout.memory_token_start
            embeddings = embeddings.clone()
            embeddings[special_mask] = self.memory_token_embed(special_indices).to(embeddings)
        return embeddings

    def compress(
        self,
        encoder_tokens: Tensor,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        """用 frozen base model with adapter 编码输入序列, 然后取出其中所有 memory token 位置的最后一层隐藏状态作为 memory slots."""

        if encoder_tokens.ndim != 2:
            raise ValueError("encoder_tokens must have shape [batch_size, seq_len]")
        expected_memory_tokens = torch.arange(
            self.token_layout.memory_token_start,
            self.token_layout.memory_token_end,
            dtype=encoder_tokens.dtype,
            device=encoder_tokens.device,
        ).expand(encoder_tokens.shape[0], -1)
        if not torch.equal(
            encoder_tokens[:, -self.config.memory_size :],
            expected_memory_tokens,
        ):
            raise ValueError(
                "each encoder input must end with the ordered ICAE memory token sequence"
            )

        return self.compress_embeddings(
            self.embed_tokens(encoder_tokens),
            encoder_attention_mask=encoder_attention_mask,
        )

    def compress_embeddings(
        self,
        encoder_embeddings: Tensor,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        """编码以有序 memory embeddings 结尾的输入并返回 memory slots。"""

        if encoder_embeddings.ndim != 3:
            raise ValueError(
                "encoder_embeddings must have shape [batch_size, seq_len, hidden_size]"
            )
        if encoder_embeddings.shape[1] < self.config.memory_size:
            raise ValueError("encoder_embeddings contain fewer positions than memory slots")
        if encoder_embeddings.shape[2] != self.hidden_size:
            raise ValueError("encoder_embeddings hidden size does not match the ICAE model")

        encoder_outputs = self.icae(
            inputs_embeds=encoder_embeddings,
            attention_mask=encoder_attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = encoder_outputs.hidden_states[-1]
        if self.memory_head is not None:
            hidden_states = self.memory_head(hidden_states)
        return hidden_states[:, -self.config.memory_size :, :]

    def decode(
        self,
        decoder_embeddings: Tensor,
        decoder_attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutputWithPast:
        """用 frozen base model 解码, 注意此时需要禁用 adapter."""

        with self._disable_adapter_for_decode():
            return self.icae(
                inputs_embeds=decoder_embeddings,
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

    def forward(
        self,
        encoder_tokens: Tensor,
        decoder_tokens: Tensor,
        labels: Tensor,
        encoder_attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
    ) -> CausalLMOutputWithPast:
        """执行一次训练时完整的压缩—解码前向流程."""

        if decoder_tokens.ndim != 2 or labels.ndim != 2:
            raise ValueError(
                "decoder_tokens and labels must be tensors of shape [batch_size, seq_len]"
            )
        if decoder_tokens.shape != labels.shape:
            raise ValueError("decoder_tokens and labels must have the same shape")
        valid_labels = labels != -100
        if valid_labels.any():
            selected_labels = labels[valid_labels]
            if (
                int(selected_labels.min().item()) < 0
                or int(selected_labels.max().item()) >= self.token_layout.tokenizer_vocabulary_size
            ):
                raise ValueError("labels must contain only tokenizer token IDs or -100")

        memory_embeddings = self.compress(
            encoder_tokens,
            encoder_attention_mask=encoder_attention_mask,
        )
        decoder_token_embeddings = self.embed_tokens(decoder_tokens)
        decoder_embeddings = torch.cat((memory_embeddings, decoder_token_embeddings), dim=1)
        full_decoder_attention_mask = None
        if decoder_attention_mask is not None:
            memory_attention_mask = torch.ones(
                (decoder_attention_mask.shape[0], self.config.memory_size),
                dtype=decoder_attention_mask.dtype,
                device=decoder_attention_mask.device,
            )
            full_decoder_attention_mask = torch.cat(
                (memory_attention_mask, decoder_attention_mask),
                dim=1,
            )

        decoder_outputs = self.decode(
            decoder_embeddings=decoder_embeddings,
            decoder_attention_mask=full_decoder_attention_mask,
            use_cache=False,
        )
        logits = decoder_outputs.logits[:, self.config.memory_size - 1 : -1, :].contiguous()
        if logits.shape[:2] != labels.shape:
            raise RuntimeError(
                "decoder logits and labels have incompatible shapes of [batch, seq_len]"
            )
        loss = functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

    def gradient_checkpointing_enable(self) -> None:
        """启用梯度检查点，并确保反向重计算时保持与前向一致的 LoRA 开关状态."""

        def context_fn() -> tuple[Any, Any]:
            recompute_context = (
                self.icae.disable_adapter() if self._decode_without_adapter.get() else nullcontext()
            )
            return nullcontext(), recompute_context

        self.icae.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "context_fn": context_fn,
            }
        )

    def gradient_checkpointing_disable(self) -> None:
        """禁用梯度检查点."""
        self.icae.gradient_checkpointing_disable()

    @contextmanager
    def _disable_adapter_for_decode(self) -> Generator[None, None, None]:
        """在解码路径中禁用 LoRA adapter, 以确保使用冻结的 base model."""
        context_token = self._decode_without_adapter.set(True)
        try:
            with self.icae.disable_adapter():
                yield
        finally:
            self._decode_without_adapter.reset(context_token)
