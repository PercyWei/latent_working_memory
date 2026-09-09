from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.model import sinusoidal_positions
from latent_working_memory.v1.objectives import ReaderOutput, build_reader_output


BACKBONE_TRAINABLE_STATE_FIELDS = frozenset(
    {"input_projection", "memory_projection", "reader_lora"}
)


@dataclass(frozen=True, slots=True)
class ReadTokens:
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_token_ids(self.prompt_ids, "prompt_ids")
        _validate_token_ids(self.target_ids, "target_ids")
        if len(self.target_ids) < 2:
            raise ValueError("target_ids must contain at least one text token and EOS")


class LatentMemoryBackbone(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        bos_token_id: int,
        eos_token_id: int,
        d_mem: int,
        lora_rank: int,
        lora_alpha: int,
        lora_target_modules: tuple[str, ...],
        lora_dropout: float,
    ) -> None:
        super().__init__()
        if type(bos_token_id) is not int or bos_token_id < 0:
            raise ValueError("bos_token_id must be a non-negative integer")
        if type(eos_token_id) is not int or eos_token_id < 0:
            raise ValueError("eos_token_id must be a non-negative integer")
        if type(d_mem) is not int or d_mem <= 0:
            raise ValueError("d_mem must be a positive integer")

        hidden_size = getattr(base_model.config, "hidden_size", None)
        max_positions = getattr(base_model.config, "max_position_embeddings", None)
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("the base model config must define a positive hidden_size")
        if type(max_positions) is not int or max_positions <= 0:
            raise ValueError("the base model config must define positive max_position_embeddings")

        base_model.requires_grad_(False)
        base_model.config.use_cache = False
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.language_model: PeftModel = get_peft_model(base_model, lora_config)
        self.input_projection = nn.Linear(hidden_size, d_mem)
        self.memory_projection = nn.Linear(d_mem, hidden_size)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.d_lm = hidden_size
        self.d_mem = d_mem
        self.max_position_embeddings = max_positions

    def text_features(
        self,
        units: list[tuple[int, ...]],
        source_starts: list[int],
    ) -> list[Tensor]:
        if not units or len(units) != len(source_starts):
            raise ValueError("units and source_starts must align and be non-empty")
        device = self._model_device
        for unit, start in zip(units, source_starts, strict=True):
            _validate_token_ids(unit, "write unit")
            if type(start) is not int or start < 0:
                raise ValueError("source_starts must be non-negative integers")
        rows = [torch.tensor((self.bos_token_id, *unit), device=device) for unit in units]
        input_ids = pad_sequence(rows, batch_first=True, padding_value=self.eos_token_id)
        self._validate_sequence_length(input_ids.shape[1])
        lengths = torch.tensor([len(row) for row in rows], device=device)
        positions = torch.arange(input_ids.shape[1], device=device)[None, :]
        mask = positions < lengths[:, None]
        position_ids = positions.expand_as(input_ids).masked_fill(~mask, 0)
        with self._frozen_base():
            # Llama body returns only final hidden states; the vocabulary head is unused here.
            hidden = (
                self.language_model.get_base_model()
                .model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                .last_hidden_state.detach()
            )
        projected = self.input_projection(hidden[:, 1:].to(self.input_projection.weight.dtype))
        return [
            row[: len(unit)]
            + sinusoidal_positions(len(unit), self.d_mem, device, row.dtype, start=start)
            for row, unit, start in zip(projected, units, source_starts, strict=True)
        ]

    def read_batch(
        self,
        memories: list[Tensor],
        tokens: list[ReadTokens],
        text_contexts: list[tuple[int, ...]] | None = None,
        use_reader_lora: bool = True,
    ) -> list[ReaderOutput]:
        if not memories or len(memories) != len(tokens):
            raise ValueError("memories and read tokens must align and be non-empty")
        if text_contexts is None:
            text_contexts = [()] * len(tokens)
        if len(text_contexts) != len(tokens):
            raise ValueError("text contexts and read tokens must align")
        rows, contexts = [], []
        device = self._model_device
        embedding = self.language_model.get_input_embeddings()
        for memory, task, text_context in zip(memories, tokens, text_contexts, strict=True):
            self._validate_memory(memory)
            _validate_token_ids(text_context, "text context", allow_empty=True)
            if task.target_ids[-1] != self.eos_token_id:
                raise ValueError("target_ids must end with EOS")
            ids = torch.tensor(
                (self.bos_token_id, *text_context, *task.prompt_ids, *task.target_ids),
                device=device,
            )
            base = embedding(ids)
            projected = self.memory_projection(memory.to(self.memory_projection.weight.dtype))
            rows.append(torch.cat((base[:1], projected.to(base.dtype), base[1:])))
            contexts.append(1 + len(memory) + len(text_context) + len(task.prompt_ids))
        inputs_embeds = pad_sequence(rows, batch_first=True)
        self._validate_sequence_length(inputs_embeds.shape[1])
        positions = torch.arange(inputs_embeds.shape[1], device=device)[None, :]
        mask = positions < torch.tensor([len(row) for row in rows], device=device)[:, None]
        position_ids = positions.expand_as(mask).masked_fill(~mask, 0)
        with nullcontext() if use_reader_lora else self._frozen_base():
            output = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
        return [
            build_reader_output(
                row[context - 1 : context - 1 + len(task.target_ids)],
                torch.tensor(task.target_ids, device=device),
            )
            for row, context, task in zip(output.logits, contexts, tokens, strict=True)
        ]

    def greedy_students(
        self,
        memories: list[Tensor],
        prompts: list[tuple[int, ...]],
        token_limits: list[int],
    ) -> list[tuple[int, ...]]:
        if not memories or not len(memories) == len(prompts) == len(token_limits):
            raise ValueError("generation memories, prompts and token limits must align")
        if any(type(limit) is not int or limit <= 0 for limit in token_limits):
            raise ValueError("generation token limits must be positive integers")
        device = self._model_device
        embedding = self.language_model.get_input_embeddings()
        was_training = self.language_model.training
        self.language_model.eval()
        try:
            with torch.no_grad():
                rows = []
                for memory, prompt in zip(memories, prompts, strict=True):
                    self._validate_memory(memory)
                    _validate_token_ids(prompt, "prompt_ids")
                    ids = torch.tensor((self.bos_token_id, *prompt), device=device)
                    base = embedding(ids)
                    projected = self.memory_projection(
                        memory.to(self.memory_projection.weight.dtype)
                    )
                    rows.append(torch.cat((base[:1], projected.to(base.dtype), base[1:])))
                current = pad_sequence([row.flip(0) for row in rows], batch_first=True).flip(1)
                self._validate_sequence_length(current.shape[1] + max(token_limits))
                positions = torch.arange(current.shape[1], device=device)[None, :]
                lengths = torch.tensor([len(row) for row in rows], device=device)[:, None]
                attention_mask = (positions >= current.shape[1] - lengths).long()
                sequences = self.language_model.generate(
                    inputs_embeds=current,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=max(token_limits),
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.eos_token_id,
                    use_cache=True,
                )
                generated = []
                for sequence, limit in zip(sequences.tolist(), token_limits, strict=True):
                    tokens = sequence[:limit]
                    if self.eos_token_id in tokens:
                        tokens = tokens[: tokens.index(self.eos_token_id) + 1]
                    generated.append(tuple(tokens))
        finally:
            self.language_model.train(was_training)
        return generated

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.input_projection.parameters()
        yield from self.memory_projection.parameters()
        yield from (
            parameter for parameter in self.language_model.parameters() if parameter.requires_grad
        )

    def trainable_state_dict(self) -> dict[str, Any]:
        return {
            "input_projection": self.input_projection.state_dict(),
            "memory_projection": self.memory_projection.state_dict(),
            "reader_lora": get_peft_model_state_dict(
                self.language_model, save_embedding_layers=False
            ),
        }

    def load_trainable_state_dict(self, state: Mapping[str, Any]) -> None:
        actual = set(state)
        if actual != set(BACKBONE_TRAINABLE_STATE_FIELDS):
            missing = sorted(BACKBONE_TRAINABLE_STATE_FIELDS - actual)
            unknown = sorted(actual - BACKBONE_TRAINABLE_STATE_FIELDS)
            raise ValueError(
                f"invalid backbone trainable state; missing={missing}, unknown={unknown}"
            )
        self.input_projection.load_state_dict(state["input_projection"])
        self.memory_projection.load_state_dict(state["memory_projection"])
        reader_lora = state["reader_lora"]
        if not isinstance(reader_lora, Mapping):
            raise TypeError("reader_lora state must be a mapping")
        expected_lora = set(
            get_peft_model_state_dict(self.language_model, save_embedding_layers=False)
        )
        actual_lora = set(reader_lora)
        if actual_lora != expected_lora:
            missing = sorted(expected_lora - actual_lora)
            unknown = sorted(actual_lora - expected_lora)
            raise ValueError(f"invalid reader LoRA state; missing={missing}, unknown={unknown}")
        set_peft_model_state_dict(self.language_model, reader_lora)

    @property
    def _model_device(self) -> torch.device:
        return self.language_model.get_input_embeddings().weight.device

    @contextmanager
    def _frozen_base(self) -> Iterator[None]:
        was_training = self.language_model.training
        self.language_model.eval()
        try:
            with torch.no_grad(), self.language_model.disable_adapter():
                yield
        finally:
            self.language_model.train(was_training)

    def _validate_memory(self, memory: Tensor) -> None:
        if memory.ndim != 2 or memory.shape[1] != self.d_mem:
            raise ValueError(f"memory must have shape [num_slots, {self.d_mem}]")
        if not memory.is_floating_point():
            raise TypeError("memory must use a floating-point dtype")
        if memory.device != self.memory_projection.weight.device:
            raise ValueError("memory and backbone must be on the same device")

    def _validate_sequence_length(self, length: int) -> None:
        if length > self.max_position_embeddings:
            raise ValueError(
                f"sequence length {length} exceeds model limit {self.max_position_embeddings}"
            )


def load_backbone(
    config: ExperimentConfig,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedTokenizerBase, LatentMemoryBackbone]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
    )
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer must define BOS and EOS token IDs")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
        dtype=dtype,
    )
    if getattr(base_model.config, "model_type", None) != "llama":
        raise ValueError("v1 requires a Llama causal language model")
    if base_model.config.bos_token_id != tokenizer.bos_token_id:
        raise ValueError("model and tokenizer BOS token IDs do not match")
    if base_model.config.eos_token_id != tokenizer.eos_token_id:
        raise ValueError("model and tokenizer EOS token IDs do not match")
    if base_model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError("model vocabulary and tokenizer size do not match")

    base_model.to(device=device)
    if config.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    backbone = LatentMemoryBackbone(
        base_model=base_model,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        d_mem=config.d_mem,
        lora_rank=config.reader_lora_rank,
        lora_alpha=config.reader_lora_alpha,
        lora_target_modules=config.reader_lora_target_modules,
        lora_dropout=config.reader_lora_dropout,
    )
    backbone.input_projection.to(device=device)
    backbone.memory_projection.to(device=device)
    return tokenizer, backbone


def _validate_token_ids(
    token_ids: tuple[int, ...],
    label: str,
    allow_empty: bool = False,
) -> None:
    if not isinstance(token_ids, tuple):
        raise TypeError(f"{label} must be a tuple")
    if not token_ids and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise ValueError(f"{label} must contain non-negative integer token IDs")
