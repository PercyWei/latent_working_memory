from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.data import EncoderCell
from latent_working_memory.v1.model import sinusoidal_positions
from latent_working_memory.v1.objectives import ReaderOutput, build_reader_output


BACKBONE_TRAINABLE_STATE_FIELDS = frozenset(
    {"input_projection", "memory_projection", "reader_lora"}
)


@dataclass(frozen=True, slots=True)
class QuestionAnswerTokens:
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_token_ids(self.prompt_ids, "prompt_ids")
        _validate_token_ids(self.target_ids, "target_ids")
        if len(self.target_ids) < 2:
            raise ValueError("target_ids must contain at least one answer token and EOS")

    @property
    def answer_token_count(self) -> int:
        return len(self.target_ids) - 1


@dataclass(frozen=True, slots=True)
class CellEncoding:
    hidden_states: Tensor
    source_start: int

    def __post_init__(self) -> None:
        if self.hidden_states.ndim != 2 or self.hidden_states.shape[0] <= 0:
            raise ValueError("hidden_states must have shape [token_count, d_lm]")
        if not self.hidden_states.is_floating_point():
            raise TypeError("hidden_states must use a floating-point dtype")
        if self.hidden_states.requires_grad:
            raise ValueError("frozen cell hidden_states must not require gradients")
        if type(self.source_start) is not int or self.source_start < 0:
            raise ValueError("source_start must be a non-negative integer")

    @property
    def token_count(self) -> int:
        return self.hidden_states.shape[0]

    @property
    def source_end(self) -> int:
        return self.source_start + self.token_count


def tokenize_question_answer(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    answer: str,
) -> QuestionAnswerTokens:
    if not isinstance(question, str) or not question:
        raise ValueError("question must be a non-empty string")
    if not isinstance(answer, str) or not answer:
        raise ValueError("answer must be a non-empty string")
    if tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer must define eos_token_id")

    prompt_ids = tuple(tokenizer.encode(f"Question: {question}\nAnswer:", add_special_tokens=False))
    answer_ids = tuple(tokenizer.encode(f" {answer}", add_special_tokens=False))
    if not prompt_ids:
        raise ValueError("the serialized question prompt must produce at least one token")
    if not answer_ids:
        raise ValueError("the serialized answer must produce at least one token")
    return QuestionAnswerTokens(prompt_ids, answer_ids + (tokenizer.eos_token_id,))


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

    def frozen_cell_encoding(self, cells: tuple[EncoderCell, ...]) -> CellEncoding:
        self._validate_cells(cells)
        device = self._model_device
        hidden_rows: list[Tensor] = []
        with self._frozen_base():
            for cell in cells:
                input_ids = torch.tensor(
                    (self.bos_token_id, *cell.input_ids),
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0)
                self._validate_sequence_length(input_ids.shape[1])
                position_ids = torch.arange(
                    input_ids.shape[1], device=device, dtype=torch.long
                ).unsqueeze(0)
                output = self.language_model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hidden_rows.append(output.hidden_states[-1][0, 1:].detach())
        return CellEncoding(torch.cat(hidden_rows, dim=0), cells[0].source_start)

    def project_cell_encoding(self, encoding: CellEncoding) -> Tensor:
        if encoding.hidden_states.shape[1] != self.d_lm:
            raise ValueError(f"cell hidden width must be {self.d_lm}")
        hidden_states = encoding.hidden_states.to(
            device=self.input_projection.weight.device,
            dtype=self.input_projection.weight.dtype,
        )
        projected = self.input_projection(hidden_states)
        positions = sinusoidal_positions(
            encoding.token_count,
            self.d_mem,
            projected.device,
            projected.dtype,
            start=encoding.source_start,
        )
        return projected + positions

    def teacher_input_ids(
        self,
        prefix_ids: tuple[int, ...],
        tokens: QuestionAnswerTokens,
    ) -> tuple[int, ...]:
        _validate_token_ids(prefix_ids, "prefix_ids", allow_empty=True)
        self._validate_qa_tokens(tokens)
        return (self.bos_token_id, *prefix_ids, *tokens.prompt_ids, *tokens.target_ids)

    def teacher_output(
        self,
        prefix_ids: tuple[int, ...],
        tokens: QuestionAnswerTokens,
    ) -> ReaderOutput:
        serialized = self.teacher_input_ids(prefix_ids, tokens)
        self._validate_sequence_length(len(serialized))
        device = self._model_device
        input_ids = torch.tensor(serialized, device=device, dtype=torch.long).unsqueeze(0)
        position_ids = torch.arange(len(serialized), device=device, dtype=torch.long).unsqueeze(0)
        context_length = 1 + len(prefix_ids) + len(tokens.prompt_ids)
        with self._frozen_base():
            output = self.language_model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            target_logits = _answer_relative_logits(
                output.logits,
                context_length,
                len(tokens.target_ids),
            ).detach()
        target_ids = torch.tensor(tokens.target_ids, device=device, dtype=torch.long)
        return build_reader_output(target_logits, target_ids)

    def student_output(
        self,
        memory: Tensor,
        tokens: QuestionAnswerTokens,
    ) -> ReaderOutput:
        self._validate_memory(memory)
        self._validate_qa_tokens(tokens)
        device = self._model_device
        base_ids = torch.tensor(
            (self.bos_token_id, *tokens.prompt_ids, *tokens.target_ids),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        embedding = self.language_model.get_input_embeddings()
        base_embeddings = embedding(base_ids).squeeze(0)
        projected_memory = self.memory_projection(
            memory.to(
                device=self.memory_projection.weight.device,
                dtype=self.memory_projection.weight.dtype,
            )
        ).to(dtype=base_embeddings.dtype)
        inputs_embeds = torch.cat(
            (base_embeddings[:1], projected_memory, base_embeddings[1:]),
            dim=0,
        ).unsqueeze(0)
        self._validate_sequence_length(inputs_embeds.shape[1])
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=device, dtype=torch.long
        ).unsqueeze(0)
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        context_length = 1 + memory.shape[0] + len(tokens.prompt_ids)
        target_logits = _answer_relative_logits(
            output.logits,
            context_length,
            len(tokens.target_ids),
        )
        target_ids = torch.tensor(tokens.target_ids, device=device, dtype=torch.long)
        return build_reader_output(target_logits, target_ids)

    def greedy_student(
        self,
        memory: Tensor,
        prompt_ids: tuple[int, ...],
        max_new_tokens: int,
    ) -> tuple[int, ...]:
        self._validate_memory(memory)
        _validate_token_ids(prompt_ids, "prompt_ids")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        context_length = 1 + memory.shape[0] + len(prompt_ids)
        self._validate_sequence_length(context_length + max_new_tokens)

        device = self._model_device
        embedding = self.language_model.get_input_embeddings()
        generated: list[int] = []
        was_training = self.language_model.training
        self.language_model.eval()
        try:
            with torch.no_grad():
                base_ids = torch.tensor(
                    (self.bos_token_id, *prompt_ids),
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0)
                base_embeddings = embedding(base_ids).squeeze(0)
                projected_memory = self.memory_projection(
                    memory.to(
                        device=self.memory_projection.weight.device,
                        dtype=self.memory_projection.weight.dtype,
                    )
                ).to(dtype=base_embeddings.dtype)
                current = torch.cat(
                    (base_embeddings[:1], projected_memory, base_embeddings[1:]),
                    dim=0,
                ).unsqueeze(0)
                position_ids = torch.arange(
                    current.shape[1], device=device, dtype=torch.long
                ).unsqueeze(0)
                attention_mask = torch.ones_like(position_ids)
                sequences = self.language_model.generate(
                    inputs_embeds=current,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.eos_token_id,
                    use_cache=True,
                )
                generated.extend(int(token_id) for token_id in sequences[0].tolist())
        finally:
            self.language_model.train(was_training)
        return tuple(generated)

    def greedy_teacher(
        self,
        prefix_ids: tuple[int, ...],
        prompt_ids: tuple[int, ...],
        max_new_tokens: int,
    ) -> tuple[int, ...]:
        _validate_token_ids(prefix_ids, "prefix_ids", allow_empty=True)
        _validate_token_ids(prompt_ids, "prompt_ids")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        context_ids = (self.bos_token_id, *prefix_ids, *prompt_ids)
        self._validate_sequence_length(len(context_ids) + max_new_tokens)
        device = self._model_device
        input_ids = torch.tensor(context_ids, device=device, dtype=torch.long).unsqueeze(0)
        position_ids = torch.arange(input_ids.shape[1], device=device, dtype=torch.long).unsqueeze(
            0
        )
        attention_mask = torch.ones_like(input_ids)
        with self._frozen_base():
            sequences = self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.eos_token_id,
                use_cache=True,
            )
        generated = sequences[0, input_ids.shape[1] :]
        return tuple(int(token_id) for token_id in generated.tolist())

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

    def _validate_cells(self, cells: tuple[EncoderCell, ...]) -> None:
        if not cells:
            raise ValueError("cells must not be empty")
        for previous, current in zip(cells, cells[1:], strict=False):
            if previous.source_end != current.source_start:
                raise ValueError("cells must be contiguous and ordered")

    def _validate_qa_tokens(self, tokens: QuestionAnswerTokens) -> None:
        if tokens.target_ids[-1] != self.eos_token_id:
            raise ValueError("target_ids must end with the backbone EOS token")

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


def _answer_relative_logits(logits: Tensor, context_length: int, target_length: int) -> Tensor:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("causal LM logits must have shape [1, sequence_length, vocab_size]")
    start = context_length - 1
    end = start + target_length
    target_logits = logits[0, start:end]
    if target_logits.shape[0] != target_length:
        raise ValueError("causal LM output is shorter than the requested answer positions")
    return target_logits


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
