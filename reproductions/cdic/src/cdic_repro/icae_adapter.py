from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from icae.llama_icae_modeling import LlamaICAE, ModelArguments, TrainingArguments
from peft import LoraConfig

from cdic_repro.checkpoint import (
    ICAE_V1_LORA_RANK,
    load_icae_checkpoint_state_dict,
    restore_icae_checkpoint_state,
)
from cdic_repro.credit import CreditPlan, build_compression_gradient_plan
from cdic_repro.memory_state import ThreadState
from cdic_repro.model_protocol import CompressedTurn, TrainingLoss


@dataclass(frozen=True, slots=True)
class IcaeV1AdapterConfig:
    model_path: Path
    checkpoint_path: Path
    device: str = "cuda"
    devices: tuple[str, ...] = ()
    dtype: str = "bfloat16"
    memory_size: int = 128
    max_turn_tokens: int = 512
    max_new_tokens: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_rank: int | None = None
    seed: int = 42
    use_ft_markers: bool = True
    turn_template: str = "<s>[INST] {query} [/INST] {response} </s>"
    gradient_checkpointing: bool = False
    gradient_window_size: int = 1

    def __post_init__(self) -> None:
        if self.memory_size < 1:
            raise ValueError("memory_size must be positive")
        if self.max_turn_tokens < 1:
            raise ValueError("max_turn_tokens must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.dtype != "bfloat16":
            raise ValueError("dtype must be bfloat16 for the ICAE v1 reproduction")
        if self.lora_rank is not None and self.lora_rank < 1:
            raise ValueError("lora_rank must be positive")
        if self.gradient_window_size < 1:
            raise ValueError("gradient_window_size must be positive")
        if self.devices:
            if len(set(self.devices)) != len(self.devices):
                raise ValueError("devices must not contain duplicates")
            if self.device != self.devices[0]:
                raise ValueError("device must match the first entry in devices")
        if "{query}" not in self.turn_template or "{response}" not in self.turn_template:
            raise ValueError("turn_template must contain {query} and {response}")


class IcaeV1InferenceAdapter:
    """Inference-only ICAE v1 adapter for the C-DIC state machine."""

    def __init__(self, config: IcaeV1AdapterConfig, model: LlamaICAE) -> None:
        self.config = config
        self.model = model
        self.device = torch.device(config.device)
        self.model.eval()

    @classmethod
    def load(cls, config: IcaeV1AdapterConfig) -> IcaeV1InferenceAdapter:
        return cls(config=config, model=_load_icae_model(config, do_train=False))

    def encode_query(self, query: str) -> object:
        with torch.inference_mode():
            token_ids = self._tokenize(query, add_special_tokens=True)
            latent = self._compress_token_ids((), token_ids)
            return self._pool(latent).detach()

    def generate(self, supports: tuple[ThreadState, ...], query: str) -> str:
        with torch.inference_mode():
            decoder_input = self._build_decoder_input(supports, query)
            generated_ids: list[int] = []
            past_key_values = None
            current_input = decoder_input
            stop_token_id = int(self.model.eos_id)
            base_vocabulary_size = self.model.pad_token_id

            for _ in range(self.config.max_new_tokens):
                output = self.model.icae(
                    inputs_embeds=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    enable_lora=False,
                )
                next_token = torch.argmax(output.logits[:, -1, :], dim=-1)
                next_token_id = int(next_token.item())
                past_key_values = output.past_key_values
                if next_token_id == stop_token_id or next_token_id >= base_vocabulary_size:
                    break
                generated_ids.append(next_token_id)
                current_input = self._base_embeddings(next_token).unsqueeze(1)

            return self.model.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

    def compress(
        self,
        supports: tuple[ThreadState, ...],
        query: str,
        response: str,
    ) -> CompressedTurn:
        with torch.inference_mode():
            turn_text = self.config.turn_template.format(query=query, response=response)
            token_ids = self._tokenize(turn_text, add_special_tokens=False)
            latent = self._compress_token_ids(supports, token_ids)
            return CompressedTurn(
                latent=latent.detach(),
                retrieval_key=self._pool(latent).detach(),
                provenance=("generated-response",),
            )

    def load_trainable_state_dict(
        self,
        state_dict: Mapping[str, object],
        strict: bool = True,
    ) -> None:
        parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if _is_trainable_icae_parameter(name)
        }
        missing = sorted(set(parameters).difference(state_dict))
        unexpected = sorted(set(state_dict).difference(parameters))
        if strict and (missing or unexpected):
            raise ValueError(
                f"trainable state mismatch: missing={missing}, unexpected={unexpected}"
            )
        with torch.no_grad():
            for name, parameter in parameters.items():
                if name not in state_dict:
                    continue
                value = state_dict[name]
                if not hasattr(value, "shape") or tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"trainable parameter shape mismatch for {name}")
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    def _compress_token_ids(
        self,
        supports: tuple[ThreadState, ...],
        token_ids: list[int],
        gradient_state_id: str | None = None,
    ) -> Tensor:
        """将检索到的旧 latent states 和当前 turn 的文本压缩为新的 latent state.

        Args:
            supports: 检索到的旧 latent states.
            token_ids: 当前 turn 的文本的 token IDs.
            gradient_state_id: 允许保留计算图连接的旧 latent state ID, 为 None 时所有旧 latent states 均从计算图中分离.

        Returns:
            Tensor: 新的 latent state, 形状为 (memory_size, hidden_dim).
        """
        pieces = [
            self._support_embeddings(
                supports,
                gradient_state_ids=frozenset()
                if gradient_state_id is None
                else frozenset((gradient_state_id,)),
            ),
            self._embed_base_ids(token_ids),
        ]
        pieces.append(self._memory_token_embeddings())
        inputs_embeds = torch.cat([piece for piece in pieces if piece.shape[1] > 0], dim=1)
        self._validate_position_budget(inputs_embeds.shape[1])
        output = self.model.icae(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            enable_lora=True,
        )
        hidden_states = output.hidden_states[-1]
        if self.model.memory_head is not None:
            hidden_states = self.model.memory_head(hidden_states)
        latent = hidden_states[:, -self.config.memory_size :, :]
        if latent.shape[1] != self.config.memory_size:
            raise RuntimeError(f"unexpected compressed state shape: {tuple(latent.shape)}")
        return latent.squeeze(0)

    def _build_decoder_input(self, supports: tuple[ThreadState, ...], query: str) -> Any:
        support_embeddings = self._support_embeddings(supports)
        prompt_embeddings = self._query_prompt_embeddings(query)
        result = torch.cat((support_embeddings, prompt_embeddings), dim=1)
        self._validate_position_budget(result.shape[1])
        return result

    def _query_prompt_embeddings(self, query: str) -> Any:
        query_ids = self._tokenize(query, add_special_tokens=False)
        if self.config.use_ft_markers:
            prompt_ids = [self.model.ft_token_id, *query_ids, self.model.ft_token_id]
            return self._embed_mixed_ids(prompt_ids)
        return self._embed_base_ids(query_ids)

    def _support_embeddings(
        self,
        supports: tuple[ThreadState, ...],
        gradient_state_ids: frozenset[str] | None = None,
    ) -> Tensor:
        """将检索到的旧 latent states 整理并拼接成输入嵌入."""
        if gradient_state_ids is not None:
            unknown_state_ids = gradient_state_ids.difference(state.state_id for state in supports)
            if unknown_state_ids:
                raise ValueError(
                    f"gradient states are not present in supports: {sorted(unknown_state_ids)}"
                )
        if not supports:
            return torch.empty(
                (1, 0, self.model.dim),
                dtype=self._model_dtype(),
                device=self.device,
            )
        tensors = []
        for state in supports:
            latent = state.latent
            if not hasattr(latent, "shape") or tuple(latent.shape) != (
                self.config.memory_size,
                self.model.dim,
            ):
                raise ValueError(f"invalid latent shape for {state.state_id}")
            if gradient_state_ids is not None and state.state_id not in gradient_state_ids:
                # 仅保留明确纳入当前 gradient window 的 latent state, 其余 state
                # 从计算图中分离，避免梯度沿无关检索分支或窗口外 revision 继续传播。
                latent = latent.detach()
            tensors.append(latent.to(device=self.device, dtype=self._model_dtype()).unsqueeze(0))
        return torch.cat(tensors, dim=1)

    def _memory_token_embeddings(self) -> Any:
        token_indices = torch.arange(self.config.memory_size, device=self.device)
        return self.model.memory_token_embed(token_indices).to(self._model_dtype()).unsqueeze(0)

    def _embed_base_ids(self, token_ids: list[int]) -> Any:
        if not token_ids:
            return torch.empty(
                (1, 0, self.model.dim),
                dtype=self._model_dtype(),
                device=self.device,
            )
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        return self._base_embeddings(ids)

    def _embed_mixed_ids(self, token_ids: list[int]) -> Any:
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        special_mask = ids >= self.model.vocab_size
        safe_ids = ids.masked_fill(special_mask, 0)
        embeddings = self._base_embeddings(safe_ids)
        if special_mask.any():
            special_indices = ids[special_mask] - self.model.vocab_size
            embeddings[special_mask] = self.model.memory_token_embed(special_indices).to(
                embeddings.dtype
            )
        return embeddings

    def _base_embeddings(self, token_ids: Any) -> Any:
        return self.model.icae.get_base_model().model.embed_tokens(token_ids)

    def _tokenize(self, text: str, add_special_tokens: bool) -> list[int]:
        encoded = self.model.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=self.config.max_turn_tokens,
            padding=False,
            return_attention_mask=False,
        )
        return list(encoded["input_ids"])

    def _pool(self, latent: Any) -> Any:
        return latent.mean(dim=0)

    def _model_dtype(self) -> Any:
        return self.model.icae.get_base_model().model.embed_tokens.weight.dtype

    def _validate_position_budget(self, sequence_length: int) -> None:
        maximum = int(self.model.icae.config.max_position_embeddings)
        if sequence_length > maximum:
            raise ValueError(
                f"sequence length {sequence_length} exceeds model position budget {maximum}"
            )


class IcaeV1TrainingAdapter(IcaeV1InferenceAdapter):
    """Differentiable ICAE adapter for teacher-forced C-DIC training."""

    def __init__(self, config: IcaeV1AdapterConfig, model: LlamaICAE) -> None:
        super().__init__(config=config, model=model)
        _configure_trainable_parameters(self.model)
        self.model.train()

    @classmethod
    def load(cls, config: IcaeV1AdapterConfig) -> IcaeV1TrainingAdapter:
        return cls(config=config, model=_load_icae_model(config, do_train=True))

    def encode_query(self, query: str) -> object:
        was_training = self.model.training
        self.model.eval()
        try:
            return super().encode_query(query)
        finally:
            if was_training:
                self.model.train()

    @property
    def gradient_window_size(self) -> int:
        return self.config.gradient_window_size

    def response_loss(
        self,
        supports: tuple[ThreadState, ...],
        query: str,
        response: str,
        credit: CreditPlan,
        collect_token_nll: bool = False,
    ) -> TrainingLoss:
        support_embeddings = self._support_embeddings(
            supports,
            gradient_state_ids=frozenset()
            if credit.connected_state_id is None
            else frozenset((credit.connected_state_id,)),
        )
        prompt_embeddings = self._query_prompt_embeddings(query)
        response_ids = self._tokenize(response, add_special_tokens=False)
        response_ids.append(int(self.model.eos_id))
        response_embeddings = self._embed_base_ids(response_ids)
        inputs_embeds = torch.cat(
            (support_embeddings, prompt_embeddings, response_embeddings),
            dim=1,
        )
        self._validate_position_budget(inputs_embeds.shape[1])
        labels = torch.full(
            (1, inputs_embeds.shape[1]),
            -100,
            dtype=torch.long,
            device=self.device,
        )
        response_start = support_embeddings.shape[1] + prompt_embeddings.shape[1]
        labels[:, response_start:] = torch.tensor(
            [response_ids],
            dtype=torch.long,
            device=self.device,
        )
        output = self.model.icae(
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=False,
            enable_lora=False,
        )
        if output.loss is None:
            raise RuntimeError("frozen generator returned no teacher-forced loss")
        token_nll = None
        if collect_token_nll:
            # The last target is the ICAE stop token. Keep its NLL separately
            # identifiable without changing the existing training objective.
            token_losses = torch.nn.functional.cross_entropy(
                output.logits[0, response_start - 1 : -1].detach().float(),
                labels[0, response_start:],
                reduction="none",
            )
            token_nll = tuple(float(value) for value in token_losses.cpu().tolist())
        return TrainingLoss(value=output.loss, token_count=len(response_ids), token_nll=token_nll)

    def compress_gold(
        self,
        supports: tuple[ThreadState, ...],
        query: str,
        response: str,
        credit: CreditPlan,
    ) -> CompressedTurn:
        turn_text = self.config.turn_template.format(query=query, response=response)
        token_ids = self._tokenize(turn_text, add_special_tokens=False)
        gradient_plan = build_compression_gradient_plan(
            supports,
            credit,
            gradient_window_size=self.config.gradient_window_size,
        )
        latent = self._compress_token_ids(
            supports,
            token_ids,
            gradient_state_id=gradient_plan.retained_state_id,
        )
        return CompressedTurn(
            latent=latent,
            retrieval_key=self._pool(latent).detach(),
            provenance=("gold-response",),
            gradient_depth=gradient_plan.new_state_gradient_depth,
        )

    def trainable_state_dict(self) -> Mapping[str, object]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def trainable_parameter_report(self) -> dict[str, object]:
        records = [
            (name, parameter.numel())
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        return {
            "parameter_count": sum(count for _, count in records),
            "tensor_count": len(records),
            "names": [name for name, _ in records],
        }

    def gradient_coverage_report(self) -> dict[str, object]:
        groups = {
            "lora": {"trainable_tensors": 0, "gradient_tensors": 0},
            "compression_tokens": {"trainable_tensors": 0, "gradient_tensors": 0},
        }
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = "compression_tokens" if name.startswith("memory_token_embed.") else "lora"
            groups[group]["trainable_tensors"] += 1
            if parameter.grad is not None:
                groups[group]["gradient_tensors"] += 1
        return groups


def torch_cosine_similarity(left: object, right: object) -> float:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise TypeError("torch_cosine_similarity requires torch.Tensor inputs")
    left_vector = left.detach().float().reshape(-1)
    right_vector = right.detach().float().reshape(-1)
    if left_vector.shape != right_vector.shape:
        raise ValueError("retrieval-key shapes must match")
    similarity = torch.nn.functional.cosine_similarity(
        left_vector.unsqueeze(0),
        right_vector.unsqueeze(0),
        dim=-1,
    )
    return float(similarity.item())


def _load_icae_model(config: IcaeV1AdapterConfig, do_train: bool) -> LlamaICAE:
    state_dict = load_icae_checkpoint_state_dict(config.checkpoint_path)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the configured ICAE device")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.lora_rank is not None and config.lora_rank != ICAE_V1_LORA_RANK:
        raise ValueError(
            f"configured LoRA rank {config.lora_rank} does not match checkpoint rank "
            f"{ICAE_V1_LORA_RANK}"
        )

    model_arguments = ModelArguments(
        model_name_or_path=str(config.model_path),
        memory_head=False,
        better_transformer=False,
        mem_size=config.memory_size,
        lora_r=ICAE_V1_LORA_RANK,
        lora_dropout=config.lora_dropout,
    )
    training_arguments = TrainingArguments(
        output_dir="/tmp/cdic_icae_runtime",
        do_train=do_train,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        model_max_length=config.max_turn_tokens,
        report_to=[],
        disable_tqdm=True,
        gradient_checkpointing=do_train and config.gradient_checkpointing,
    )
    lora_config = LoraConfig(
        r=ICAE_V1_LORA_RANK,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = LlamaICAE(model_arguments, training_arguments, lora_config)
    restored_state, _ = restore_icae_checkpoint_state(
        state_dict,
        model.state_dict(),
    )
    model.load_state_dict(restored_state, strict=True)
    _validate_execution_devices((config.device,))
    model.to(config.device)
    if do_train and config.gradient_checkpointing:
        model.icae.gradient_checkpointing_enable()
        model.icae.config.use_cache = False
    return model


def _configure_trainable_parameters(model: Any) -> None:
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = _is_trainable_icae_parameter(name)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError("ICAE training adapter exposed no trainable parameters")


def _is_trainable_icae_parameter(name: str) -> bool:
    return name.startswith("memory_token_embed.") or ".lora_A." in name or ".lora_B." in name


def _validate_execution_devices(devices: tuple[str, ...]) -> None:
    for device in devices:
        parsed = torch.device(device)
        if parsed.type != "cuda":
            raise ValueError("ICAE execution devices must be CUDA devices")
        if parsed.index is not None and parsed.index >= torch.cuda.device_count():
            raise ValueError(f"configured CUDA device is unavailable: {device}")
