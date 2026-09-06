from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from icae.llama_icae_modeling import LlamaICAE, ModelArguments, TrainingArguments
from icae_repro.checkpoint import restore_zero_weight_state_dict
from peft import LoraConfig

from cdic_repro.checkpoint import infer_lora_rank, load_checkpoint_state_dict
from cdic_repro.credit import CreditPlan
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

    def __post_init__(self) -> None:
        if self.memory_size < 1:
            raise ValueError("memory_size must be positive")
        if self.max_turn_tokens < 1:
            raise ValueError("max_turn_tokens must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.lora_rank is not None and self.lora_rank < 1:
            raise ValueError("lora_rank must be positive")
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
            stop_token_id = resolve_stop_token_id(self.model)
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
            turn_text = format_turn(self.config.turn_template, query=query, response=response)
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
        detach_supports: bool = False,
    ) -> Any:
        pieces = [
            self._support_embeddings(supports, detach_all=detach_supports),
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
        connected_state_id: str | None = None,
        detach_unconnected: bool = False,
        detach_all: bool = False,
    ) -> Any:
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
            if detach_all or (detach_unconnected and state.state_id != connected_state_id):
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

    def response_loss(
        self,
        supports: tuple[ThreadState, ...],
        query: str,
        response: str,
        credit: CreditPlan,
    ) -> TrainingLoss:
        support_embeddings = self._support_embeddings(
            supports,
            connected_state_id=credit.connected_state_id,
            detach_unconnected=True,
        )
        prompt_embeddings = self._query_prompt_embeddings(query)
        response_ids = self._tokenize(response, add_special_tokens=False)
        response_ids.append(resolve_stop_token_id(self.model))
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
        return TrainingLoss(value=output.loss, token_count=len(response_ids))

    def compress_gold(
        self,
        supports: tuple[ThreadState, ...],
        query: str,
        response: str,
    ) -> CompressedTurn:
        turn_text = format_turn(self.config.turn_template, query=query, response=response)
        token_ids = self._tokenize(turn_text, add_special_tokens=False)
        latent = self._compress_token_ids(supports, token_ids, detach_supports=True)
        return CompressedTurn(
            latent=latent,
            retrieval_key=self._pool(latent).detach(),
            provenance=("gold-response",),
        )

    def backward(self, loss: object, scale: float) -> None:
        if scale <= 0.0:
            raise ValueError("loss scale must be positive")
        (loss * scale).backward()  # type: ignore[operator, union-attr]

    def loss_requires_grad(self, loss: object) -> bool:
        return bool(getattr(loss, "requires_grad", False))

    def loss_to_float(self, loss: object) -> float:
        return float(loss.detach().float().item())  # type: ignore[union-attr]

    def trainable_parameters(self) -> Iterable[object]:
        return tuple(parameter for parameter in self.model.parameters() if parameter.requires_grad)

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


def format_turn(template: str, query: str, response: str) -> str:
    return template.format(query=query, response=response)


def resolve_stop_token_id(model: Any) -> int:
    stop_token_id = getattr(model, "eos_id", None)
    if stop_token_id is None:
        stop_token_id = model.tokenizer.eos_token_id
    if stop_token_id is None:
        raise ValueError("no generation stop token is configured")
    return int(stop_token_id)


def _resolve_dtype(dtype: str) -> torch.dtype:
    supported = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return supported[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {dtype}") from error


def _load_icae_model(config: IcaeV1AdapterConfig, do_train: bool) -> LlamaICAE:
    state_dict = load_checkpoint_state_dict(config.checkpoint_path)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the configured ICAE device")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    inferred_rank = infer_lora_rank(state_dict)
    if config.lora_rank is not None and config.lora_rank != inferred_rank:
        raise ValueError(
            f"configured LoRA rank {config.lora_rank} does not match checkpoint rank "
            f"{inferred_rank}"
        )

    dtype = _resolve_dtype(config.dtype)
    model_arguments = ModelArguments(
        model_name_or_path=str(config.model_path),
        memory_head=False,
        better_transformer=False,
        mem_size=config.memory_size,
        lora_r=inferred_rank,
        lora_dropout=config.lora_dropout,
    )
    training_arguments = TrainingArguments(
        output_dir="/tmp/cdic_icae_runtime",
        do_train=do_train,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        per_device_train_batch_size=1,
        model_max_length=config.max_turn_tokens,
        report_to=[],
        disable_tqdm=True,
        gradient_checkpointing=do_train and config.gradient_checkpointing,
    )
    lora_config = LoraConfig(
        r=inferred_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = LlamaICAE(model_arguments, training_arguments, lora_config)
    restored_state, _ = restore_zero_weight_state_dict(
        state_dict,
        model.state_dict(),
        is_tensor=torch.is_tensor,
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
