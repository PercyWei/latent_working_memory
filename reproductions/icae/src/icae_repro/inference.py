from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from icae.llama_icae_modeling import LlamaICAE, ModelArguments, TrainingArguments
from peft import LoraConfig

from icae_repro.checkpoint import (
    ICAE_V1_LORA_RANK,
    load_checkpoint_state_dict,
    load_zero_placeholder_checkpoint,
)


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    model_path: Path
    checkpoint_path: Path
    context: str
    prompt: str
    device: str = "cuda"
    memory_size: int = 128
    model_max_length: int = 512
    max_new_tokens: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    repeat: int = 2

    def __post_init__(self) -> None:
        if self.memory_size < 1:
            raise ValueError("memory_size must be positive")
        if self.model_max_length < 1:
            raise ValueError("model_max_length must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.repeat < 1:
            raise ValueError("repeat must be positive")


def load_model(config: InferenceConfig) -> tuple[LlamaICAE, object]:
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    raw_checkpoint = load_checkpoint_state_dict(config.checkpoint_path)
    model_arguments = ModelArguments(
        model_name_or_path=str(config.model_path),
        memory_head=False,
        better_transformer=False,
        mem_size=config.memory_size,
        lora_r=ICAE_V1_LORA_RANK,
        lora_dropout=config.lora_dropout,
    )
    training_arguments = TrainingArguments(
        output_dir="/tmp/icae-inference",
        do_train=False,
        bf16=True,
        per_device_train_batch_size=1,
        model_max_length=config.model_max_length,
        report_to=[],
        disable_tqdm=True,
    )
    lora_config = LoraConfig(
        r=ICAE_V1_LORA_RANK,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model = LlamaICAE(model_arguments, training_arguments, lora_config)
    report = load_zero_placeholder_checkpoint(model, raw_checkpoint)
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            f"strict checkpoint load failed: missing={report.missing_keys}, "
            f"unexpected={report.unexpected_keys}"
        )
    model.to(config.device)
    model.eval()
    return model, report


def compress_context(model: Any, context: str, device: str) -> tuple[Any, int]:
    encoded = model.tokenizer(
        context,
        truncation=True,
        max_length=model.training_args.model_max_length,
        padding=False,
        return_attention_mask=False,
    )
    context_token_count = len(encoded["input_ids"])
    memory_token_ids = list(range(model.vocab_size, model.vocab_size + model.model_args.mem_size))
    input_ids = torch.tensor(
        [encoded["input_ids"] + memory_token_ids], dtype=torch.long, device=device
    )
    memory_mask = input_ids >= model.vocab_size
    embeddings = model.icae.get_base_model().model.embed_tokens(input_ids)
    embeddings[memory_mask] = model.memory_token_embed(
        input_ids[memory_mask] - model.vocab_size
    ).to(embeddings)
    output = model.icae(
        inputs_embeds=embeddings,
        output_hidden_states=True,
        enable_lora=True,
    )
    hidden_states = output.hidden_states[-1]
    if model.memory_head is not None:
        hidden_states = model.memory_head(hidden_states)
    memory = hidden_states[memory_mask].view(1, model.model_args.mem_size, model.dim)
    return memory, context_token_count


def generate_answer(
    model: Any,
    memory: Any,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> tuple[str, list[int], int]:
    prompt_ids = model.tokenizer(
        prompt,
        add_special_tokens=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    mixed_ids = torch.tensor(
        [[model.ft_token_id, *prompt_ids, model.ft_token_id]],
        dtype=torch.long,
        device=device,
    )
    special_mask = mixed_ids >= model.vocab_size
    safe_ids = mixed_ids.masked_fill(special_mask, 0)
    prompt_embeddings = model.icae.get_base_model().model.embed_tokens(safe_ids)
    prompt_embeddings[special_mask] = model.memory_token_embed(
        mixed_ids[special_mask] - model.vocab_size
    ).to(prompt_embeddings)
    current_input = torch.cat((memory, prompt_embeddings), dim=1)
    past_key_values = None
    generated_ids: list[int] = []
    stop_token_id = int(model.eos_id)
    base_vocabulary_size = model.pad_token_id

    for _ in range(max_new_tokens):
        output = model.icae(
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
        current_input = model.icae.get_base_model().model.embed_tokens(next_token).unsqueeze(1)

    text = model.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text, generated_ids, len(prompt_ids)


def run_smoke(config: InferenceConfig) -> dict[str, object]:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    load_started = time.perf_counter()
    model, checkpoint_report = load_model(config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        compression_started = time.perf_counter()
        memory, context_token_count = compress_context(model, config.context, device=config.device)
        torch.cuda.synchronize()
        compression_seconds = time.perf_counter() - compression_started
        memory_is_finite = bool(torch.isfinite(memory).all().item())

        generations = []
        generation_token_ids = []
        generation_seconds = []
        prompt_token_count = 0
        for _ in range(config.repeat):
            generation_started = time.perf_counter()
            text, token_ids, prompt_token_count = generate_answer(
                model,
                memory,
                config.prompt,
                device=config.device,
                max_new_tokens=config.max_new_tokens,
            )
            torch.cuda.synchronize()
            generations.append(text)
            generation_token_ids.append(token_ids)
            generation_seconds.append(time.perf_counter() - generation_started)

    return {
        "config": {
            **asdict(config),
            "model_path": str(config.model_path),
            "checkpoint_path": str(config.checkpoint_path),
        },
        "checkpoint": asdict(checkpoint_report),
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(),
        },
        "context_token_count": context_token_count,
        "prompt_token_count": prompt_token_count,
        "stop_token_id": int(model.eos_id),
        "memory_shape": list(memory.shape),
        "memory_dtype": str(memory.dtype),
        "memory_is_finite": memory_is_finite,
        "generations": generations,
        "generation_token_ids": generation_token_ids,
        "deterministic": all(ids == generation_token_ids[0] for ids in generation_token_ids[1:]),
        "load_seconds": load_seconds,
        "compression_seconds": compression_seconds,
        "generation_seconds": generation_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ICAE v1 end-to-end smoke test")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--context", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=2)
    arguments = parser.parse_args()

    result = run_smoke(
        InferenceConfig(
            model_path=arguments.model_path,
            checkpoint_path=arguments.checkpoint,
            context=arguments.context,
            prompt=arguments.prompt,
            device=arguments.device,
            max_new_tokens=arguments.max_new_tokens,
            repeat=arguments.repeat,
        )
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if arguments.output is None:
        print(serialized)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
