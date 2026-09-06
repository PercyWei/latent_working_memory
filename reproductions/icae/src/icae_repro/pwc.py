from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from icae_repro.inference import (
    InferenceConfig,
    compress_context,
    generate_answer,
    load_model,
)


class Condition(str, Enum):
    ICAE_128 = "icae-128"
    FULL_CONTEXT = "full-context"


@dataclass(frozen=True, slots=True)
class PwcRecord:
    sample_index: int
    sample_id: str
    context: str
    prompt: str
    answer: str


@dataclass(frozen=True, slots=True)
class BatchConfig:
    condition: Condition
    model_path: Path
    input_path: Path
    output_path: Path
    checkpoint_path: Path | None = None
    repository_root: Path = Path.cwd()
    device: str = "cuda"
    context_max_length: int = 512
    max_new_tokens: int = 512
    memory_size: int = 128
    seed: int = 42
    shard_index: int = 0
    num_shards: int = 1
    max_samples: int | None = None
    max_consecutive_errors: int = 3
    resume: bool = True

    def __post_init__(self) -> None:
        if self.condition is Condition.ICAE_128 and self.checkpoint_path is None:
            raise ValueError("ICAE condition requires checkpoint_path")
        if self.context_max_length < 1:
            raise ValueError("context_max_length must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.memory_size < 1:
            raise ValueError("memory_size must be positive")
        if self.num_shards < 1:
            raise ValueError("num_shards must be positive")
        if not 0 <= self.shard_index < self.num_shards:
            raise ValueError("shard_index must be within [0, num_shards)")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be positive when provided")
        if self.max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be positive")


class ConditionRunner(Protocol):
    load_report: Mapping[str, object]

    def run(self, record: PwcRecord) -> dict[str, object]: ...


class IcaeRunner:
    def __init__(self, config: BatchConfig) -> None:
        self.config = config
        inference_config = InferenceConfig(
            model_path=config.model_path,
            checkpoint_path=_require_checkpoint(config),
            context="",
            prompt="",
            device=config.device,
            memory_size=config.memory_size,
            model_max_length=config.context_max_length,
            max_new_tokens=config.max_new_tokens,
            repeat=1,
        )
        model, report = load_model(inference_config)
        self.model = model
        self.load_report = asdict(report)

    def run(self, record: PwcRecord) -> dict[str, object]:
        _reset_peak_memory(self.config.device)
        with torch.inference_mode():
            compression_started = time.perf_counter()
            memory, context_token_count = compress_context(
                self.model,
                record.context,
                device=self.config.device,
            )
            _synchronize(self.config.device)
            compression_seconds = time.perf_counter() - compression_started
            if not bool(torch.isfinite(memory).all().item()):
                raise RuntimeError("compressed memory contains NaN or Inf")

            generation_started = time.perf_counter()
            text, token_ids, prompt_token_count = generate_answer(
                self.model,
                memory,
                record.prompt,
                device=self.config.device,
                max_new_tokens=self.config.max_new_tokens,
            )
            _synchronize(self.config.device)
            generation_seconds = time.perf_counter() - generation_started

        return {
            "output": text,
            "output_token_ids": token_ids,
            "context_token_count": context_token_count,
            "prompt_token_count": prompt_token_count,
            "memory_shape": list(memory.shape),
            "memory_dtype": str(memory.dtype),
            "compression_seconds": compression_seconds,
            "generation_seconds": generation_seconds,
            "peak_cuda_memory_bytes": _peak_memory(self.config.device),
        }


class FullContextRunner:
    def __init__(self, config: BatchConfig) -> None:
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.config = config
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            local_files_only=True,
            use_fast=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(config.device)
        self.model.eval()
        self.load_report = {
            "model_class": type(self.model).__name__,
            "dtype": str(self.model.dtype),
            "prompt_protocol": "raw context tokens followed by raw prompt tokens",
        }

    def run(self, record: PwcRecord) -> dict[str, object]:
        context_ids = self.tokenizer(
            record.context,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.context_max_length,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        prompt_ids = self.tokenizer(
            record.prompt,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        input_ids = torch.tensor(
            [[*context_ids, *prompt_ids]],
            dtype=torch.long,
            device=self.config.device,
        )
        maximum = int(self.model.config.max_position_embeddings)
        if input_ids.shape[1] > maximum:
            raise ValueError(
                f"full-context input length {input_ids.shape[1]} exceeds model limit {maximum}"
            )

        _reset_peak_memory(self.config.device)
        with torch.inference_mode():
            generation_started = time.perf_counter()
            token_ids = _greedy_generate(
                model=self.model,
                input_ids=input_ids,
                max_new_tokens=self.config.max_new_tokens,
                stop_token_id=self.tokenizer.eos_token_id,
            )
            _synchronize(self.config.device)
            generation_seconds = time.perf_counter() - generation_started

        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {
            "output": text,
            "output_token_ids": token_ids,
            "context_token_count": len(context_ids),
            "prompt_token_count": len(prompt_ids),
            "memory_shape": None,
            "memory_dtype": None,
            "compression_seconds": 0.0,
            "generation_seconds": generation_seconds,
            "peak_cuda_memory_bytes": _peak_memory(self.config.device),
        }


def parse_record(value: object, sample_index: int) -> PwcRecord:
    if not isinstance(value, Mapping):
        raise TypeError("PwC record must be a JSON object")
    required = ("input", "prompt", "answer")
    missing = [key for key in required if not isinstance(value.get(key), str)]
    if missing:
        raise ValueError(f"PwC record has missing or non-string fields: {missing}")
    raw_id = value.get("id", f"pwc-{sample_index:06d}")
    return PwcRecord(
        sample_index=sample_index,
        sample_id=str(raw_id),
        context=str(value["input"]),
        prompt=str(value["prompt"]),
        answer=str(value["answer"]),
    )


def iter_records(config: BatchConfig, completed_ids: set[str]) -> Iterator[PwcRecord]:
    yielded = 0
    with config.input_path.open(encoding="utf-8") as source:
        for sample_index, line in enumerate(source):
            if sample_index % config.num_shards != config.shard_index:
                continue
            record = parse_record(json.loads(line), sample_index=sample_index)
            if record.sample_id in completed_ids:
                continue
            if config.max_samples is not None and yielded >= config.max_samples:
                break
            yielded += 1
            yield record


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            sample_id = record.get("id")
            if not isinstance(sample_id, str):
                raise ValueError(f"missing string id in {path} line {line_number}")
            if sample_id in completed:
                raise ValueError(f"duplicate id {sample_id!r} in {path}")
            completed.add(sample_id)
    return completed


def run_batch(config: BatchConfig) -> dict[str, object]:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = config.output_path.with_suffix(".errors.jsonl")
    manifest_path = config.output_path.with_suffix(".manifest.json")
    completed_ids = load_completed_ids(config.output_path) if config.resume else set()
    if not config.resume and config.output_path.exists():
        raise FileExistsError(f"output exists and resume is disabled: {config.output_path}")

    load_started = time.perf_counter()
    runner: ConditionRunner
    if config.condition is Condition.ICAE_128:
        runner = IcaeRunner(config)
    else:
        runner = FullContextRunner(config)
    load_seconds = time.perf_counter() - load_started

    manifest = _manifest(config, runner.load_report, load_seconds=load_seconds, status="running")
    _write_json(manifest_path, manifest)
    processed = 0
    failed = 0
    consecutive_errors = 0
    started = time.perf_counter()
    with (
        config.output_path.open("a", encoding="utf-8") as output,
        error_path.open("a", encoding="utf-8") as errors,
    ):
        for record in iter_records(config, completed_ids):
            try:
                result = runner.run(record)
                output.write(
                    json.dumps(
                        {
                            "id": record.sample_id,
                            "sample_index": record.sample_index,
                            "condition": config.condition.value,
                            "input": record.context,
                            "prompt": record.prompt,
                            "answer": record.answer,
                            **result,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output.flush()
                processed += 1
                consecutive_errors = 0
            except Exception as error:
                errors.write(
                    json.dumps(
                        {
                            "id": record.sample_id,
                            "sample_index": record.sample_index,
                            "condition": config.condition.value,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                errors.flush()
                failed += 1
                consecutive_errors += 1
                if consecutive_errors >= config.max_consecutive_errors:
                    raise RuntimeError(
                        f"stopped after {consecutive_errors} consecutive sample errors"
                    ) from error

    summary = {
        "status": "complete",
        "processed_this_run": processed,
        "previously_completed": len(completed_ids),
        "failed_this_run": failed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(
        manifest_path,
        {
            **manifest,
            **summary,
        },
    )
    return summary


def _greedy_generate(
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
    stop_token_id: int | None,
) -> list[int]:
    if stop_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    generated_ids: list[int] = []
    current_ids = input_ids
    past_key_values = None
    for _ in range(max_new_tokens):
        output = model(
            input_ids=current_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = torch.argmax(output.logits[:, -1, :], dim=-1)
        next_token_id = int(next_token.item())
        past_key_values = output.past_key_values
        if next_token_id == stop_token_id:
            break
        generated_ids.append(next_token_id)
        current_ids = next_token.unsqueeze(0)
    return generated_ids


def _manifest(
    config: BatchConfig,
    load_report: Mapping[str, object],
    load_seconds: float,
    status: str,
) -> dict[str, object]:
    return {
        "status": status,
        "condition": config.condition.value,
        "config": {
            **asdict(config),
            "condition": config.condition.value,
            "model_path": str(config.model_path),
            "input_path": str(config.input_path),
            "output_path": str(config.output_path),
            "checkpoint_path": (
                str(config.checkpoint_path) if config.checkpoint_path is not None else None
            ),
            "repository_root": str(config.repository_root),
        },
        "git": _git_state(config.repository_root),
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": (
                torch.cuda.get_device_name() if config.device.startswith("cuda") else None
            ),
        },
        "load_report": dict(load_report),
        "load_seconds": load_seconds,
        "started_at_unix": time.time(),
    }


def _git_state(repository_root: Path) -> dict[str, object]:
    commit = _git(repository_root, "rev-parse", "HEAD")
    status = _git(repository_root, "status", "--short")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_checkpoint(config: BatchConfig) -> Path:
    if config.checkpoint_path is None:
        raise ValueError("checkpoint_path is required")
    return config.checkpoint_path


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _peak_memory(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    return int(torch.cuda.max_memory_allocated())


def _reset_peak_memory(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PwC predictions for ICAE reproduction")
    parser.add_argument("--condition", required=True, choices=[item.value for item in Condition])
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--memory-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()

    summary = run_batch(
        BatchConfig(
            condition=Condition(arguments.condition),
            model_path=arguments.model_path,
            checkpoint_path=arguments.checkpoint,
            input_path=arguments.input,
            output_path=arguments.output,
            repository_root=arguments.repository_root,
            device=arguments.device,
            context_max_length=arguments.context_max_length,
            max_new_tokens=arguments.max_new_tokens,
            memory_size=arguments.memory_size,
            seed=arguments.seed,
            shard_index=arguments.shard_index,
            num_shards=arguments.num_shards,
            max_samples=arguments.max_samples,
            max_consecutive_errors=arguments.max_consecutive_errors,
            resume=not arguments.no_resume,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
