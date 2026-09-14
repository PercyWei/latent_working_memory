"""Paired local evaluation benchmark; identical panel, requests and generation limits."""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import platform
import statistics
from tempfile import TemporaryDirectory
import time
import argparse

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.pretrain.evaluation import evaluate_pretraining
from evaluation_reference import evaluate_pretraining_reference
from test_evaluation_batching import panel_index, assert_results_equal
from latent_working_memory.v1.backbone import LatentMemoryBackbone
from latent_working_memory.v1.data import EpisodeIndex, write_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--long", action="store_true", help="Use 8x longer texts to inspect decode scheduling"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(913)
    vocab = {
        token: i
        for i, token in enumerate(
            ["<pad>", "<s>", "</s>", "<unk>"] + [f"t{i}" for i in range(4, 48)]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    config = replace(
        ExperimentConfig(),
        d_mem=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        k_limit=32,
        max_input_tokens=64,
        max_continuation_tokens=64,
        write_context_tokens=256,
        read_context_tokens=256,
        input_length_bounds=(8, 16, 32, 64),
        eval_examples=24,
        eval_generation_examples=12,
    )
    base = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=48,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    )
    backbone = LatentMemoryBackbone(base, 1, 2, 8, 2, 4, ("q_proj", "v_proj"), 0.0)
    with torch.no_grad():
        for name, parameter in backbone.language_model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.05)
    if args.long:
        config = replace(
            config,
            max_input_tokens=256,
            max_continuation_tokens=256,
            write_context_tokens=1024,
            read_context_tokens=1024,
            k_limit=128,
            input_length_bounds=(32, 64, 128, 256),
        )
    writer = JointMemoryWriter(8, 1, 2, 16, config.k_limit)
    original_read = backbone.read_batch
    original_generate = backbone.greedy_students
    statistics_rows = {}
    generated = []

    def read(memories, tasks, text_contexts=None, use_reader_lora=True):
        contexts = text_contexts if text_contexts is not None else [()] * len(tasks)
        lengths = [
            1 + len(m) + len(c) + len(t.prompt_ids) + len(t.target_ids)
            for m, c, t in zip(memories, contexts, tasks, strict=True)
        ]
        statistics_rows["read_calls"] += 1
        statistics_rows["read_requests"] += len(tasks)
        statistics_rows["read_padded_positions"] += len(tasks) * max(lengths)
        start = time.perf_counter()
        result = original_read(memories, tasks, text_contexts, use_reader_lora)
        statistics_rows["read_seconds"] += time.perf_counter() - start
        return result

    def generate(memories, prompts, limits, use_reader_lora=True):
        lengths = [1 + len(m) + len(p) for m, p in zip(memories, prompts, strict=True)]
        statistics_rows["generation_calls"] += 1
        statistics_rows["generation_requests"] += len(limits)
        statistics_rows["generation_prefill_positions"] += len(limits) * max(lengths)
        statistics_rows["generation_decode_budget"] += len(limits) * max(limits)
        start = time.perf_counter()
        result = original_generate(memories, prompts, limits, use_reader_lora)
        statistics_rows["generation_seconds"] += time.perf_counter() - start
        generated.extend(
            (tuple(m.shape), m.float().numpy().tobytes(), p, limit, use_reader_lora, ids)
            for m, p, limit, ids in zip(memories, prompts, limits, result, strict=True)
        )
        return result

    backbone.read_batch = read
    backbone.greedy_students = generate
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    expected_metrics = None
    expected_rows = None
    expected_tokens = None
    with TemporaryDirectory(prefix="eval-batching-") as tmp:
        root = Path(tmp)
        index = panel_index(root / "input.jsonl", tokenizer)
        if args.long:
            episodes = []
            for i in range(len(index.ids)):
                episode = index[i]
                ids = episode.input_ids * 8
                read = episode.reads[0]
                reference = replace(
                    read.references[0], text=" ".join([read.references[0].text] * 8)
                )
                source = replace(episode.sources[0], token_end=len(ids))
                episodes.append(
                    replace(
                        episode,
                        input_ids=ids,
                        write_ends=(len(ids),),
                        sources=(source,),
                        reads=(replace(read, prefix_end=len(ids), references=(reference,)),),
                    )
                )
            write_episodes(episodes, root / "long.jsonl")
            index = EpisodeIndex(root / "long.jsonl")
        for trial in range(4):
            order = [
                ("original", evaluate_pretraining_reference),
                ("batched", evaluate_pretraining),
            ]
            if trial % 2:
                order.reverse()
            for mode, evaluate in order:
                statistics_rows = {
                    k: 0
                    for k in (
                        "read_calls",
                        "read_requests",
                        "read_padded_positions",
                        "generation_calls",
                        "generation_requests",
                        "generation_prefill_positions",
                        "generation_decode_budget",
                        "read_seconds",
                        "generation_seconds",
                    )
                }
                generated.clear()
                output = root / f"{mode}-{trial}"
                start = time.perf_counter()
                metrics = evaluate(config, tokenizer, backbone, writer, index, output, 0, 0)
                elapsed = time.perf_counter() - start
                rows = [
                    json.loads(line)
                    for line in (output / "dev-step-000000.jsonl").read_text().splitlines()
                ]
                tokens = Counter(generated)
                if expected_metrics is None:
                    expected_metrics, expected_rows, expected_tokens = metrics, rows, tokens
                else:
                    assert_results_equal(expected_metrics, metrics)
                    assert_results_equal(expected_rows, rows)
                    assert expected_tokens == tokens
                if trial:
                    runs.append(
                        {"mode": mode, "trial": trial, "seconds": elapsed, **statistics_rows}
                    )
    report = {
        "platform": platform.platform(),
        "scope": "CPU FP32 reduced Qwen2 (48 vocabulary, 16 hidden, 2 layers), 24 synthetic samples, 12 AE generations; one warm-up per mode, three alternating trials; no GPU measurement",
        "text_scale": 8 if args.long else 1,
        "runs": runs,
        "summary": [],
    }
    for mode in ("original", "batched"):
        rows = [r for r in runs if r["mode"] == mode]
        report["summary"].append(
            {
                "mode": mode,
                **{
                    k: statistics.median(r[k] for r in rows)
                    for k in rows[0]
                    if k not in ("mode", "trial")
                },
            }
        )
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
