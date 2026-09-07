from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from latent_working_memory.v1.backbone import LatentMemoryBackbone
from latent_working_memory.v1.checkpoint import (
    capture_rng_state,
    load_model_checkpoint,
    restore_rng_state,
    save_model_checkpoint,
)
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.data import Episode, Probe, write_episodes
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.v1.training import (
    P0Trainer,
    TeacherCacheIdentity,
    TeacherLogitCache,
    aggregate_p0_dev_metrics,
    evaluate_p0_example,
    load_trainable_model_state,
    run_p0_training,
    sample_p0_example,
    trainable_model_state,
)


class WordTokenizer:
    name_or_path = "test-word-tokenizer"
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [3 + sum(ord(character) for character in word) % 59 for word in text.split()]

    def decode(self, token_ids, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids if token_id != self.eos_token_id)


def _config() -> ExperimentConfig:
    return replace(
        ExperimentConfig(),
        d_mem=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        cell_tokens=4,
        k_init=16,
        k_limit=32,
        probes_per_prefix=3,
    )


def _episode() -> Episode:
    probes = (
        Probe("p0", 16, "Where is A?", "North", "recall", ()),
        Probe("p1", 16, "What changed?", "Blue", "update", ()),
        Probe("p2", 16, "How are they linked?", "West", "compose", ()),
    )
    return Episode(1, "episode-0", tuple(range(3, 19)), (), probes)


def _components() -> tuple[LatentMemoryBackbone, JointMemoryWriter, GrowthValueNetwork]:
    torch.manual_seed(29)
    base_model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    )
    backbone = LatentMemoryBackbone(
        base_model,
        bos_token_id=1,
        eos_token_id=2,
        d_mem=8,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules=("q_proj", "v_proj"),
        lora_dropout=0.0,
    )
    writer = JointMemoryWriter(8, 1, 2, 16, slot_limit=32)
    value_network = GrowthValueNetwork(8, hidden_dim=12)
    return backbone, writer, value_network


def _identity() -> TeacherCacheIdentity:
    return TeacherCacheIdentity(
        model_name_or_path="tiny-llama",
        model_revision="test-revision",
        model_dtype="torch.float32",
        tokenizer_name_or_path=WordTokenizer.name_or_path,
        tokenizer_revision="test-revision",
    )


def test_p0_sampling_uses_a_complete_prefix_and_step_of_eight_capacity() -> None:
    random.seed(7)
    example = sample_p0_example(_episode(), k_limit=32, probes_per_prefix=3)
    assert example.prefix_end == 16
    assert example.capacity == 16
    assert {probe.probe_id for probe in example.probes} == {"p0", "p1", "p2"}


def test_p0_step_trains_student_and_reuses_persisted_teacher_cache(tmp_path: Path) -> None:
    config = _config()
    tokenizer = WordTokenizer()
    backbone, writer, _ = _components()
    cache = TeacherLogitCache(_identity())
    trainer = P0Trainer(config, tokenizer, backbone, writer, cache, torch.device("cpu"))
    example = sample_p0_example(_episode(), k_limit=32, probes_per_prefix=3)
    before = writer.output_projection.weight.detach().clone()

    first = trainer.step(example)
    second = trainer.step(example)
    assert math.isfinite(first.loss)
    assert first.teacher_cache_misses == 3
    assert first.teacher_cache_hits == 0
    assert second.teacher_cache_misses == 3
    assert second.teacher_cache_hits == 3
    assert not torch.equal(before, writer.output_projection.weight.detach())

    cache_path = tmp_path / "teacher-cache.pt"
    cache.save(cache_path)
    loaded = TeacherLogitCache.load(cache_path, _identity())
    assert len(loaded) == 3
    with pytest.raises(ValueError, match="identity"):
        TeacherLogitCache.load(
            cache_path,
            replace(_identity(), model_revision="different-revision"),
        )


def test_p0_checkpoint_resume_reproduces_the_next_update(tmp_path: Path) -> None:
    config = _config()
    tokenizer = WordTokenizer()
    episode = _episode()
    example = sample_p0_example(episode, k_limit=32, probes_per_prefix=3)
    backbone_a, writer_a, value_a = _components()
    cache_a = TeacherLogitCache(_identity())
    trainer_a = P0Trainer(config, tokenizer, backbone_a, writer_a, cache_a, torch.device("cpu"))
    trainer_a.step(example)

    checkpoint_path = tmp_path / "p0.pt"
    cache_path = tmp_path / "teacher-cache.pt"
    rng_state = capture_rng_state()
    save_model_checkpoint(
        checkpoint_path,
        "p0",
        config,
        trainable_model_state(backbone_a, writer_a, value_a),
        trainer_a.optimizer.state_dict(),
        {"next_step": 1},
        rng_state,
    )
    cache_a.save(cache_path)
    expected = trainer_a.step(example)

    backbone_b, writer_b, value_b = _components()
    cache_b = TeacherLogitCache.load(cache_path, _identity())
    trainer_b = P0Trainer(config, tokenizer, backbone_b, writer_b, cache_b, torch.device("cpu"))
    checkpoint = load_model_checkpoint(checkpoint_path)
    load_trainable_model_state(checkpoint.model_state, backbone_b, writer_b, value_b)
    trainer_b.optimizer.load_state_dict(checkpoint.optimizer_state)
    restore_rng_state(checkpoint.rng_state)
    resumed = trainer_b.step(example)

    assert resumed.loss == expected.loss
    assert resumed.gold_nll == expected.gold_nll
    assert resumed.distill_kl == expected.distill_kl
    for expected_parameter, resumed_parameter in zip(
        writer_a.parameters(), writer_b.parameters(), strict=True
    ):
        assert torch.equal(expected_parameter, resumed_parameter)


def test_p0_dev_reports_teacher_memory_and_no_memory_with_eos_variants() -> None:
    config = _config()
    tokenizer = WordTokenizer()
    backbone, writer, _ = _components()
    cache = TeacherLogitCache(_identity())
    example = sample_p0_example(_episode(), k_limit=32, probes_per_prefix=3)
    records = evaluate_p0_example(
        config,
        tokenizer,
        backbone,
        writer,
        cache,
        example,
        torch.device("cpu"),
    )
    metrics = aggregate_p0_dev_metrics(records)
    assert len(records) == 3
    assert metrics["probes"] == 3
    assert metrics["target_tokens_with_eos"] == metrics["answer_tokens"] + 3
    assert set(metrics) >= {
        "teacher",
        "student_memory",
        "student_no_memory",
        "memory_nll_gain_with_eos",
        "memory_nll_gain_without_eos",
    }


def test_real_tiny_llama_p0_run_and_resume(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-llama"
    vocabulary = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "Question": 4,
        ":": 5,
        "Answer": 6,
        "North": 7,
        "Blue": 8,
        "West": 9,
        "Where": 10,
        "is": 11,
        "A": 12,
        "What": 13,
        "changed": 14,
        "How": 15,
        "are": 16,
        "they": 17,
        "linked": 18,
        "?": 19,
        "fact": 20,
    }
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            attention_dropout=0.0,
        )
    )
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    config = replace(
        _config(),
        model_name_or_path=str(model_dir),
        teacher_model_name_or_path=str(model_dir),
        reader_lora_rank=2,
        reader_lora_alpha=4,
    )
    data_dir = tmp_path / "data"
    train_episode = _episode()
    dev_episode = Episode(
        train_episode.schema_version,
        "dev-0",
        train_episode.input_ids,
        train_episode.events,
        tuple(
            Probe(
                f"dev-{probe.probe_id}",
                probe.prefix_end,
                probe.question,
                probe.answer,
                probe.kind,
                probe.evidence_event_ids,
            )
            for probe in train_episode.probes
        ),
    )
    write_episodes((train_episode,), data_dir / "train.jsonl")
    write_episodes((dev_episode,), data_dir / "dev.jsonl")
    output_dir = tmp_path / "run"

    first = run_p0_training(
        config,
        data_dir,
        output_dir,
        torch.device("cpu"),
        max_steps=1,
        episode_limit=1,
        dev_episode_limit=1,
        save_every=1,
    )
    assert first.final_checkpoint.exists()
    assert first.completed_steps == 1
    assert (output_dir / "teacher_cache.pt").exists()
    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "memory_trace.jsonl").exists()
    assert (output_dir / "resource_usage.json").exists()
    assert first.dev_metrics["probes"] == 3

    resumed = run_p0_training(
        config,
        data_dir,
        output_dir,
        torch.device("cpu"),
        max_steps=2,
        episode_limit=1,
        dev_episode_limit=1,
        save_every=1,
        resume=first.final_checkpoint,
    )
    assert resumed.completed_steps == 2
    assert resumed.final_checkpoint.name == "p0-step-000002.pt"
    resource_usage = json.loads((output_dir / "resource_usage.json").read_text())
    assert len(resource_usage["segments"]) == 2
    trace_lines = (output_dir / "memory_trace.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in trace_lines] == [0, 1]
    prediction = json.loads((output_dir / "predictions.jsonl").read_text().splitlines()[0])
    assert set(prediction) >= {
        "reference",
        "teacher_prediction",
        "student_memory_prediction",
        "student_no_memory_prediction",
        "teacher_exact_match",
        "student_memory_exact_match",
        "student_no_memory_exact_match",
    }
