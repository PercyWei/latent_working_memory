from __future__ import annotations

from dataclasses import replace
import random

from latent_working_memory.data_preparation.fineweb import SemanticSpans

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from latent_working_memory.v1.backbone import LatentMemoryBackbone
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.data_preparation.config import PreparationConfig


@pytest.fixture
def tiny_config():
    return replace(
        ExperimentConfig(),
        d_mem=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        k_limit=32,
        reader_lora_rank=2,
        reader_lora_alpha=4,
        max_input_tokens=64,
        max_continuation_tokens=64,
        write_context_tokens=256,
        read_context_tokens=256,
        gradient_checkpointing=False,
        batch_size=2,
        gradient_accumulation_steps=2,
        eval_examples=4,
        eval_generation_examples=1,
        eval_generation_every=2,
        eval_every=2,
    )


@pytest.fixture
def tokenizer():
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "<pad>",
                "<s>",
                "</s>",
                "<unk>",
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                ".",
                "First",
                "Second",
                "sentence",
                "Next",
                "paragraph",
                "follows",
                "One",
                "Two",
                "Three",
                "Four",
                "five",
                "six",
                "is",
                "here",
                "ends",
                "Reconstruct",
                "Continue",
                "the",
                "text",
                "stored",
                "in",
                "memory",
                ":",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


@pytest.fixture
def components(tiny_config, tokenizer):
    torch.manual_seed(29)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    )
    backbone = LatentMemoryBackbone(base, 1, 2, 8, 2, 4, ("q_proj", "v_proj"), 0.0)
    writer = JointMemoryWriter(8, 1, 2, 16, slot_limit=32)
    return backbone, writer


@pytest.fixture
def source_records():
    # Local interface fixtures: validate boundaries and gradients, not semantic performance.
    return [
        dict(
            id=f"fixture-{i}",
            url=f"https://example.org/document/{i}",
            date="2026-09-08",
            dump="unit-fixture",
            file_path="unit-fixture",
            language="en",
            language_score=1.0,
            text=f"A sentence is here. B sentence follows. C sentence ends.\n"
            f"D paragraph is here. E sentence follows. F sentence ends.\n"
            f"First sentence follows. Second sentence ends. {i}.",
        )
        for i in range(64)
    ]


@pytest.fixture
def preparation_records(source_records):
    return [
        dict(
            row,
            text=(row["text"] + "\n" + row["text"].replace("sentence", "passage")).replace(
                "sentence", f"sentence number {i}"
            ),
        )
        for i, row in enumerate(source_records)
    ]


@pytest.fixture
def preparation_recipe():
    return PreparationConfig(
        max_documents=64,
        samples_per_task=(12, 4, 4),
        min_document_chars=1,
        min_sample_tokens=4,
        max_sample_tokens=64,
        length_bounds=(8, 32, 64),
        near_duplicate_min_words=128,
    )


@pytest.fixture
def semantic_examples(preparation_recipe):
    def generate(record, tokenizer, config, recipe=None):
        recipe = recipe or preparation_recipe
        sampler = SemanticSpans(record, tokenizer, config, recipe)
        rows = []
        for task in ("ae", "continuation"):
            rng = random.Random(f"{config.data_seed}:{task}")
            for lower, upper in recipe.length_intervals():
                for _ in range(32):
                    row = sampler.sample(task, lower, upper, rng)
                    if row is not None:
                        rows.append(row)
        return rows

    return generate
