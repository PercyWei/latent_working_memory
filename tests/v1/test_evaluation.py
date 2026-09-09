from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import json

from latent_working_memory.v1.evaluation import (
    MemoryFootprintPoint,
    NllSummary,
    aggregate_nll,
    aggregate_pretrain_metrics,
    byte_token_area,
    correct_prefix_ratio,
    evaluate_pretraining,
    exact_match,
    persistent_memory_bytes,
    normalized_token_edit_distance,
)
from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.v1.data import EpisodeIndex
from latent_working_memory.data_preparation.pipeline import prepare_fineweb
from latent_working_memory.v1.state import MemoryState


def test_exact_match_only_normalizes_whitespace() -> None:
    assert exact_match("  Alpha   Beta\n", "Alpha Beta")
    assert not exact_match("alpha beta", "Alpha Beta")


def test_nll_aggregation_is_token_weighted() -> None:
    summary = aggregate_nll((NllSummary(2.0, 1), NllSummary(2.0, 3)))
    assert summary.mean_nll == 1.0
    assert summary.perplexity == pytest.approx(2.718281828459045)


def test_generation_edit_distance_counts_insertions_deletions_and_substitutions():
    assert normalized_token_edit_distance((1, 2, 3), (1, 2, 3)) == 0
    assert normalized_token_edit_distance((1, 3), (1, 2, 3)) == pytest.approx(1 / 3)
    assert normalized_token_edit_distance((1, 2, 4, 3), (1, 2, 3)) == 0.25
    assert normalized_token_edit_distance((1, 4, 3), (1, 2, 3)) == pytest.approx(1 / 3)
    assert normalized_token_edit_distance((), (1, 2)) == 1


def test_correct_prefix_stops_at_first_error_and_uses_reference_content_length():
    assert correct_prefix_ratio((4, 5, 6), (4, 5, 6)) == 1
    assert correct_prefix_ratio((4, 8, 6), (4, 5, 6)) == pytest.approx(1 / 3)
    assert correct_prefix_ratio((4, 5), (4, 5, 6)) == pytest.approx(2 / 3)
    assert correct_prefix_ratio((), (4, 5, 6)) == 0
    assert correct_prefix_ratio((4, 5, 6, 7), (4, 5, 6)) == 1


def test_generated_corpus_bleu_and_paired_token_weighted_comparisons():
    records = []
    for i, prediction in enumerate(("a b c d", "a b c x")):
        common = {
            "episode_id": f"episode-{i}",
            "document_id": f"document-{i}",
            "granularity": "sentence",
            "boundary_method": "natural",
            "input_tokens": 4,
            "capacity": i + 1,
            "effective_ratio": 4 / (i + 1),
            "target_tokens": 4,
            "eos_nll": 100,
            "correct_tokens": 4,
        }
        records.append(
            common
            | {
                "task": "ae",
                "condition": "memory",
                "nll_sum": 4 + 8 * i,
                "reference": "a b c d",
                "prediction": prediction,
                "sequence_match": i == 0,
                "normalized_token_edit_distance": i / 4,
                "correct_prefix_ratio": 1 - i / 4,
            }
        )
        for condition, nll in (
            ("memory", 2),
            ("no_memory", 4),
            ("wrong_memory", 3),
            ("recent_context", 2.5),
            ("full_context", 1),
            ("base_full_context", 1.2),
        ):
            records.append(
                common
                | {
                    "task": "continuation",
                    "condition": condition,
                    "nll_sum": 4 * nll,
                }
            )
    metrics = aggregate_pretrain_metrics(records)
    ae = metrics["groups"]["all/ae/memory"]
    # Pooled 1/2/3/4-gram precisions; no smoothing or brevity penalty needed here.
    expected_bleu = 100 * ((7 / 8) * (5 / 6) * (3 / 4) * (1 / 2)) ** 0.25
    assert ae["bleu_4"] == pytest.approx(expected_bleu)
    assert ae["correct_prefix_ratio"] == 0.875
    assert ae["sequence_match"] == 0.5 and ae["generated_reads"] == 2
    assert ae["nll"] == 2 and ae["nll_with_eos"] == 21.6
    assert "tok:13a" in metrics["bleu"]["signature"]
    assert "eff:yes" in metrics["bleu"]["signature"]
    assert metrics["bleu"]["scale"] == "0-100"
    assert metrics["comparisons"]["all/continuation"] == pytest.approx(
        {
            "gain_vs_no_memory": 2,
            "gain_vs_wrong_memory": 1,
            "gain_vs_recent_context": 0.5,
            "nll_gap_to_full_context": 1,
            "ppl_ratio_to_full_context": 2.718281828459045,
            "nll_gap_to_base_full_context": 0.8,
            "ppl_ratio_to_base_full_context": 2.225540928492468,
        }
    )
    assert metrics["groups"]["capacity/1/ae/memory"]["bleu_4"] == pytest.approx(100)
    assert metrics["comparisons"]["length_up_to/4/continuation"]["gain_vs_recent_context"] == 0.5


def test_evaluation_controls_share_targets_budgets_and_write_test_split(
    tmp_path,
    tiny_config,
    tokenizer,
    source_records,
    components,
    monkeypatch,
):
    tiny_config = replace(tiny_config, split_fractions=(0.6, 0.2, 0.2))
    data = tmp_path / "data"
    prepare_fineweb(
        source_records,
        tokenizer,
        tiny_config,
        data,
        PreparationConfig(max_documents=len(source_records)),
    )
    index = EpisodeIndex(data / "test.jsonl")
    backbone, writer = components
    original = backbone.read_batch
    raw_calls = []

    def traced(memories, tokens, text_contexts=None, use_reader_lora=True):
        if text_contexts is not None:
            raw_calls.append((tokens[0], text_contexts[0], use_reader_lora))
            assert all(len(memory) == 0 for memory in memories)
        return original(memories, tokens, text_contexts, use_reader_lora)

    monkeypatch.setattr(backbone, "read_batch", traced)
    backbone.train()
    writer.eval()
    rng_before = torch.get_rng_state()
    metrics = evaluate_pretraining(
        tiny_config, tokenizer, backbone, writer, index, tmp_path / "eval", 2, 1234, "test"
    )
    assert backbone.training and not writer.training
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    assert metrics["split"] == "test" and metrics["training_input_tokens"] == 1234
    assert (tmp_path / "eval/test-step-000002.json").exists()
    assert not (tmp_path / "eval/dev-step-000002.jsonl").exists()
    records = [
        json.loads(line)
        for line in (tmp_path / "eval/test-step-000002.jsonl").read_text().splitlines()
    ]
    for episode_id in {r["episode_id"] for r in records}:
        selected = [
            r for r in records if r["episode_id"] == episode_id and r["task"] == "continuation"
        ]
        assert len({r["target_tokens"] for r in selected}) == 1
        for capacity in {r["capacity"] for r in selected}:
            paired = {r["condition"]: r for r in selected if r["capacity"] == capacity}
            assert len(paired) == 6
            assert (
                paired["recent_context"]["text_context_tokens"] == paired["memory"]["memory_tokens"]
            )
            for condition in ("full_context", "base_full_context"):
                assert paired[condition]["text_context_tokens"] == paired[condition]["input_tokens"]
            assert paired["base_full_context"]["reader_lora"] is False
    cursor = 0
    for i in index.evaluation_panel(tiny_config.eval_examples, tiny_config.data_seed + 1):
        episode = index[i]
        task, context, lora = raw_calls[cursor]
        assert context == episode.input_ids and lora
        assert raw_calls[cursor + 1] == (task, episode.input_ids, False)
        capacities = sorted(
            {r["capacity"] for r in records if r["episode_id"] == episode.episode_id}
        )
        for offset, capacity in enumerate(capacities, 2):
            assert raw_calls[cursor + offset] == (task, episode.input_ids[-capacity:], True)
        cursor += 2 + len(capacities)
    assert cursor == len(raw_calls)
    ae = metrics["groups"]["all/ae/memory"]
    assert 0 <= ae["bleu_4"] <= 100 and 0 <= ae["correct_prefix_ratio"] <= 1


def test_memory_bytes_and_byte_token_area() -> None:
    state = MemoryState(torch.zeros(4, 8, dtype=torch.bfloat16), seen_tokens=10)
    assert persistent_memory_bytes(state, metadata_bytes=8) == 72
    points = (
        MemoryFootprintPoint(0, 32),
        MemoryFootprintPoint(10, 64),
    )
    assert byte_token_area(points, stream_end=20) == 32 * 10 + 64 * 10
