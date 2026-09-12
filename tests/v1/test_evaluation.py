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
)
from latent_working_memory.v1.data import EpisodeIndex
from latent_working_memory.data_preparation.pipeline import prepare_fineweb
from latent_working_memory.v1.state import MemoryState
from latent_working_memory.v1.sampling import read_tokens


def test_exact_match_only_normalizes_whitespace() -> None:
    assert exact_match("  Alpha   Beta\n", "Alpha Beta")
    assert not exact_match("alpha beta", "Alpha Beta")


def test_nll_aggregation_is_token_weighted() -> None:
    summary = aggregate_nll((NllSummary(2.0, 1), NllSummary(2.0, 3)))
    assert summary.mean_nll == 1.0
    assert summary.perplexity == pytest.approx(2.718281828459045)


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
            "boundary_method": "natural",
            "input_tokens": 4,
            "capacity": i + 1,
            "effective_ratio": 4 / (i + 1),
            "target_tokens": 4,
            "eos_nll": 100,
        }
        records.append(
            common
            | {
                "task": "ae",
                "condition": "memory",
                "nll_sum": 4 + 8 * i,
                "reference": "a b c d",
                "prediction": prediction,
                "correct_prefix_ratio": 1 - i / 4,
            }
        )
        for condition, nll in (
            ("memory", 2),
            ("no_memory", 4),
            ("wrong_memory", 3),
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
    assert ae["generated_reads"] == 2
    assert ae["nll"] == 2 and ae["nll_with_eos"] == 21.6
    assert "tok:13a" in metrics["bleu"]["signature"]
    assert "eff:yes" in metrics["bleu"]["signature"]
    assert metrics["bleu"]["scale"] == "0-100"
    assert metrics["comparisons"]["all/continuation"] == pytest.approx(
        {
            "gain_vs_no_memory": 2,
            "gain_vs_wrong_memory": 1,
            "nll_gap_to_full_context": 1,
            "ppl_ratio_to_full_context": 2.718281828459045,
            "nll_gap_to_base_full_context": 0.8,
            "ppl_ratio_to_base_full_context": 2.225540928492468,
        }
    )
    assert metrics["groups"]["length_ratio/4/4/ae/memory"]["bleu_4"] == pytest.approx(100)
    assert metrics["comparisons"]["length_ratio/4/4/continuation"]["gain_vs_wrong_memory"] == 1


def test_evaluation_controls_share_targets_budgets_and_write_test_split(
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    components,
    monkeypatch,
):
    tiny_config = replace(tiny_config, split_fractions=(0.6, 0.2, 0.2))
    preparation_recipe = replace(
        preparation_recipe, samples_per_task=(16, 16, 16), candidates_per_document=16
    )
    data = tmp_path / "data"
    prepare_fineweb(
        preparation_records,
        tokenizer,
        tiny_config,
        data,
        preparation_recipe,
    )
    data = data / "semantic"
    index = EpisodeIndex(data / "test.jsonl")
    backbone, writer = components
    original = backbone.read_batch
    raw_calls = []
    generation_calls = []
    original_generate = backbone.greedy_students

    def traced_generate(memories, prompts, limits, use_reader_lora=True):
        generation_calls.extend(
            (len(m), p, limit, use_reader_lora)
            for m, p, limit in zip(memories, prompts, limits, strict=True)
        )
        return original_generate(memories, prompts, limits, use_reader_lora=use_reader_lora)

    monkeypatch.setattr(backbone, "greedy_students", traced_generate)

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
        selected = [r for r in records if r["episode_id"] == episode_id]
        if not selected:
            continue
        assert len({r["target_tokens"] for r in selected}) == 1
        for capacity in {r["capacity"] for r in selected}:
            paired = {r["condition"]: r for r in selected if r["capacity"] == capacity}
            assert set(paired) == (
                {"memory", "wrong_memory", "full_context", "base_full_context"}
                | ({"no_memory"} if selected[0]["task"] == "continuation" else set())
            )
            for condition in ("full_context", "base_full_context"):
                assert paired[condition]["text_context_tokens"] == paired[condition]["input_tokens"]
            assert paired["base_full_context"]["reader_lora"] is False
    cursor = 0
    panel_episodes = {}
    for i in index.evaluation_panel(
        tiny_config.eval_examples, tiny_config.data_seed + 1, tiny_config.input_length_bounds
    ):
        episode = index[i]
        panel_episodes[episode.episode_id] = episode
        task, context, lora = raw_calls[cursor]
        assert context == episode.input_ids and lora
        assert raw_calls[cursor + 1] == (task, episode.input_ids, False)
        cursor += 2
    assert cursor == len(raw_calls)
    assert all(r["condition"] != "recent_context" for r in records)
    assert all(r["condition"] != "no_memory" for r in records if r["task"] == "ae")
    removed = {"sequence_match", "normalized_token_edit_distance", "correct_tokens"}
    assert all(not removed.intersection(r) for r in records)
    assert all("token_accuracy" not in v for v in metrics["groups"].values())
    for condition in ("memory", "wrong_memory", "full_context", "base_full_context"):
        ae = metrics["groups"][f"all/ae/{condition}"]
        assert 0 <= ae["bleu_4"] <= 100 and 0 <= ae["correct_prefix_ratio"] <= 1
        assert ae["generated_reads"] == metrics["groups"]["all/ae/memory"]["generated_reads"]
    # Full-text generation is done once per document and adapter setting, shared across K.
    for episode_id in {r["episode_id"] for r in records if "prediction" in r}:
        selected = [r for r in records if r["episode_id"] == episode_id]
        episode = panel_episodes[episode_id]
        ae, _ = read_tokens(episode, tokenizer)
        for use_lora, condition in [(True, "full_context"), (False, "base_full_context")]:
            assert (
                generation_calls.count(
                    (0, (*episode.input_ids, *ae.prompt_ids), len(ae.target_ids), use_lora)
                )
                == 1
            )
            paired = [r for r in selected if r["condition"] == condition]
            assert len({r["prediction"] for r in paired}) == 1
            assert len({r["nll_sum"] for r in paired}) == 1
    assert len(generation_calls) == sum(
        r["memory_tokens"] > 0 for r in records if "prediction" in r
    ) + 2 * len({r["episode_id"] for r in records if "prediction" in r})


    diagnostic = evaluate_pretraining(
        tiny_config, tokenizer, backbone, writer, index, tmp_path / "prefix", 2, 1234,
        "test", prefix_tokens=(1,),
    )
    assert set(diagnostic["prefix_diagnostics"]) == {"memory/prefix-1", "wrong_memory/prefix-1"}
    assert diagnostic["groups"]["all/ae/memory"]["nll"] == metrics["groups"]["all/ae/memory"]["nll"]
    suffix_rows = [json.loads(line) for line in
                   (tmp_path / "prefix/test-step-000002-prefix.jsonl").read_text().splitlines()]
    for row in suffix_rows:
        episode = panel_episodes[row["episode_id"]]
        target, _ = read_tokens(episode, tokenizer)
        assert row["reference"] == tokenizer.decode(target.target_ids[1:-1], skip_special_tokens=True)
        assert row["target_tokens"] == len(target.target_ids) - 2



def test_memory_bytes_and_byte_token_area() -> None:
    state = MemoryState(torch.zeros(4, 8, dtype=torch.bfloat16), seen_tokens=10)
    assert persistent_memory_bytes(state, metadata_bytes=8) == 72
    points = (
        MemoryFootprintPoint(0, 32),
        MemoryFootprintPoint(10, 64),
    )
    assert byte_token_area(points, stream_end=20) == 32 * 10 + 64 * 10
