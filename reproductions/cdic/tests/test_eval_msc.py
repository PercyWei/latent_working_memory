from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cdic_repro.config import RetrievalConfig
from cdic_repro.experiments.eval_msc import (
    MscEvaluationConfig,
    _serialize_config,
    compare_runs,
    evaluate_episode,
    load_config,
    paired_episode_bootstrap,
    scoped_records,
    select_episodes,
    summarize_records,
)
from cdic_repro.experiments.msc import MscEpisode, MscTurn
from cdic_repro.model_protocol import CompressedTurn, TrainingLoss


def episode(name="example"):
    return MscEpisode(name, 5, tuple(
        MscTurn(f"{name}:s{s}:p{p}", f"q{s}{p}", f"r{s}{p}", s, p)
        for s in range(1, 6) for p in (1, 2)
    ))


class Adapter:
    def __init__(self):
        self.history = []
        self.scored = []

    def encode_query(self, query):
        return query

    def response_loss(self, retrieved_states, query, response, credit, collect_token_nll):
        assert collect_token_nll
        assert response not in self.history
        self.scored.append((query, tuple(self.history)))
        return TrainingLoss(2.0, 3, (1.0, 2.0, 3.0))

    def generate(self, retrieved_states, query):
        return f"answer to {query}"

    def compress_gold(self, retrieved_states, query, response, credit):
        del credit
        self.history.append(response)
        return CompressedTurn(response, response)


def collect(adapter, completed_ids=None):
    return list(evaluate_episode(
        adapter, episode(), completed_ids=completed_ids or set(),
        retrieval_config=RetrievalConfig(), similarity=lambda a, b: 0.9,
    ))


def test_full_history_precedes_targets_and_scopes_have_correct_denominators():
    adapter = Adapter()
    records = collect(adapter)
    assert adapter.scored[0] == ("q21", ("r11", "r12"))
    assert adapter.history[-1] == "r52"
    assert len(records) == 8
    assert len(scoped_records(records, "session_final_s2_s5")) == 4
    assert [r["id"] for r in scoped_records(records, "episode_final_s5")] == ["example:s5:p2"]
    metrics = summarize_records(records)
    assert metrics["loss_tokens"] == 24
    assert metrics["content_tokens"] == 16
    assert metrics["eos_tokens"] == 8
    assert metrics["ppl_including_eos"] == pytest.approx(math.exp(2))
    assert metrics["ppl_excluding_eos"] == pytest.approx(math.exp(1.5))
    assert metrics["mean_eos_nll"] == 3
    assert "rouge_1_f1" in metrics and "rouge_1_recall" in metrics


def test_resume_replays_gold_history_without_rescoring_completed_turns():
    original = collect(Adapter())
    adapter = Adapter()
    resumed = collect(adapter, {row["id"] for row in original[:3]})
    assert resumed == original[3:]
    assert len(adapter.history) == 10
    assert len(adapter.scored) == 5


def test_episode_selection_supports_full_first_n_and_seeded_sampling():
    data = tuple(episode(str(i)) for i in range(40))
    selected = select_episodes(data, count=8, seed=20260906)
    assert selected == select_episodes(data, count=8, seed=20260906)
    assert len({e.episode_id for e in selected}) == 8
    assert select_episodes(data, count=8, seed=None) == data[:8]
    assert select_episodes(data, count=None, seed=None) == data
    incomplete = MscEpisode("short", 5, episode().turns[:2])
    with pytest.raises(ValueError, match="complete sessions"):
        select_episodes((incomplete,), count=None, seed=None)


@pytest.mark.parametrize(
    ("name", "condition", "split", "episode_count"),
    (
        ("msc_alignment_initialization_a800.json", "initialization", "valid", 32),
        ("msc_alignment_final_a800.json", "final", "valid", 32),
        ("table1_msc_initialization_pilot_a800.json", "initialization", "test", 2),
        ("table1_msc_pilot_a800.json", "final", "test", 2),
    ),
)
def test_shipped_configs_use_the_unified_schema(name, condition, split, episode_count):
    config = load_config(Path(__file__).parents[1] / "configs" / name)
    assert (config.condition, config.split, config.episode_count) == (
        condition,
        split,
        episode_count,
    )


def test_ppl_uses_token_weighting_and_keeps_eos_out_of_content():
    rows = collect(Adapter())[:2]
    rows[0].update(token_nll=[0.0, 0.0], loss=0.0, loss_tokens=2)
    rows[1].update(token_nll=[2.0, 2.0, 2.0, 10.0], loss=4.0, loss_tokens=4)
    metrics = summarize_records(rows)
    assert metrics["ppl_including_eos"] == pytest.approx(math.exp(16 / 6))
    assert metrics["ppl_excluding_eos"] == pytest.approx(math.exp(6 / 4))
    assert metrics["mean_turn_nll"] == 2


def test_paired_bootstrap_preserves_a_constant_per_token_difference():
    rows = collect(Adapter())
    right = {row["id"]: {**row, "token_nll": [nll + 0.5 for nll in row["token_nll"]]}
             for row in rows}
    result = paired_episode_bootstrap(rows, right, resamples=100)
    assert result["percentile_95_interval"] == [0.5, 0.5]


def test_comparison_rejects_incomplete_or_mismatched_runs(tmp_path):
    rows = collect(Adapter())
    folders = [tmp_path / "init", tmp_path / "final"]
    for folder, condition in zip(folders, ("initialization", "final")):
        folder.mkdir()
        config = MscEvaluationConfig(
            model_path=Path("model"),
            checkpoint_path=Path("icae.pt"),
            training_checkpoint_path=(
                Path("final.pt") if condition == "final" else None
            ),
            data_root=Path("data"),
            artifact_dir=folder,
            condition=condition,
        )
        (folder / "protocol.json").write_text(json.dumps({
            "config": _serialize_config(config),
            "episode_ids": ["example"], "expected_scored_turns": 8,
        }))
        (folder / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = compare_runs(*folders, tmp_path / "comparison.json")
    assert result["scopes"]["all_turns_s2_s5"]["token_weighted_nll_delta"] == 0
    (folders[1] / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows[:-1]))
    with pytest.raises(ValueError, match="complete"):
        compare_runs(*folders, tmp_path / "comparison.json")
    rows[0]["reference"] = "wrong target"
    (folders[1] / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="target mismatch"):
        compare_runs(*folders, tmp_path / "comparison.json")
