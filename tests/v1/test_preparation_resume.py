import json
import threading
from dataclasses import replace

import pytest

from latent_working_memory.data_preparation.pipeline import prepare_sources, prepare_variant
from latent_working_memory.data_preparation.scoring import SampleScorer


def test_resume_rolls_back_uncommitted_tail_and_matches_uninterrupted(
    tmp_path, tiny_config, tokenizer, preparation_records, preparation_recipe, accepting_scorer
):
    recipe = replace(preparation_recipe, candidate_window_documents=1)
    root = tmp_path / "resume"
    prepare_sources(preparation_records, tiny_config, root, recipe)
    original = accepting_scorer.score_batch
    calls = 0

    def fail_second(samples):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted window")
        return original(samples)

    accepting_scorer.score_batch = fail_second
    with pytest.raises(RuntimeError, match="interrupted window"):
        prepare_variant(root, "semantic", tokenizer, tiny_config, recipe, accepting_scorer)
    progress = json.loads((root / "semantic/progress.json").read_text())
    assert progress["next_source"] == 1
    with (root / "semantic/train.jsonl").open("a") as handle:
        handle.write('{"uncommitted":')
    accepting_scorer.score_batch = original
    result = prepare_variant(
        root,
        "semantic",
        tokenizer,
        tiny_config,
        replace(recipe, scoring_batch_size=32),
        accepting_scorer,
        resume=True,
    )
    baseline = tmp_path / "baseline"
    prepare_sources(preparation_records, tiny_config, baseline, recipe)
    expected = prepare_variant(
        baseline, "semantic", tokenizer, tiny_config, recipe, accepting_scorer
    )
    assert result["statistics"] == expected["statistics"]
    for name in ("train", "dev", "test", "documents", "sample-decisions"):
        assert (root / f"semantic/{name}.jsonl").read_bytes() == (
            baseline / f"semantic/{name}.jsonl"
        ).read_bytes()


def test_resume_rejects_recipe_change_without_touching_output(
    tmp_path, tiny_config, tokenizer, preparation_records, preparation_recipe, accepting_scorer
):
    prepare_sources(preparation_records, tiny_config, tmp_path / "data", preparation_recipe)
    accepting_scorer.score_batch = lambda _: (_ for _ in ()).throw(RuntimeError("stop"))
    root = tmp_path / "data"
    with pytest.raises(RuntimeError):
        prepare_variant(
            root, "semantic", tokenizer, tiny_config, preparation_recipe, accepting_scorer
        )
    output = root / "semantic/train.jsonl"
    output.write_text("uncommitted tail")
    with pytest.raises(ValueError, match="configuration"):
        prepare_variant(
            root,
            "semantic",
            tokenizer,
            tiny_config,
            replace(preparation_recipe, candidates_per_document=128),
            accepting_scorer,
            resume=True,
        )
    assert output.read_text() == "uncommitted tail"


def test_adopt_paused_preserves_all_existing_rows(
    tmp_path, tiny_config, tokenizer, preparation_records, preparation_recipe, accepting_scorer
):
    root = tmp_path / "data"
    recipe = replace(preparation_recipe, candidate_window_documents=1)
    prepare_sources(preparation_records, tiny_config, root, recipe)
    original = accepting_scorer.score_batch
    calls = 0

    def interrupt(samples):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("pause")
        return original(samples)

    accepting_scorer.score_batch = interrupt
    with pytest.raises(RuntimeError):
        prepare_variant(root, "semantic", tokenizer, tiny_config, recipe, accepting_scorer)
    (root / "semantic/progress.json").unlink()
    before = {n: (root / f"semantic/{n}.jsonl").read_bytes() for n in ("train", "dev", "test")}
    accepting_scorer.score_batch = original
    result = prepare_variant(
        root, "semantic", tokenizer, tiny_config, recipe, accepting_scorer, adopt_paused=True
    )
    assert result["input_histogram"] == recipe.balanced_histogram()
    for name, prefix in before.items():
        assert (root / f"semantic/{name}.jsonl").read_bytes().startswith(prefix)


def test_continuous_requests_refill_before_slowest_finishes(tmp_path, preparation_recipe):
    third_started = threading.Event()
    scorer = SampleScorer(
        replace(preparation_recipe, scoring_batch_size=2), tmp_path / "cache.jsonl"
    )

    def request(payload):
        if payload["X"] == "first":
            assert third_started.wait(3), "third request waited for the entire batch"
        elif payload["X"] == "third":
            third_started.set()
        return {"result": {"decision": "keep", "reason": payload["X"]}, "usage": {}, "failures": []}

    scorer._request = request
    samples = [{"X": x} for x in ("first", "second", "third")]
    assert [x["reason"] for x in scorer.score_batch(samples)] == ["first", "second", "third"]
    assert len(scorer.cache) == 3


def test_service_failure_keeps_other_completed_reviews(tmp_path, preparation_recipe):
    scorer = SampleScorer(
        replace(preparation_recipe, scoring_batch_size=2), tmp_path / "cache.jsonl"
    )
    arrived = threading.Barrier(2)

    def request(payload):
        arrived.wait(timeout=3)
        if payload["X"] == "failure":
            raise RuntimeError("service unavailable")
        return {"result": {"decision": "keep", "reason": "valid"}, "usage": {}, "failures": []}

    scorer._request = request
    with pytest.raises(RuntimeError, match="service unavailable"):
        scorer.score_batch([{"X": "failure"}, {"X": "success"}])
    cached = list(SampleScorer(preparation_recipe, tmp_path / "cache.jsonl").cache.values())
    assert len(cached) == 1
    assert cached[0]["sample"]["X"] == "success"
