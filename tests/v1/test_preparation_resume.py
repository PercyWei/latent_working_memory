import json
from dataclasses import replace

import pytest

from latent_working_memory.data_preparation.pipeline import (
    VariantBuilder,
    prepare_sources,
    prepare_variant,
)


def test_resume_rolls_back_uncommitted_tail_and_matches_uninterrupted(
    tmp_path, tiny_config, tokenizer, preparation_records, preparation_recipe, monkeypatch
):
    recipe = replace(preparation_recipe, candidate_window_documents=1)
    root = tmp_path / "resume"
    prepare_sources(preparation_records, tiny_config, root, recipe)
    original = VariantBuilder.consider
    calls = 0

    def fail_second(self, candidates, handles):
        nonlocal calls
        calls += 1
        original(self, candidates, handles)
        if calls == 2:
            raise RuntimeError("interrupted window")

    monkeypatch.setattr(VariantBuilder, "consider", fail_second)
    with pytest.raises(RuntimeError, match="interrupted window"):
        prepare_variant(root, "semantic", tokenizer, tiny_config, recipe)
    progress = json.loads((root / "semantic/progress.json").read_text())
    assert progress["next_source"] == 1
    with (root / "semantic/train.jsonl").open("a") as handle:
        handle.write('{"uncommitted":')
    monkeypatch.setattr(VariantBuilder, "consider", original)
    result = prepare_variant(root, "semantic", tokenizer, tiny_config, recipe, resume=True)
    baseline = tmp_path / "baseline"
    prepare_sources(preparation_records, tiny_config, baseline, recipe)
    expected = prepare_variant(baseline, "semantic", tokenizer, tiny_config, recipe)
    assert result["statistics"] == expected["statistics"]
    for name in ("train", "dev", "test", "documents", "sample-decisions"):
        assert (root / f"semantic/{name}.jsonl").read_bytes() == (
            baseline / f"semantic/{name}.jsonl"
        ).read_bytes()


def test_resume_rejects_recipe_change_without_touching_output(
    tmp_path, tiny_config, tokenizer, preparation_records, preparation_recipe, monkeypatch
):
    root = tmp_path / "data"
    prepare_sources(preparation_records, tiny_config, root, preparation_recipe)

    def interrupt(*args):
        raise RuntimeError("stop")

    monkeypatch.setattr(VariantBuilder, "consider", interrupt)
    with pytest.raises(RuntimeError):
        prepare_variant(root, "semantic", tokenizer, tiny_config, preparation_recipe)
    output = root / "semantic/train.jsonl"
    output.write_text("uncommitted tail")
    with pytest.raises(ValueError, match="configuration"):
        prepare_variant(
            root,
            "semantic",
            tokenizer,
            tiny_config,
            replace(preparation_recipe, candidates_per_document=128),
            resume=True,
        )
    assert output.read_text() == "uncommitted tail"
