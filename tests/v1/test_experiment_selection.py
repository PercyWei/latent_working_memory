from dataclasses import replace
import json
from pathlib import Path

import pytest

from latent_working_memory.data_preparation.experiment import prepare_experiment
from latent_working_memory.data_preparation.pipeline import prepare_fineweb
from latent_working_memory.v1.data import EpisodeIndex
from latent_working_memory.v1.sampling import PretrainSampler
from latent_working_memory.v1.training import learning_rate_at


@pytest.mark.parametrize("shared_names", [True, False])
def test_selection_equal_cells_mixture_and_curriculum(
    tmp_path, tokenizer, tiny_config, preparation_records, preparation_recipe, shared_names
):
    config = replace(
        tiny_config,
        input_length_bounds=preparation_recipe.length_bounds,
        input_length_weights=(0.5, 0.3, 0.2),
        input_length_weights_end=(1 / 3,) * 3,
        input_length_curriculum_steps=10,
        max_input_tokens=64,
        max_continuation_tokens=64,
        warmup_steps=2,
        lr_decay_steps=10,
    )
    raw = tmp_path / "raw"
    prepare_fineweb(preparation_records, tokenizer, config, raw, preparation_recipe)
    names = ("first", "second") if shared_names else ("one", "two")
    spec = {
        "sources": {"first": str(raw / "semantic"), "second": str(raw / "random")},
        "runs": {names[0]: {"first": 1}, names[1]: {"second": 1}, "both": {"first": 0.5, "second": 0.5}},
        "seed": 3,
    }
    output = tmp_path / "selected"
    report = prepare_experiment(spec, config, tokenizer, output)
    assert len({r["samples"] for r in report["runs"].values()}) == 1
    q = report["quota_per_task_length"]["train"]
    assert {p.name for p in output.iterdir()} == (
        spec["sources"].keys() | spec["runs"].keys() | {"selection.json"}
    )
    preparations = []
    for name in spec["sources"].keys() | spec["runs"].keys():
        directory = output / name
        metadata = json.loads((directory / "preparation.json").read_text())
        preparations.append(metadata["preparation_id"])
        splits = ({"train"} if name in spec["runs"] else set()) | (
            {"dev", "test"} if name in spec["sources"] else set()
        )
        assert set(metadata["counts"]) == splits
        assert {p.name for p in directory.iterdir()} == (
            {f"{split}.jsonl" for split in splits} | {"preparation.json"}
        )
        for split in splits:
            assert len(EpisodeIndex(directory / f"{split}.jsonl").offsets) == (
                metadata["counts"][split]
            )
        if name in report["runs"]:
            assert Path(report["runs"][name]["data_dir"]) == directory
        if name in report["evaluation_dirs"]:
            assert Path(report["evaluation_dirs"][name]) == directory
    assert len(set(preparations)) == len(preparations)
    assert json.loads((output / "selection.json").read_text()) == report
    index = EpisodeIndex(output / "both/train.jsonl")
    counts = {}
    for i in range(len(index.offsets)):
        e = index[i]
        key = (
            e.sources[0].provenance["boundary_variant"],
            e.reads[0].task,
            next(b for b in config.input_length_bounds if len(e.input_ids) <= b),
        )
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 12 and set(counts.values()) == {q // 2}
    sampler = PretrainSampler(index, tokenizer, config)
    assert list(sampler.length_weights(0).values()) == [0.5, 0.3, 0.2]
    assert list(sampler.length_weights(10).values()) == [1 / 3] * 3
    assert list(sampler.length_weights(5).values()) == pytest.approx(
        [(a + 1 / 3) / 2 for a in [0.5, 0.3, 0.2]]
    )
    for step in range(5):
        sampler.sample(step)
    state = sampler.state_dict()
    expected = [sampler.sample(step).episode.episode_id for step in range(12)]
    restored = PretrainSampler(index, tokenizer, config)
    restored.load_state_dict(state)
    assert expected == [restored.sample(step).episode.episode_id for step in range(12)]
    assert learning_rate_at(config, 0) == config.learning_rate / 2
    assert learning_rate_at(config, 2) == config.learning_rate
    assert learning_rate_at(config, 10) == pytest.approx(
        config.learning_rate * config.min_lr_fraction
    )

    repeated = tmp_path / "repeated"
    prepare_experiment(spec, config, tokenizer, repeated)
    for path in output.glob("*/*.jsonl"):
        assert path.read_bytes() == (repeated / path.relative_to(output)).read_bytes()


def test_source_dataset_name_cannot_describe_a_different_mixture(tmp_path, tiny_config, tokenizer):
    spec = {
        "sources": {"first": "unused/first", "second": "unused/second"},
        "runs": {"first": {"first": 0.5, "second": 0.5}},
        "seed": 3,
    }
    output = tmp_path / "selected"
    with pytest.raises(ValueError, match="named after a source"):
        prepare_experiment(spec, tiny_config, tokenizer, output)
    assert not output.exists()
