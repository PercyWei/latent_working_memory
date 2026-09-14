from types import SimpleNamespace
import latent_working_memory.v1.pretrain.objective_comparison as series
import latent_working_memory.v1.experiment_execution as execution
import json
from dataclasses import replace

from latent_working_memory.data_preparation.pretrain.config import PreparationConfig
from latent_working_memory.data_preparation.pretrain.pipeline import prepare_sources
from latent_working_memory.v1.pretrain.prepare_objective_data import prepare
from latent_working_memory.data_preparation.pretrain.text_samples import TextSample
from latent_working_memory.v1.pretrain.data_selection import select_experiment
from latent_working_memory.v1.config import write_resolved_config


def test_objective_constructor_writes_reusable_text_sources(
    parquet_source,
    tmp_path, tokenizer, tiny_config, preparation_records
):
    model = tmp_path / "tokenizer"
    tokenizer.save_pretrained(model)
    cfg = replace(tiny_config, model_name_or_path=str(model), split_fractions=(0.3, 0.35, 0.35))
    path = tmp_path / "config.json"
    write_resolved_config(cfg, path)
    recipe = PreparationConfig(
        max_documents=64,
        min_document_chars=1,
        min_sample_tokens=4,
        max_sample_tokens=32,
        length_bounds=(32,),
        samples_per_task=(2, 2, 2),
    )
    pool = tmp_path / "pool"
    prepare_sources(parquet_source(preparation_records), cfg, pool, recipe)
    spec = {
        "source_root": str(pool),
        "seed": 20260912,
        "workers": 1,
        "train_per_source_task": 2,
        "evaluation_per_source_task": 2,
        "recipe": recipe.to_dict(),
    }
    output = tmp_path / "short-text"
    prepare(spec, path, output)
    assert {p.name for p in output.iterdir()} == {"semantic", "random"}
    documents = {s: set() for s in ("dev", "test")}
    for variant in ("semantic", "random"):
        folder = output / variant
        assert {p.name for p in folder.iterdir()} == {
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "preparation.json",
        }
        metadata = json.loads((folder / "preparation.json").read_text())
        assert (folder / metadata["source_pool"]).resolve() == pool / "source-pool.json"
        for split in ("train", "dev", "test"):
            samples = [
                TextSample(**json.loads(line))
                for line in (folder / f"{split}.jsonl").read_text().splitlines()
            ]
            assert len(samples) == 4
            if split != "train":
                ids = {s.document_id for s in samples}
                assert len(ids) == 4 and not (ids & documents[split])
                documents[split].update(ids)
    selection = {
        "sources": {v: str(output / v) for v in ("semantic", "random")},
        "runs": {"mixed": {"semantic": 0.5, "random": 0.5}},
        "seed": 20260912,
        "balance_task_lengths": False,
        "samples_per_split": {"train": 8, "dev": 4, "test": 4},
    }
    indices, _ = select_experiment(selection, cfg, tokenizer)
    assert len(indices["mixed", "train"].ids) == 8
    assert set(indices["mixed", "train"].tasks) == {"ae", "continuation"}


def test_series_uses_selection_and_parallel_source_evaluation(tmp_path, monkeypatch):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"sources": {"semantic": "unused", "random": "unused"}}))
    note = tmp_path / "note.md"
    note.write_text("test note\n")
    commands = []

    def execute(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(execution.subprocess, "run", execute)
    monkeypatch.setattr(execution.subprocess, "check_output", lambda *a, **k: "test-commit")
    monkeypatch.setattr(series, "summarize", lambda *a: "test summary")
    spec = {
        "data_selection": str(path),
        "gpus": [4, 5],
        "project": "test",
        "group": "test",
        "max_steps": 1,
        "save_every": 1,
        "evaluation": {"examples": 4, "generation_examples": 2, "prefix_tokens": [1]},
        "comparison_name": "compare",
        "note": str(note),
        "runs": [
            {"name": "run", "label": "joint", "evaluation_name": "eval", "config": "config.json"}
        ],
    }
    series.run_series(spec, tmp_path / "artifacts")
    train = next(c for c in commands if "latent_working_memory.v1.pretrain.train" in c)
    assert "--data-selection" in train and "--data-run" in train and "--data-dir" not in train
    evaluations = [c for c in commands if "latent_working_memory.v1.pretrain.evaluate" in c]
    assert {c[c.index("--evaluation-source") + 1] for c in evaluations} == {"semantic", "random"}
    assert all("--data-selection" in c and "--data-dir" not in c for c in evaluations)
