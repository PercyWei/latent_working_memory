import json
from dataclasses import replace

from latent_working_memory.data_preparation.pretrain.config import PreparationConfig
from latent_working_memory.data_preparation.pretrain.pipeline import prepare_sources
from latent_working_memory.v1.pretrain.prepare_objective_data import prepare
from latent_working_memory.data_preparation.pretrain.text_samples import TextSample
from latent_working_memory.v1.pretrain.data_selection import select_experiment
from latent_working_memory.data_preparation.pretrain.config import DataConfig


def test_objective_constructor_writes_reusable_text_sources(
    parquet_source, tmp_path, tokenizer, tiny_config, preparation_records, epoch_selection
):
    model = tmp_path / "tokenizer"
    tokenizer.save_pretrained(model)
    cfg = replace(tiny_config, model_name_or_path=str(model), split_fractions=(0.3, 0.35, 0.35))
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
        "data": DataConfig(
            model_name_or_path=str(model),
            split_fractions=cfg.split_fractions,
            data_seed=cfg.data_seed,
        ).to_dict(),
        "seed": 20260912,
        "workers": 1,
        "train_per_source_task": 2,
        "evaluation_per_source_task": 2,
        "recipe": recipe.to_dict(),
    }
    output = tmp_path / "short-text"
    prepare(spec, output)
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
    selection = json.loads(
        epoch_selection(output, cfg, {v: v for v in ("semantic", "random")}).read_text()
    )
    indices, _ = select_experiment(selection, cfg, tokenizer)
    assert len(indices["train", "train"].ids) == 8
    assert set(indices["train", "train"].tasks) == {"ae", "continuation"}
