import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latent_working_memory.data_preparation.__main__ import main
from latent_working_memory.data_preparation.config import ConstructionConfig
from latent_working_memory.data_preparation.fineweb import data_contract


def test_single_config_runs_all_stages(
    tmp_path, tokenizer, tiny_config, preparation_records, preparation_recipe
):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    data = data_contract(tiny_config) | {"model_name_or_path": str(tokenizer_dir)}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"data": data, "recipe": preparation_recipe.to_dict()}))
    source = tmp_path / "fineweb" / data["pretrain_subset"]
    source.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(preparation_records), source / "fixture.parquet")
    output = tmp_path / "prepared"
    main([
        "--config", str(config_path), "--dataset-dir", str(source.parent),
        "--output-dir", str(output),
    ])
    assert (output / "comparison.json").is_file()
    for variant in ("semantic", "random"):
        metadata = json.loads((output / variant / "preparation.json").read_text())
        assert metadata["contract"] == data
        assert metadata["input_histogram"] == preparation_recipe.balanced_histogram()


def test_formal_config_contains_only_construction_parameters():
    config = ConstructionConfig.load(Path("configs/data_preparation/fineweb-4096-204k.json"))
    assert config.recipe.samples_per_task == (98000, 2100, 2100)
    assert config.recipe.max_documents == 100000
    assert config.recipe.max_sample_tokens == 4096
    assert config.data.data_seed == 20260907


@pytest.mark.parametrize("change", [
    {"learning_rate": 0.001},
    {"split_fractions": [0.9, 0.1, 0.1]},
    {"data_seed": -1},
])
def test_invalid_data_contract_is_rejected(tmp_path, change):
    config = {"data": {"model_name_or_path": "local-tokenizer", **change}, "recipe": {}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        ConstructionConfig.load(path)
