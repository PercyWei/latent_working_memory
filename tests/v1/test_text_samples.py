import json
from dataclasses import replace

import pytest
from tokenizers.pre_tokenizers import Split

from latent_working_memory.data_preparation.pretrain.fineweb import data_contract
from latent_working_memory.data_preparation.pretrain.migrate_text_samples import migrate
from latent_working_memory.data_preparation.pretrain.pipeline import prepare_fineweb
from latent_working_memory.data_preparation.pretrain.text_samples import TextSample
from latent_working_memory.v1.pretrain.prepared_data import pretraining_index
from latent_working_memory.v1.pretrain.sampling import read_tokens


def test_text_index_reuses_lengths_and_retokenizes_other_models(
    parquet_source,
    tmp_path, tiny_config, tokenizer, preparation_records, preparation_recipe, monkeypatch
):
    root = tmp_path / "data"
    prepare_fineweb(parquet_source(preparation_records), tokenizer, tiny_config, root, preparation_recipe)
    path = root / "semantic/train.jsonl"
    before = path.read_bytes()
    calls = []
    encode = tokenizer.encode

    def tracked(text, **kwargs):
        calls.append(text)
        return encode(text, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", tracked)
    first = pretraining_index(path, tokenizer, tiny_config)
    assert calls == [tiny_config.ae_prompt, tiny_config.lm_prompt]
    episode = first[0]
    ae, lm = read_tokens(episode, tokenizer)
    assert (ae or lm).target_ids[-1] == tokenizer.eos_token_id
    # Changing prompt settings does not change the dataset contract or persisted text.
    changed = replace(tiny_config, ae_prompt="Copy exactly:")
    second = pretraining_index(path, tokenizer, changed)
    ae_index = second.tasks.index("ae")
    assert second[ae_index].reads[0].prompt == "Copy exactly:"
    assert second.ids == first.ids

    tokenizer.backend_tokenizer.pre_tokenizer = Split("", "isolated")
    changed = replace(tiny_config, model_name_or_path="character-tokenizer")
    other = pretraining_index(path, tokenizer, changed)
    assert not other.reference_lengths
    assert len(other.ids) < len(first.ids)
    assert all(
        len(other[i].input_ids) == n <= changed.max_input_tokens
        for i, n in enumerate(other.input_lengths)
    )
    assert path.read_bytes() == before
    assert {p.name for p in path.parent.iterdir()} == {
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "preparation.json",
    }


@pytest.mark.parametrize("corrupt", [False, True])
def test_migration_preserves_text_order_and_rolls_back_on_invalid_input(
    parquet_source,
    tmp_path, tiny_config, tokenizer, preparation_records, preparation_recipe, corrupt
):
    root = tmp_path / "data"
    metadata = prepare_fineweb(
        parquet_source(preparation_records), tokenizer, tiny_config, root, preparation_recipe
    )
    expected = {}
    for variant in ("semantic", "random"):
        directory = root / variant
        for split in ("train", "dev", "test"):
            path = directory / f"{split}.jsonl"
            expected[variant, split] = path.read_bytes()
            rows = [
                TextSample(**json.loads(line))
                .to_episode(tokenizer, tiny_config, variant)
                .to_record()
                for line in path.read_text().splitlines()
            ]
            if corrupt and variant == "random" and split == "train":
                rows[0]["input_ids"] = (100, *rows[0]["input_ids"][1:])
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        meta = metadata[variant]
        old = {
            k: v
            for k, v in meta.items()
            if k not in {"tokenizer", "source_pool", "checks", "lengths"}
        }
        old["contract"] = data_contract(tiny_config)
        old["audit"] = {"checks": meta["checks"], "lengths": meta["lengths"]}
        (directory / "preparation.json").write_text(json.dumps(old))
        for filename in (
            "documents.jsonl",
            "sample-decisions.jsonl",
            "progress.json",
            "audit.json",
        ):
            (directory / filename).write_text("{}\n")
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.glob("*/*")}
    if corrupt:
        with pytest.raises(ValueError, match="input tokens"):
            migrate(root, tokenizer)
        assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.glob("*/*")}
    else:
        report = migrate(root, tokenizer)
        for variant in ("semantic", "random"):
            assert report[variant]["samples"] == sum(preparation_recipe.samples_per_task) * 2
            for split in ("train", "dev", "test"):
                assert (root / variant / f"{split}.jsonl").read_bytes() == expected[variant, split]
            assert len(list((root / variant).iterdir())) == 4
    assert not (root / ".text-migration").exists()
