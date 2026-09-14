from dataclasses import replace
import json
import random

import pytest

from latent_working_memory.data_preparation.pretrain.pipeline import prepare_fineweb
from latent_working_memory.data_preparation.pretrain.text_samples import TextSample
from latent_working_memory.v1.pretrain.data_selection import select_experiment
from latent_working_memory.v1.pretrain.prepared_data import eligible_input_length
from latent_working_memory.v1.pretrain.sampling import EpochSampler, read_tokens
from latent_working_memory.v1.pretrain.tokenization import (
    TokenizationPool,
    inspect_block,
    text_blocks,
    tokenize_entries,
)


@pytest.fixture
def prepared(
    parquet_source,
    preparation_records,
    tokenizer,
    tiny_config,
    preparation_recipe,
    epoch_selection,
    tmp_path,
):
    root = tmp_path / "data"
    prepare_fineweb(
        parquet_source(preparation_records), tokenizer, tiny_config, root, preparation_recipe
    )
    spec = json.loads(
        epoch_selection(root, tiny_config, {v: v for v in ("semantic", "random")}).read_text()
    )
    return root, spec


def test_block_lengths_and_batch_tokens_equal_scalar_encoding(prepared, tokenizer, tiny_config):
    root, spec = prepared
    prompts = {
        t: len(tokenizer.encode(p, add_special_tokens=False))
        for t, p in (("ae", tiny_config.ae_prompt), ("continuation", tiny_config.lm_prompt))
    }
    for variant in ("semantic", "random"):
        path = root / variant / "train.jsonl"
        for block in text_blocks(path, 5):
            inspected = inspect_block(tokenizer, tiny_config, block)
            batch = tokenize_entries(
                tokenizer, tiny_config, [(path, offset, variant) for offset, _ in block]
            )
            for (offset, line), result, actual in zip(block, inspected, batch, strict=True):
                sample = TextSample(**json.loads(line))
                expected_episode = sample.to_episode(tokenizer, tiny_config, variant)
                ae, lm = read_tokens(expected_episode, tokenizer)
                assert actual.episode == expected_episode
                assert (actual.prompt_ids, actual.target_ids) == (
                    (ae or lm).prompt_ids,
                    (ae or lm).target_ids,
                )
                assert result[0] == offset
                assert result[-2] == eligible_input_length(
                    sample, tokenizer, tiny_config, False, prompts
                )
    # Even a matching tokenizer identity does not cause stored reference lengths to be reused.
    path = root / "semantic/train.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        record.update(reference_input_tokens=99999, reference_target_tokens=99999)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    indices, _ = select_experiment(spec, tiny_config, tokenizer)
    assert all(
        size <= tiny_config.max_input_tokens for size in indices["train", "train"].input_lengths
    )


@pytest.mark.parametrize("mode", ["sample", "mean"])
def test_spawn_selection_prefetch_and_resume_preserve_order(prepared, tokenizer, tiny_config, mode):
    _, spec = prepared
    config = replace(tiny_config, compression_mode=mode)
    indices, report = select_experiment(spec, config, tokenizer, tokenization_batch_size=1)
    original_random = random.getstate()
    with TokenizationPool(tokenizer, config, 2) as pool:
        parallel, parallel_report = select_experiment(
            spec, config, tokenizer, tokenization=pool, tokenization_batch_size=5
        )
        assert report == parallel_report
        assert all(index.entries == parallel[key].entries for key, index in indices.items())
        index = parallel["train", "train"]
        expected = EpochSampler(
            index,
            tokenizer,
            config,
            spec["training"],
            8,
            4,
            3,
            max_samples_per_epoch=8,
            prefetch_batches=0,
        )
        with EpochSampler(
            index,
            tokenizer,
            config,
            spec["training"],
            8,
            4,
            3,
            max_samples_per_epoch=8,
            tokenization=pool,
            prefetch_batches=2,
        ) as sampler:
            first = sampler.sample_batch()
            assert first == expected.sample_batch()
            assert sampler.pending and len(sampler.pending) <= 2
            checkpoint = sampler.state_dict()
            assert checkpoint == expected.state_dict()
            # Outstanding prefetch may complete, but must not alter committed sampler state.
            for future in sampler.pending:
                future.result()
            assert checkpoint == sampler.state_dict()
            remaining = [expected.sample_batch() for _ in range(expected.total_steps - 1)]
            sampler.load_state_dict(checkpoint)
            assert [sampler.sample_batch() for _ in remaining] == remaining
            assert sampler.state_dict() == expected.state_dict()
            with pytest.raises(StopIteration):
                sampler.sample_batch()
        assert not sampler.pending
    assert random.getstate() == original_random


def test_worker_failure_propagates_and_pool_closes(prepared, tokenizer, tiny_config):
    root, _ = prepared
    pool = TokenizationPool(tokenizer, tiny_config, 1)
    with pool:
        with pytest.raises(json.JSONDecodeError):
            list(pool.inspect_blocks([[(0, b"not-json")]]))
        future = pool.submit_batch([(root / "missing.jsonl", 0, "semantic")])
        with pytest.raises(FileNotFoundError):
            future.result()
    assert pool.executor is None


def test_parallel_merge_keeps_first_duplicate_and_checks_split_isolation(
    prepared, tokenizer, tiny_config
):
    root, spec = prepared
    for split in ("train", "dev", "test"):
        source = root / "semantic" / f"{split}.jsonl"
        target = root / "random" / f"{split}.jsonl"
        sample = json.loads(source.read_text().splitlines()[0])
        sample.update(sample_id=f"duplicate-{split}", boundary_method="random_token")
        with target.open("a") as f:
            f.write(json.dumps(sample) + "\n")
    serial, report = select_experiment(spec, tiny_config, tokenizer, tokenization_batch_size=1)
    with TokenizationPool(tokenizer, tiny_config, 2) as pool:
        parallel, other_report = select_experiment(
            spec, tiny_config, tokenizer, tokenization=pool, tokenization_batch_size=7
        )
        assert report == other_report
        assert all(i.ids == parallel[key].ids for key, i in serial.items())
        assert report["rejected"]["random/train/duplicate_content"] >= 1
        assert "duplicate-train" not in parallel["train", "train"].ids
        row = json.loads((root / "semantic/train.jsonl").read_text().splitlines()[0])
        row.update(sample_id="leak", boundary_method="random_token")
        with (root / "random/test.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="source crosses splits"):
            select_experiment(
                spec, tiny_config, tokenizer, tokenization=pool, tokenization_batch_size=7
            )
