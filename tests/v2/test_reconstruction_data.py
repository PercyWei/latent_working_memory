"""原始 Parquet 定位、无正文副本、真实长度筛选与跨实验配对。"""

from contextlib import ExitStack
from dataclasses import replace
import os
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from transformers import AutoTokenizer

from latent_working_memory.v2.pretrain.config import SelectionConfig, TrainingConfig
from latent_working_memory.v2.pretrain.data import (
    epoch_batches,
    load_datasets,
    rejection_reason,
    load_referenced_documents,
)
from latent_working_memory.data_preparation.pretrain.fineweb import document_split
from latent_working_memory.data_preparation.pretrain.sources import parquet_records
from latent_working_memory.v2.pretrain.prepare_data import (
    DataPreparationConfig,
    Document,
    STAGE_DIRECTORIES,
    build_indices,
    prepare_dataset,
    segment_lengths,
)


@pytest.mark.parametrize("rounds", [3, 4, 5])
def test_estimated_segment_constraints(rounds):
    for length in range(rounds * 4, 33):
        parts = segment_lengths(length, rounds, 4, random.Random(length))
        assert sum(parts) == length and len(parts) == rounds
        assert all(4 <= p <= 12 for p in parts)


def test_stage_sampling_uses_characters_and_independent_rng():
    documents = [
        Document(f"{split}-{i}", "abcdefgh" * (60 + i), split, "source.parquet", 0, i, str(i))
        for split in ("train", "dev", "test")
        for i in range(4)
    ]
    config = DataPreparationConfig(
        "unused",
        capacity=4,
        continuation_tokens=3,
        continuation_reserve_tokens=5,
        warmup={"train": 400, "dev": 4, "test": 4},
        multiround={"train": 40, "dev": 4, "test": 4},
    )
    multi = build_indices(documents, config, "multiround", "train")
    single = build_indices(documents, config, "warmup", "train")
    assert multi == build_indices(documents, config, "multiround", "train")
    for row in multi:
        ends = row["write_token_ends"]
        parts = [b - a for a, b in zip([0] + ends[:-1], ends)]
        assert 3 <= len(parts) <= 5 and all(4 <= n <= 12 for n in parts)
        assert ends[-1] <= 32 and row["char_end"] - row["char_start"] == 6 * ends[-1] + 20
    assert min(r["write_token_ends"][-1] for r in single) == 8
    assert max(r["write_token_ends"][-1] for r in single) == 32


def test_prepare_saves_only_references_without_tokenizer(tmp_path, monkeypatch):
    source = tmp_path / "source.parquet"
    records = [
        {
            "id": str(i),
            "url": f"https://example.org/{i}",
            "text": f"article {i} " + "red blue sky water " * 16,
        }
        for i in range(100)
    ]
    pq.write_table(pa.Table.from_pylist(records), source, row_group_size=17)

    def reject(*args, **kwargs):
        raise AssertionError("preparation must not load a tokenizer")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", reject)
    config = DataPreparationConfig(
        str(source),
        capacity=2,
        continuation_tokens=2,
        continuation_reserve_tokens=3,
        max_documents=100,
        split_fractions=(0.6, 0.2, 0.2),
        warmup={"train": 20, "dev": 4, "test": 4},
        multiround={"train": 20, "dev": 4, "test": 4},
    )
    first, second = tmp_path / "first", tmp_path / "second"
    metadata = prepare_dataset(config, first)
    prepare_dataset(config, second)
    assert not (first / "documents.jsonl").exists()
    original = {r["id"]: r["text"] for r in records}
    referenced = set()
    for stage, directory in STAGE_DIRECTORIES.items():
        for split in ("train", "dev", "test"):
            name = f"{directory}/{split}.jsonl"
            assert (first / name).read_bytes() == (second / name).read_bytes()
            rows = [json.loads(s) for s in (first / name).read_text().splitlines()]
            assert len(rows) == getattr(config, stage)[split]
            assert all("text" not in r and "token_ids" not in r for r in rows)
            assert all(
                r["char_end"] - r["char_start"] == config.candidate_chars(r["write_token_ends"][-1])
                for r in rows
            )
            loaded = load_referenced_documents(first, rows)
            for r in rows:
                document = loaded[r["source_file"], r["row_group"], r["row_index"]]
                assert document["id"] == r["document_id"]
                assert document["text"] == original[r["document_id"]]
                assert (
                    document_split(r["dedup_cluster"], config.source_seed, config.split_fractions)
                    == split
                )
                referenced.add(r["document_id"])
    assert len(referenced) == metadata["referenced_documents"]
    assert "tokenizer" not in metadata
    with pytest.raises(FileExistsError):
        prepare_dataset(config, first)


def test_tokenized_filter_is_shared_and_epochs_reuse_rows(tmp_path, tiny_base, monkeypatch):
    splits = ("train", "dev", "test")
    source = tmp_path / "raw.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"id": split, "text": "water " * 12} for split in splits]),
        source,
        row_group_size=1,
    )
    reads = []
    original_read = pq.ParquetFile.read_row_group

    def counted_read(self, index, *args, **kwargs):
        reads.append(index)
        return original_read(self, index, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", counted_read)
    (tmp_path / "preparation.json").write_text(
        json.dumps(
            {
                "preparation_id": "test",
                "config": {"continuation_tokens": 2, "continuation_reserve_tokens": 3},
            }
        )
    )
    for stage, directory in STAGE_DIRECTORIES.items():
        (tmp_path / directory).mkdir()
        for split in ("train", "dev", "test"):
            cuts = [4] if stage == "warmup" else [2, 4, 6]
            good = {
                "sample_id": stage + split,
                "document_id": split,
                "source_file": "raw.parquet",
                "row_group": splits.index(split),
                "row_index": 0,
                "dedup_cluster": split,
                "char_start": 0,
                "char_end": 60,
                "write_token_ends": cuts,
                "capacity": 2,
            }
            bad = dict(
                good,
                sample_id=stage + split + "bad",
                write_token_ends=[1] if stage == "warmup" else [1, 4, 6],
            )
            (tmp_path / directory / f"{split}.jsonl").write_text(
                json.dumps(good) + "\n" + json.dumps(bad) + "\n"
            )
    tokenizer = AutoTokenizer.from_pretrained(tiny_base)
    config, training = SelectionConfig(str(tmp_path)), TrainingConfig()
    ae, _, filtering = load_datasets(config, tokenizer, 128, training)
    assert sorted(reads) == [0, 1, 2]  # All splits/stages share each row-group read.
    reads.clear()
    joint, _, same_filtering = load_datasets(
        config, tokenizer, 128, replace(training, objective="ae_lm")
    )
    assert sorted(reads) == [0, 1, 2]
    direct, _, _ = load_datasets(config, tokenizer, 128, replace(training, warmup_epochs=0))
    assert filtering == same_filtering
    for stage in ae:
        assert filtering[stage]["train"]["candidates"] == 2
        assert filtering[stage]["train"]["retained"] == 1
        a, b = ae[stage]["train"][0], joint[stage]["train"][0]
        assert a.sample_id == b.sample_id and a.token_ids.tolist() == b.token_ids.tolist()
        assert a.write_ends == ((4,) if stage == "warmup" else (2, 4, 6))
        assert len(a.token_ids) - a.write_ends[-1] == 2
        expected = tokenizer.encode("water " * 12, add_special_tokens=False)
        assert a.token_ids.tolist() == expected[: a.write_ends[-1] + 2]
        for epoch in (0, 1):
            assert next(epoch_batches(ae[stage]["train"], 8, 42, stage, epoch))[0] is a
    assert ae["multiround"]["dev"][0].sample_id == direct["multiround"]["dev"][0].sample_id
    assert rejection_reason((4,), 6, 2, 2, "warmup", 4, {"ae": 1, "lm": 1}) == "model_window"
    assert (
        rejection_reason((4,), 5, 2, 2, "warmup", 128, {"ae": 1, "lm": 1}) == "continuation_length"
    )


def test_locators_preserve_legacy_interleaved_sampling(tmp_path):
    files = []
    for f in range(2):
        path = tmp_path / f"source-{f}.parquet"
        pq.write_table(
            pa.Table.from_pylist([{"id": f"{f}-{i}", "text": str(i)} for i in range(700)]),
            path,
            row_group_size=173,
        )
        files.append(path)
    # Reference ordering used before location tracking was added.
    rng = random.Random(73)
    paths = files.copy()
    rng.shuffle(paths)
    expected = []
    with ExitStack() as stack:
        streams = []
        for path in paths:
            source = stack.enter_context(pq.ParquetFile(path))
            groups = list(range(source.num_row_groups))
            rng.shuffle(groups)
            streams.append(
                source.iter_batches(batch_size=256, row_groups=groups, use_threads=False)
            )
        while streams:
            active = []
            for stream in streams:
                batch = next(stream, None)
                if batch is None:
                    continue
                rows = batch.to_pylist()
                rng.shuffle(rows)
                expected.extend(rows)
                active.append(stream)
            streams = active
    located = list(parquet_records(files, 73))
    assert [r for r, _ in located] == expected
    indices = [
        dict(
            location,
            source_file=os.path.relpath(location["source_file"], tmp_path),
            sample_id=str(i),
            document_id=row["id"],
        )
        for i, (row, location) in enumerate(located)
    ]
    loaded = load_referenced_documents(tmp_path, indices)
    for row in indices:
        assert (
            loaded[row["source_file"], row["row_group"], row["row_index"]]["id"]
            == row["document_id"]
        )


def test_continuation_reserve_is_separate_from_target():
    config = DataPreparationConfig("unused")
    assert config.content_reserve_ratio == 1.5
    assert config.candidate_chars(1024) == 6 * 1024 + 3072
    assert config.continuation_tokens == 512
    assert config.continuation_reserve_tokens == 768
    with pytest.raises(ValueError, match="continuation_reserve_tokens"):
        replace(config, continuation_reserve_tokens=511)


def test_buffer_does_not_expand_actual_target_plan():
    assert rejection_reason((4,), 3, 2, 2, "warmup", 128, {"ae": 1, "lm": 1}) == "content_length"
    config = DataPreparationConfig("unused")
    for ratio in (0.9, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="content_reserve_ratio"):
            replace(config, content_reserve_ratio=ratio)
