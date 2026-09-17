"""无 tokenizer 构造、共享正文、真实长度筛选与跨实验配对。"""

from dataclasses import asdict, replace
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from transformers import AutoTokenizer

from latent_working_memory.v2.pretrain.config import SelectionConfig, TrainingConfig
from latent_working_memory.v2.pretrain.data import epoch_batches, load_datasets, rejection_reason
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
        Document(f"{split}-{i}", "abcdefgh" * (60 + i), split, "https://example.org/", str(i))
        for split in ("train", "dev", "test")
        for i in range(4)
    ]
    config = DataPreparationConfig(
        "unused",
        capacity=4,
        continuation_tokens=3,
        warmup={"train": 400, "dev": 4, "test": 4},
        multiround={"train": 40, "dev": 4, "test": 4},
    )
    multi = build_indices(documents, config, "multiround", "train")
    single = build_indices(documents, config, "warmup", "train")
    assert multi == build_indices(documents, config, "multiround", "train")
    for row in multi:
        ends = row["write_char_ends"]
        parts = [b - a for a, b in zip([0] + ends[:-1], ends)]
        assert 3 <= len(parts) <= 5 and all(16 <= n <= 48 for n in parts)
        assert ends[-1] <= 128 and row["char_end"] - row["char_start"] == ends[-1] + 12
    assert min(r["write_char_ends"][-1] for r in single) == 32
    assert max(r["write_char_ends"][-1] for r in single) == 128


def test_prepare_saves_each_source_once_without_tokenizer(tmp_path, monkeypatch):
    source = tmp_path / "source.parquet"
    records = [
        {
            "id": str(i),
            "url": f"https://example.org/{i}",
            "text": f"article {i} " + "red blue sky water " * 16,
        }
        for i in range(100)
    ]
    pq.write_table(pa.Table.from_pylist(records), source)

    def reject(*args, **kwargs):
        raise AssertionError("preparation must not load a tokenizer")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", reject)
    config = DataPreparationConfig(
        str(source),
        capacity=2,
        continuation_tokens=2,
        max_documents=100,
        split_fractions=(0.6, 0.2, 0.2),
        warmup={"train": 20, "dev": 4, "test": 4},
        multiround={"train": 20, "dev": 4, "test": 4},
    )
    first, second = tmp_path / "first", tmp_path / "second"
    metadata = prepare_dataset(config, first)
    prepare_dataset(config, second)
    documents = [json.loads(s) for s in (first / "documents.jsonl").read_text().splitlines()]
    ids = [d["document_id"] for d in documents]
    assert len(ids) == len(set(ids)) == metadata["stored_documents"]
    original = {r["id"]: r["text"] for r in records}
    assert all(d["text"] == original[d["document_id"]] for d in documents)
    document_map = {d["document_id"]: d for d in documents}
    for stage, directory in STAGE_DIRECTORIES.items():
        for split in ("train", "dev", "test"):
            name = f"{directory}/{split}.jsonl"
            assert (first / name).read_bytes() == (second / name).read_bytes()
            rows = [json.loads(s) for s in (first / name).read_text().splitlines()]
            assert len(rows) == getattr(config, stage)[split]
            assert all("text" not in r and "token_ids" not in r for r in rows)
            assert all(document_map[r["document_id"]]["split"] == split for r in rows)
    assert "tokenizer" not in metadata
    with pytest.raises(FileExistsError):
        prepare_dataset(config, first)


def test_tokenized_filter_is_shared_and_epochs_reuse_rows(tmp_path, tiny_base):
    documents = [
        asdict(Document(split, "red " * 12, split, "https://example.org/", split))
        for split in ("train", "dev", "test")
    ]
    (tmp_path / "documents.jsonl").write_text("".join(json.dumps(d) + "\n" for d in documents))
    (tmp_path / "preparation.json").write_text(
        json.dumps({"preparation_id": "test", "config": {"continuation_tokens": 2}})
    )
    for stage, directory in STAGE_DIRECTORIES.items():
        (tmp_path / directory).mkdir()
        for split in ("train", "dev", "test"):
            cuts = [16] if stage == "warmup" else [8, 16, 24]
            good = {
                "sample_id": stage + split,
                "document_id": split,
                "char_start": 0,
                "char_end": 40,
                "write_char_ends": cuts,
                "capacity": 2,
            }
            bad = dict(
                good,
                sample_id=stage + split + "bad",
                write_char_ends=[4] if stage == "warmup" else [4, 16, 24],
            )
            (tmp_path / directory / f"{split}.jsonl").write_text(
                json.dumps(good) + "\n" + json.dumps(bad) + "\n"
            )
    tokenizer = AutoTokenizer.from_pretrained(tiny_base)
    config, training = SelectionConfig(str(tmp_path)), TrainingConfig()
    ae, _, filtering = load_datasets(config, tokenizer, 128, training)
    joint, _, same_filtering = load_datasets(
        config, tokenizer, 128, replace(training, objective="ae_lm")
    )
    direct, _, _ = load_datasets(config, tokenizer, 128, replace(training, warmup_epochs=0))
    assert filtering == same_filtering
    for stage in ae:
        assert filtering[stage]["train"]["candidates"] == 2
        assert filtering[stage]["train"]["retained"] == 1
        a, b = ae[stage]["train"][0], joint[stage]["train"][0]
        assert a.sample_id == b.sample_id and a.token_ids.tolist() == b.token_ids.tolist()
        assert len(a.token_ids) - a.write_ends[-1] == 2
        for epoch in (0, 1):
            assert next(epoch_batches(ae[stage]["train"], 8, 42, stage, epoch))[0] is a
    assert ae["multiround"]["dev"][0].sample_id == direct["multiround"]["dev"][0].sample_id
    assert rejection_reason((4,), 6, 2, 2, "warmup", 4, {"ae": 1, "lm": 1}) == "model_window"
    assert (
        rejection_reason((4,), 5, 2, 2, "warmup", 128, {"ae": 1, "lm": 1}) == "continuation_length"
    )
