"""自足文本加载、配置化切分约束、目标配对与 epoch 数据复用。"""

from contextlib import ExitStack
from dataclasses import asdict, replace
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from transformers import AutoTokenizer

from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig
from latent_working_memory.data_preparation.pretrain.sources import parquet_records
from latent_working_memory.v2.pretrain.config import SelectionConfig, TrainingConfig
from latent_working_memory.v2.pretrain.data import epoch_batches, load_datasets, rejection_reason


def preparation_config():
    return DataPreparationConfig(
        "unused",
        capacity=2,
        continuation_tokens=2,
        min_segments=2,
        max_segments=4,
        min_segment_ratio=2,
        max_segment_ratio=4,
    )


def write_dataset(root):
    root.joinpath("preparation.json").write_text(
        json.dumps({"preparation_id": "test", "config": asdict(preparation_config())})
    )
    for split in ("train", "dev", "test"):
        rows = []
        for source_index, (suffix, cuts, tokens) in enumerate(
            (
                ("good", [8, 14, 18], 24),
                ("content", [8, 14, 18], 17),
                ("continuation", [8, 14, 18], 19),
            )
        ):
            text = " ".join(("red", "blue", "sky", "water")[i % 4] for i in range(tokens))
            rows.append(
                {
                    "sample_id": f"{split}/{suffix}",
                    "document_id": f"{split}/{suffix}",
                    "dedup_cluster": f"{split}/{suffix}",
                    "text": text,
                    "write_token_ends": cuts,
                    "source": {
                        "file": "missing-raw.parquet",
                        "row_group": 0,
                        "row_index": source_index,
                        "char_span": [11, 11 + len(text)],
                    },
                }
            )
        root.joinpath(f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_tokenized_filter_is_shared_and_epochs_reuse_rows(tmp_path, tiny_base, monkeypatch):
    write_dataset(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(tiny_base)
    tokenized_batches = []
    original_call = type(tokenizer).__call__

    def counted_call(self, texts, *args, **kwargs):
        tokenized_batches.append(texts)
        return original_call(self, texts, *args, **kwargs)

    monkeypatch.setattr(type(tokenizer), "__call__", counted_call)
    config, training = SelectionConfig(str(tmp_path)), TrainingConfig()
    ae, _, filtering = load_datasets(config, tokenizer, 128, training)
    assert len(tokenized_batches) == 3  # Each split is tokenized once for both write modes.
    joint, _, same_filtering = load_datasets(
        config, tokenizer, 128, replace(training, objective="ae_lm")
    )
    direct, _, _ = load_datasets(config, tokenizer, 128, replace(training, warmup_epochs=0))
    assert filtering == same_filtering
    for stage in ae:
        assert filtering[stage]["train"] == {
            "candidates": 3,
            "retained": 1,
            "rejected": {"content_length": 1, "continuation_length": 1},
        }
        a, b = ae[stage]["train"][0], joint[stage]["train"][0]
        assert a.sample_id == b.sample_id and a.token_ids.tolist() == b.token_ids.tolist()
        assert a.write_ends == ((18,) if stage == "warmup" else (8, 14, 18))
        assert a.source_char_start == 11 and a.capacity == 2
        assert len(a.token_ids) - a.write_ends[-1] == 2
        expected = tokenizer.encode("red blue sky water " * 6, add_special_tokens=False)
        assert a.token_ids.tolist() == expected[:20]
        for epoch in (0, 1):
            assert next(epoch_batches(ae[stage]["train"], 8, 42, stage, epoch))[0] is a
    single, multi = ae["warmup"]["train"][0], ae["multiround"]["train"][0]
    assert single.sample_id == multi.sample_id
    assert single.token_ids is multi.token_ids
    assert multi.token_ids[18:20].tolist() == tokenizer.encode(
        "sky water", add_special_tokens=False
    )
    assert ae["multiround"]["dev"][0].sample_id == direct["multiround"]["dev"][0].sample_id


def test_warmup_only_keeps_multi_evaluation_and_skips_multi_training(tmp_path, tiny_base):
    write_dataset(tmp_path)
    datasets, _, filtering = load_datasets(
        SelectionConfig(str(tmp_path)),
        AutoTokenizer.from_pretrained(tiny_base),
        128,
        TrainingConfig(multiround_epochs=0),
    )
    assert datasets["multiround"]["train"] == ()
    assert filtering["multiround"]["train"] == {"candidates": 0, "retained": 0, "rejected": {}}
    assert datasets["warmup"]["train"][0].write_ends == (18,)
    assert datasets["multiround"]["dev"][0].write_ends == (8, 14, 18)
    assert datasets["multiround"]["test"][0].write_ends == (8, 14, 18)


@pytest.mark.parametrize(
    "ends,tokens,positions,reason",
    [
        ((8, 14, 18), 24, 128, None),
        ((8, 14, 18), 17, 128, "content_length"),
        ((8, 14, 18), 19, 128, "continuation_length"),
        ((8, 14, 18), 24, 21, "model_window"),
    ],
)
def test_configured_plan_and_actual_token_constraints(ends, tokens, positions, reason):
    assert (
        rejection_reason(ends, tokens, preparation_config(), positions, {"ae": 1, "lm": 1})
        == reason
    )


@pytest.mark.parametrize("cuts", [[8], [4, 8, 12, 16, 20], [3, 9, 13], [9, 14, 18]])
def test_loader_rejects_plans_outside_saved_config(tmp_path, tiny_base, cuts):
    write_dataset(tmp_path)
    path = tmp_path / "train.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row["write_token_ends"] = cuts
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        load_datasets(
            SelectionConfig(str(tmp_path)),
            AutoTokenizer.from_pretrained(tiny_base),
            128,
            TrainingConfig(),
        )


def test_full_segment_plan_has_no_independent_total_length_cap(tmp_path, tiny_base):
    write_dataset(tmp_path)
    path = tmp_path / "train.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row["write_token_ends"] = [8, 16, 24, 32]
    row["text"] = "red blue sky water " * 9
    row["source"]["char_span"][1] = row["source"]["char_span"][0] + len(row["text"])
    path.write_text(json.dumps(row) + "\n")
    datasets, _, filtering = load_datasets(
        SelectionConfig(str(tmp_path)),
        AutoTokenizer.from_pretrained(tiny_base),
        128,
        TrainingConfig(),
    )
    assert filtering["multiround"]["train"]["retained"] == 1
    assert datasets["multiround"]["train"][0].write_ends == (8, 16, 24, 32)
    assert datasets["warmup"]["train"][0].write_ends == (32,)
    assert len(datasets["warmup"]["train"][0].token_ids) == 34


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
    groups = {}
    for row, location in located:
        key = location["source_file"], location["row_group"]
        if key not in groups:
            with pq.ParquetFile(key[0]) as source:
                groups[key] = source.read_row_group(key[1]).to_pylist()
        assert groups[key][location["row_index"]] == row
