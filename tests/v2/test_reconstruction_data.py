"""固定字符分段的独立分词、目标配对与 epoch 数据复用。"""

from contextlib import ExitStack
from dataclasses import replace
from itertools import accumulate
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig
from latent_working_memory.data_preparation.segmentation import SegmentationConfig
from latent_working_memory.data_preparation.pretrain.sources import parquet_records
from latent_working_memory.v2.pretrain.config import SelectionConfig, TrainingConfig
from latent_working_memory.v2.pretrain.data import epoch_batches, load_datasets, rejection_reason


PARTS = ("red blue   sky water", "sky water   red blue", "water sky   blue red")


def preparation_config(content_reserve_ratio=1.5):
    return DataPreparationConfig(
        source_dir="unused",
        window=SegmentationConfig(
            capacity=2,
            continuation_tokens=2,
            min_segments=2,
            max_segments=4,
            min_segment_ratio=2,
            max_segment_ratio=4,
            content_reserve_ratio=content_reserve_ratio,
        ),
    )


def reserved_parts(parts, config):
    return tuple(
        part.replace(
            " ", " " * (1 + config.window.reserved_chars((len(part) + 3) // 4) - len(part)), 1
        )
        for part in parts
    )


def sample_row(split, suffix, parts=PARTS, continuation="sky water", config=None):
    config = preparation_config() if config is None else config
    parts = reserved_parts(parts, config)
    text = "".join(parts)
    ends = list(accumulate(map(len, parts)))
    assert len(continuation) <= config.window.continuation_chars
    continuation = continuation.ljust(config.window.continuation_chars)
    return {
        "trajectory_id": f"{split}/{suffix}",
        "document_id": f"{split}/{suffix}",
        "dedup_cluster": f"{split}/{suffix}",
        "split": split,
        "text": text,
        "segments": [
            {"segment_id": f"seg{i}", "char_span": [start, end]}
            for i, (start, end) in enumerate(zip([0] + ends[:-1], ends, strict=True))
        ],
        "continuation": continuation,
        "text_char_length": len(text),
        "estimated_tokens": len(text) / 4,
        "estimated_tokens_rule": "len(text) / 4",
        "window_char_span": [11, 11 + len(text)],
        "source": {"file": "missing-raw.parquet", "row_group": 0, "row_index": 0},
    }


def write_dataset(root, config=None):
    config = preparation_config() if config is None else config
    root.joinpath("preparation.json").write_text(
        json.dumps({"preparation_id": "test", "config": config.to_dict()})
    )
    for split in ("train", "dev", "test"):
        rows = [
            sample_row(split, "good", config=config),
            sample_row(split, "empty", (" " * 20, *PARTS[1:]), config=config),
            sample_row(split, "continuation", continuation="sky", config=config),
        ]
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
    assert (
        len(tokenized_batches) == 3
    )  # One batch call per split, with independently encoded segments.
    parts = reserved_parts(PARTS, preparation_config())
    assert tokenized_batches[0][:3] == list(parts)
    assert tokenized_batches[0][3].strip() == "sky water"
    assert "".join(parts) not in tokenized_batches[0]
    joint, _, same_filtering = load_datasets(
        config, tokenizer, 128, replace(training, objective="ae_lm")
    )
    direct, _, _ = load_datasets(config, tokenizer, 128, replace(training, warmup_epochs=0))
    assert filtering == same_filtering
    expected = [
        token for part in parts for token in tokenizer.encode(part, add_special_tokens=False)
    ]
    assert expected != tokenizer.encode("".join(parts), add_special_tokens=False)
    expected += tokenizer.encode("sky water", add_special_tokens=False)
    for stage in ae:
        assert filtering[stage]["train"] == {
            "candidates": 3,
            "retained": 1,
            "rejected": {"empty_segment": 1, "continuation_length": 1},
        }
        a, b = ae[stage]["train"][0], joint[stage]["train"][0]
        assert a.sample_id == b.sample_id == "train/good"
        assert a.token_ids.tolist() == b.token_ids.tolist() == expected
        assert a.write_ends == ((12,) if stage == "warmup" else (4, 8, 12))
        assert a.source_char_start == 11 and a.capacity == 2
        assert len(a.token_ids) - a.write_ends[-1] == 2
        for epoch in (0, 1):
            assert next(epoch_batches(ae[stage]["train"], 8, 42, stage, epoch))[0] is a
    single, multi = ae["warmup"]["train"][0], ae["multiround"]["train"][0]
    assert single.sample_id == multi.sample_id
    assert single.token_ids is multi.token_ids
    assert multi.token_ids[12:14].tolist() == tokenizer.encode(
        "sky water", add_special_tokens=False
    )
    assert ae["multiround"]["dev"][0].sample_id == direct["multiround"]["dev"][0].sample_id


@pytest.mark.parametrize("reserve_ratio,segment_chars,tail_chars", [(1.5, 30, 12), (1.1, 22, 9)])
def test_fixed_text_segments_survive_tokenizer_changes_and_actual_lengths_can_exceed_estimates(
    tmp_path, tiny_base, reserve_ratio, segment_chars, tail_chars
):
    config = preparation_config(reserve_ratio)
    write_dataset(tmp_path, config)
    parts = reserved_parts(PARTS, config)
    for split in ("train", "dev", "test"):
        row = sample_row(split, "good", config=config)
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        assert [segment["char_span"] for segment in row["segments"]] == [
            [index * segment_chars, (index + 1) * segment_chars] for index in range(3)
        ]
        assert len(row["continuation"]) == tail_chars
        assert row["estimated_tokens"] == 3 * segment_chars / 4
    before = (tmp_path / "train.jsonl").read_bytes()
    vocabulary = {"[UNK]": 0, **{char: i + 1 for i, char in enumerate(sorted(set("".join(PARTS))))}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Split("", behavior="isolated")
    character_tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizers = (AutoTokenizer.from_pretrained(tiny_base), character_tokenizer)
    results = []
    for tokenizer in tokenizers:
        datasets, _, filtering = load_datasets(
            SelectionConfig(str(tmp_path)), tokenizer, 10000, TrainingConfig()
        )
        row = datasets["multiround"]["train"][0]
        lengths = []
        start = 0
        for part, end in zip(parts, row.write_ends, strict=True):
            expected = tokenizer.encode(part, add_special_tokens=False)
            assert row.token_ids[start:end].tolist() == expected
            lengths.append(len(expected))
            start = end
        assert (
            row.token_ids[start:].tolist()
            == tokenizer.encode("sky water", add_special_tokens=False)[:2]
        )
        assert filtering["multiround"]["train"]["retained"] == 1
        results.append(lengths)
    assert results[0] == [4, 4, 4]
    assert results[1] == [segment_chars] * 3  # Actual tokens need not match nominal [4, 8].
    assert (tmp_path / "train.jsonl").read_bytes() == before


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
    assert datasets["warmup"]["train"][0].write_ends == (12,)
    assert datasets["multiround"]["dev"][0].write_ends == (4, 8, 12)
    assert datasets["multiround"]["test"][0].write_ends == (4, 8, 12)


@pytest.mark.parametrize(
    "ends,tokens,positions,reason",
    [
        ((8, 14, 18), 24, 128, None),
        ((0, 4, 8), 10, 128, "empty_segment"),
        ((4, 4, 8), 10, 128, "empty_segment"),
        ((8, 14, 18), 19, 128, "continuation_length"),
        ((8, 14, 18), 24, 21, "model_window"),
    ],
)
def test_actual_token_constraints(ends, tokens, positions, reason):
    assert (
        rejection_reason(ends, tokens, preparation_config().window, positions, {"ae": 1, "lm": 1})
        == reason
    )


@pytest.mark.parametrize("cuts", [[90], [18, 36, 54, 72, 90], [18, 54, 90], [54, 90]])
def test_loader_rejects_character_plans_outside_saved_config(tmp_path, tiny_base, cuts):
    write_dataset(tmp_path)
    path = tmp_path / "train.jsonl"
    row = sample_row("train", "bad-plan")
    row["segments"] = [
        {"segment_id": f"seg{i}", "char_span": [start, end]}
        for i, (start, end) in enumerate(zip([0] + cuts[:-1], cuts, strict=True))
    ]
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        load_datasets(
            SelectionConfig(str(tmp_path)),
            AutoTokenizer.from_pretrained(tiny_base),
            128,
            TrainingConfig(),
        )


def test_loader_rejects_wrong_split(tmp_path, tiny_base):
    write_dataset(tmp_path)
    (tmp_path / "train.jsonl").write_text(json.dumps(sample_row("dev", "wrong-split")) + "\n")
    with pytest.raises(ValueError, match="trajectory split differs"):
        load_datasets(
            SelectionConfig(str(tmp_path)),
            AutoTokenizer.from_pretrained(tiny_base),
            128,
            TrainingConfig(),
        )


def test_full_character_plan_has_no_independent_total_length_cap(tmp_path, tiny_base):
    write_dataset(tmp_path)
    row = sample_row("train", "maximum-segments", ("red " * 8,) * 4)
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
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
