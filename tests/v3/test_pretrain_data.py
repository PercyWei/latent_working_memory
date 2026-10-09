from dataclasses import replace
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig
from latent_working_memory.data_preparation.segmentation import SegmentationConfig
from latent_working_memory.data_preparation.pretrain.text_samples import TextSample
from latent_working_memory.v3.config import (
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainingConfig,
)
from latent_working_memory.v3.pretrain_data import (
    load_pretraining,
    load_multisegment_pretraining,
)
from latent_working_memory.v3.runtime import dataset_identity, load_splits


@pytest.fixture
def tokenizer():
    vocabulary = {
        word: index
        for index, word in enumerate(("<unk>", "<bos>", "<eos>", "a", "b", "c", "d", "e"))
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 1), ("<eos>", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=2,
    )


def sample(identifier, text="a b c", task="ae", document="document", continuation="d e"):
    return TextSample(
        sample_id=identifier,
        document_id=document,
        source_id=f"{document}:source",
        dedup_cluster=f"{document}:cluster",
        task=task,
        text=text,
        continuation=continuation if task == "continuation" else None,
        x_char_span=[100, 100 + len(text)],
        y_char_span=[100 + len(text), 100 + len(text) + len(continuation)]
        if task == "continuation"
        else None,
        boundary_method="random_token",
        reference_input_tokens=999,
        reference_target_tokens=777 if task == "continuation" else 999,
    ).to_record()


def write_splits(path, **splits):
    for split in ("train", "dev", "test"):
        (path / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in splits.get(split, [])), encoding="utf-8"
        )


def test_ae_and_continuation_reuse_source_without_labels_or_special_tokens(tmp_path, tokenizer):
    ae = sample("ae")
    lm = sample("lm", task="continuation", continuation="d e a b c d e")
    write_splits(tmp_path, train=[ae, lm], dev=[sample("dev", document="dev-document")])
    before = (tmp_path / "train.jsonl").read_bytes()
    splits, statistics = load_pretraining(tmp_path, tokenizer, 3, 3)
    first, second = splits["train"]
    assert first.document_id == second.document_id
    assert first.input_ids == second.input_ids == (3, 4, 5)
    assert first.target_ids is first.input_ids
    assert second.target_ids == (6, 7, 3, 4, 5, 6, 7)
    assert len(second.target_ids) > 3
    assert not set(tokenizer.all_special_ids).intersection(first.input_ids + second.target_ids)
    assert (tmp_path / "train.jsonl").read_bytes() == before
    assert isinstance(splits["train"], tuple)
    assert statistics["splits"]["train"]["kept_by_task"] == {"ae": 1, "continuation": 1}
    assert statistics["splits"]["train"]["target_tokens"] == {"min": 3, "max": 7, "total": 10}
    assert statistics["splits"]["test"]["input_tokens"] == {"min": None, "max": None, "total": 0}
    json.dumps(statistics)


def test_length_selection_uses_current_tokens_and_reports_whole_sample_filtering(
    tmp_path, tokenizer
):
    rows = [
        sample("short", text="a"),
        sample("lower", text="a b"),
        sample("upper", text="a b c d"),
        sample("long", text="a b c d e"),
    ]
    write_splits(tmp_path, train=rows)
    splits, statistics = load_pretraining(tmp_path, tokenizer, 2, 4)
    assert [example.sample_id for example in splits["train"]] == ["lower", "upper"]
    assert [len(example.input_ids) for example in splits["train"]] == [2, 4]
    assert statistics["input_token_interval"] == [2, 4]
    assert statistics["splits"]["train"] == {
        "read": 4,
        "kept": 2,
        "filtered_too_short": 1,
        "filtered_too_long": 1,
        "read_by_task": {"ae": 4, "continuation": 0},
        "kept_by_task": {"ae": 2, "continuation": 0},
        "input_tokens": {"min": 2, "max": 4, "total": 6},
        "target_tokens": {"min": 2, "max": 4, "total": 6},
    }


@pytest.mark.parametrize("leak", ("document", "cluster", "sample_id"))
def test_invalid_source_is_rejected_even_if_its_input_would_be_filtered(tmp_path, tokenizer, leak):
    train = sample("train", text="a")
    dev = sample("dev", text="a b c", document="dev-document")
    if leak == "document":
        dev["document_id"] = train["document_id"]
    elif leak == "cluster":
        dev["dedup_cluster"] = train["dedup_cluster"]
    else:
        dev["sample_id"] = train["sample_id"]
    write_splits(tmp_path, train=[train], dev=[dev])
    with pytest.raises(ValueError, match="source document|dedup cluster|duplicate"):
        load_pretraining(tmp_path, tokenizer, 3, 4)


def test_duplicate_sample_in_one_split_is_rejected(tmp_path, tokenizer):
    row = sample("same")
    write_splits(tmp_path, train=[row, row])
    with pytest.raises(ValueError, match="duplicate") as caught:
        load_pretraining(tmp_path, tokenizer, 1, 4)
    assert caught.value.__notes__ == [f"{tmp_path / 'train.jsonl'}:2"]


@pytest.mark.parametrize("invalid", ("span", "continuation", "field", "task"))
def test_existing_textsample_contract_is_enforced(tmp_path, tokenizer, invalid):
    row = sample("sample", task="continuation")
    if invalid == "span":
        row["x_char_span"][1] += 1
    elif invalid == "continuation":
        row["y_char_span"][0] += 1
    elif invalid == "field":
        row["input_ids"] = [1, 2, 3]
    else:
        row["task"] = "qa"
    write_splits(tmp_path, train=[row])
    error_type = TypeError if invalid == "field" else ValueError
    with pytest.raises(error_type) as caught:
        load_pretraining(tmp_path, tokenizer, 1, 4)
    assert caught.value.__notes__ == [f"{tmp_path / 'train.jsonl'}:1"]


@pytest.mark.parametrize("minimum,maximum", [(0, 3), (4, 3), (True, 4), (1, 4.0)])
def test_invalid_input_interval_fails_before_reading(tmp_path, tokenizer, minimum, maximum):
    with pytest.raises(ValueError, match="input token interval"):
        load_pretraining(tmp_path, tokenizer, minimum, maximum)


MULTISEGMENT_CONFIG = DataPreparationConfig(
    source_dir="unused",
    window=SegmentationConfig(
        capacity=1,
        min_segment_ratio=1,
        max_segment_ratio=4,
        min_segments=2,
        max_segments=3,
        continuation_tokens=2,
        content_reserve_ratio=1,
    ),
)


def multisegment_row(
    trajectory_id="sample",
    document_id="document",
    segment_texts=("a b", "c"),
    continuation="d e",
    split="train",
    config=MULTISEGMENT_CONFIG,
    nominal_parts=None,
    **changes,
):
    if nominal_parts is None:
        nominal_parts = [(len(part) + 3) // 4 for part in segment_texts]
    parts = [
        part.ljust(config.window.reserved_chars(nominal))
        for part, nominal in zip(segment_texts, nominal_parts, strict=True)
    ]
    text = "".join(parts)
    segments, start = [], 0
    for index, part in enumerate(parts):
        segments.append({"segment_id": f"seg{index}", "char_span": [start, start + len(part)]})
        start += len(part)
    tail_length = config.window.continuation_chars
    assert len(continuation) <= tail_length
    row = {
        "trajectory_id": trajectory_id,
        "document_id": document_id,
        "dedup_cluster": f"{document_id}:cluster",
        "split": split,
        "text": text,
        "segments": segments,
        "continuation": continuation.ljust(tail_length),
        "window_char_span": [100, 100 + len(text)],
        "source": {"file": f"data/raw/{document_id}.parquet", "row_group": 0, "row_index": 0},
        "text_char_length": len(text),
        "estimated_tokens": len(text) / 4,
        "estimated_tokens_rule": "len(text) / 4",
    }
    return row | changes


def write_multisegment(path, config=MULTISEGMENT_CONFIG, **splits):
    root = path / "multisegment"
    root.mkdir(exist_ok=True)
    (root / "preparation.json").write_text(
        json.dumps({"config": config.to_dict()}), encoding="utf-8"
    )
    write_splits(root, **splits)
    return root


@pytest.mark.parametrize("view", ("text_samples", "multisegment_random_prefix"))
@pytest.mark.parametrize("error_type", (ValueError, KeyError, TypeError))
def test_pretraining_preserves_tokenizer_error_and_notes_location(
    tmp_path, tokenizer, monkeypatch, view, error_type
):
    if view == "text_samples":
        write_splits(tmp_path, train=[sample("sample")])
        root = tmp_path
    else:
        root = write_multisegment(tmp_path, train=[multisegment_row()])
    error = error_type("tokenizer failed")

    def failing_encode(*args, **kwargs):
        raise error

    monkeypatch.setattr(tokenizer, "encode", failing_encode)
    with pytest.raises(error_type) as caught:
        if view == "text_samples":
            load_pretraining(root, tokenizer, 1, 4)
        else:
            load_multisegment_pretraining(root, tokenizer, 1, 4, view)
    assert caught.value is error
    assert caught.value.__notes__ == [f"{root / 'train.jsonl'}:1"]


@pytest.mark.parametrize("include_construction_settings", (False, True))
def test_multisegment_loader_uses_only_saved_window_configuration(
    tmp_path, tokenizer, include_construction_settings
):
    root = write_multisegment(tmp_path, train=[multisegment_row()])
    expected = load_multisegment_pretraining(root, tokenizer, 1, 4, "multisegment_random_prefix")
    config = {"window": MULTISEGMENT_CONFIG.to_dict()["window"]}
    if include_construction_settings:
        config.update(
            source_dir="",
            source_batch_size=0,
            source_seed=-1,
            selection_seed=-1,
            split_counts={"train": 0},
        )
    (root / "preparation.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    assert (
        load_multisegment_pretraining(root, tokenizer, 1, 4, "multisegment_random_prefix")
        == expected
    )


@pytest.mark.parametrize("overlap", ("none", "body", "continuation"))
def test_multisegment_same_document_windows_include_continuation_in_nonoverlap_check(
    tmp_path, tokenizer, overlap
):
    first = multisegment_row(trajectory_id="first")
    body_end = first["window_char_span"][1]
    full_end = body_end + len(first["continuation"])
    start = {"none": full_end, "body": body_end - 1, "continuation": full_end - 1}[overlap]
    second = multisegment_row(
        trajectory_id="second", window_char_span=[start, start + len(first["text"])]
    )
    directory = write_multisegment(tmp_path, train=[second, first])
    if overlap != "none":
        with pytest.raises(ValueError, match="overlapping source windows"):
            load_multisegment_pretraining(
                directory, tokenizer, 1, 8192, "multisegment_random_prefix"
            )
    else:
        splits, _ = load_multisegment_pretraining(
            directory, tokenizer, 1, 8192, "multisegment_random_prefix"
        )
        assert len(splits["train"]) == 2
        assert len({sample.document_id for sample in splits["train"]}) == 1
        assert len({sample.sample_id for sample in splits["train"]}) == 2


@pytest.mark.parametrize("lm_ratio,task", [(0.0, "ae"), (1.0, "continuation")])
def test_multisegment_samples_one_objective_per_source(tmp_path, tokenizer, lm_ratio, task):
    row = multisegment_row(segment_texts=("b c", "d"), continuation="e a")
    root = write_multisegment(tmp_path, train=[row])
    before = (root / "train.jsonl").read_bytes()
    splits, statistics = load_multisegment_pretraining(
        root, tokenizer, 1, 3, "multisegment_random_prefix", lm_ratio=lm_ratio, lm_target_tokens=2
    )
    assert len(splits["train"]) == 1
    example = splits["train"][0]
    assert example.input_ids in ((4, 5), (4, 5, 6))
    assert example.task == task
    assert example.sample_id == f"sample:{task}"
    assert example.document_id == "document"
    if task == "ae":
        assert example.target_ids is example.input_ids
    else:
        stream = (4, 5, 6, 7, 3)
        cut = len(example.input_ids)
        assert example.target_ids == stream[cut : cut + 2]
    counts = statistics["splits"]["train"]
    assert counts["read"] == counts["kept"] == counts["kept_source_samples"] == 1
    assert counts["kept_by_task"][task] == 1
    assert statistics["sampling_seed"] == 20261004
    assert statistics["lm_ratio"] == lm_ratio
    assert (root / "train.jsonl").read_bytes() == before
    assert not (tmp_path / "raw").exists()
    json.dumps(statistics)


@pytest.mark.parametrize("target_tokens", [1, 3, 6])
def test_training_lm_target_length_is_independent_of_construction_estimate(
    tmp_path, tokenizer, target_tokens
):
    row = multisegment_row(segment_texts=("a b", "c d", "e a"))
    root = write_multisegment(tmp_path, train=[row])
    metadata_before = (root / "preparation.json").read_bytes()
    splits, statistics = load_multisegment_pretraining(
        root,
        tokenizer,
        1,
        2,
        "multisegment_random_prefix",
        lm_ratio=1.0,
        lm_target_tokens=target_tokens,
    )
    example = splits["train"][0]
    assert example.input_ids == (3, 4)
    assert example.task == "continuation"
    assert example.target_ids == (5, 6, 7, 3, 6, 7)[:target_tokens]
    assert statistics["lm_target_tokens"] == target_tokens
    assert MULTISEGMENT_CONFIG.window.continuation_tokens == 2
    assert (root / "preparation.json").read_bytes() == metadata_before


def test_default_lm_target_does_not_use_construction_estimate(tmp_path, tokenizer):
    root = write_multisegment(tmp_path, train=[multisegment_row()])
    splits, statistics = load_multisegment_pretraining(
        root, tokenizer, 1, 2, "multisegment_random_prefix", lm_ratio=1.0
    )
    assert splits["train"][0].task == "ae"
    assert statistics["lm_target_tokens"] == 512
    assert statistics["splits"]["train"]["lm_to_ae_sources"] == 1


def test_random_prefix_preserves_segment_boundaries_and_upper_limit(tmp_path, tokenizer):
    rows = [
        multisegment_row(f"sample-{i}", f"document-{i}", ("a b", "c d", "e a")) for i in range(64)
    ]
    root = write_multisegment(tmp_path, train=rows)
    splits, statistics = load_multisegment_pretraining(
        root,
        tokenizer,
        1,
        4,
        "multisegment_random_prefix",
        seed=123,
        lm_ratio=0.5,
        lm_target_tokens=2,
    )
    assert len(splits["train"]) == len(rows)
    assert {len(row.input_ids) for row in splits["train"]} == {2, 4}
    assert {row.task for row in splits["train"]} == {"ae", "continuation"}
    assert all(row.input_ids in ((3, 4), (3, 4, 5, 6)) for row in splits["train"])
    for row in splits["train"]:
        if row.task == "continuation":
            stream = (3, 4, 5, 6, 7, 3, 6, 7)
            cut = len(row.input_ids)
            assert row.target_ids == stream[cut : cut + 2]
    counts = statistics["splits"]["train"]
    assert counts["available_prefix_segments"] == {"min": 2, "max": 2, "total": 128}
    assert counts["prefix_segments"]["min"] == 1
    assert counts["prefix_segments"]["max"] == 2
    assert counts["cropped_source_samples"] == 0


def test_sampling_is_stable_by_source_across_order_and_task_ratio(tmp_path, tokenizer):
    rows = [
        multisegment_row(f"sample-{i}", f"document-{i}", ("a b", "c d", "e a")) for i in range(32)
    ]
    root = write_multisegment(tmp_path, train=rows)

    def load(seed=123, lm_ratio=0.5):
        return load_multisegment_pretraining(
            root,
            tokenizer,
            1,
            6,
            "multisegment_random_prefix",
            seed=seed,
            lm_ratio=lm_ratio,
            lm_target_tokens=2,
        )

    splits, statistics = load()
    assert load() == (splits, statistics)
    originals = {row.document_id: row for row in splits["train"]}
    write_splits(root, train=list(reversed(rows)))
    reordered, _ = load()
    assert {row.document_id: row for row in reordered["train"]} == originals
    changed, _ = load(seed=456)
    assert any(originals[row.document_id] != row for row in changed["train"])
    all_ae, _ = load(lm_ratio=0.0)
    all_lm, _ = load(lm_ratio=1.0)
    assert [row.input_ids for row in all_ae["train"]] == [row.input_ids for row in all_lm["train"]]
    assert {row.task for row in all_ae["train"]} == {"ae"}
    assert {row.task for row in all_lm["train"]} == {"continuation"}


def test_short_sources_are_retained_and_oversized_first_segment_is_cropped(tmp_path, tokenizer):
    rows = [
        multisegment_row("short", "short-document", ("a", "b")),
        multisegment_row("long", "long-document", ("a b c d", "e a")),
    ]
    root = write_multisegment(tmp_path, train=rows)
    splits, statistics = load_multisegment_pretraining(
        root, tokenizer, 1, 3, "multisegment_random_prefix", lm_ratio=1.0, lm_target_tokens=2
    )
    short, long = splits["train"]
    assert len(short.input_ids) in (1, 2)
    assert long.input_ids == (3, 4, 5)
    assert long.target_ids == (6, 7)
    counts = statistics["splits"]["train"]
    assert counts["kept_source_samples"] == 2
    assert counts["filtered_too_short"] == 0
    assert counts["cropped_source_samples"] == counts["cropped_source_tokens"] == 1


@pytest.mark.parametrize("lm_only", [False, True])
def test_short_continuation_keeps_source_and_only_ae_lm_sampling_falls_back(
    tmp_path, tokenizer, lm_only
):
    config = replace(
        MULTISEGMENT_CONFIG,
        window=replace(MULTISEGMENT_CONFIG.window, continuation_tokens=3),
    )
    row = multisegment_row(segment_texts=("a b c", "d"), continuation="e", config=config)
    root = write_multisegment(tmp_path, config=config, train=[row])
    splits, statistics = load_multisegment_pretraining(
        root,
        tokenizer,
        1,
        3,
        "multisegment_random_prefix",
        lm_only=lm_only,
        lm_ratio=1.0,
        lm_target_tokens=3,
    )
    assert len(splits["train"]) == 1
    example = splits["train"][0]
    assert example.input_ids == (3, 4, 5)
    assert example.task == ("continuation" if lm_only else "ae")
    assert example.target_ids == ((6, 7) if lm_only else example.input_ids)
    counts = statistics["splits"]["train"]
    assert counts["short_continuation_sources"] == 1
    assert counts["lm_to_ae_sources"] == (0 if lm_only else 1)
    assert counts["read_by_task"] == {"ae": 0, "continuation": 1}
    assert counts["kept_by_task"] == (
        {"ae": 0, "continuation": 1} if lm_only else {"ae": 1, "continuation": 0}
    )


@pytest.mark.parametrize("reserve_ratio,segment_chars,tail_chars", [(1.5, 6, 12), (1.1, 5, 9)])
def test_multisegment_fixed_character_segments_survive_tokenizer_change(
    tmp_path, reserve_ratio, segment_chars, tail_chars
):
    config = replace(
        MULTISEGMENT_CONFIG,
        window=replace(MULTISEGMENT_CONFIG.window, content_reserve_ratio=reserve_ratio),
    )
    first, second = "a" * (segment_chars - 1) + "b", "c" * (segment_chars - 1) + "d"
    word_backend = Tokenizer(
        models.WordLevel({"<unk>": 0, first: 1, second: 2, "e": 3, "f": 4}, unk_token="<unk>")
    )
    word_backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    word_tokenizer = PreTrainedTokenizerFast(tokenizer_object=word_backend, unk_token="<unk>")
    char_backend = Tokenizer(
        models.BPE(
            {"<unk>": 0, **{char: i for i, char in enumerate("abcdef ", 1)}}, [], unk_token="<unk>"
        )
    )
    char_tokenizer = PreTrainedTokenizerFast(tokenizer_object=char_backend, unk_token="<unk>")
    row = multisegment_row(
        segment_texts=(first, second), continuation="e f", config=config, nominal_parts=(1, 1)
    )
    assert [segment["char_span"] for segment in row["segments"]] == [
        [0, segment_chars],
        [segment_chars, segment_chars * 2],
    ]
    assert len(row["continuation"]) == tail_chars
    root = write_multisegment(tmp_path, config=config, train=[row])
    before = (root / "train.jsonl").read_bytes()
    word_splits, word_stats = load_multisegment_pretraining(
        root, word_tokenizer, 1, 20, "multisegment_random_prefix", lm_ratio=0.0
    )
    char_splits, char_stats = load_multisegment_pretraining(
        root, char_tokenizer, 1, 20, "multisegment_random_prefix", lm_ratio=0.0
    )
    word_ae, char_ae = word_splits["train"][0], char_splits["train"][0]
    selected = word_stats["splits"]["train"]["prefix_segments"]["total"]
    assert char_stats["splits"]["train"]["prefix_segments"]["total"] == selected
    assert word_ae.input_ids == (1, 2)[:selected]
    all_chars = (1,) * (segment_chars - 1) + (2,) + (3,) * (segment_chars - 1) + (4,)
    assert char_ae.input_ids == all_chars[: selected * segment_chars]
    assert word_ae.target_ids is word_ae.input_ids
    assert char_ae.target_ids is char_ae.input_ids
    assert word_tokenizer.encode(row["text"], add_special_tokens=False) == [0]
    assert (root / "train.jsonl").read_bytes() == before


def test_multisegment_ac_receives_each_source_once_and_ignores_lm_ratio(tmp_path, tokenizer):
    root = write_multisegment(tmp_path, train=[multisegment_row()])
    splits, statistics = load_multisegment_pretraining(
        root,
        tokenizer,
        1,
        10,
        "multisegment_random_prefix",
        lm_only=True,
        lm_ratio=0.0,
        lm_target_tokens=2,
    )
    assert len(splits["train"]) == 1
    example = splits["train"][0]
    assert example.task == "continuation"
    assert example.input_ids + example.target_ids in ((3, 4, 5, 6), (3, 4, 5, 6, 7))
    assert statistics["lm_ratio"] == 1.0
    assert statistics["splits"]["train"]["kept_by_task"] == {"ae": 0, "continuation": 1}


@pytest.mark.parametrize("leak", ("document", "cluster", "trajectory_id"))
def test_multisegment_source_isolation_before_filtering(tmp_path, tokenizer, leak):
    train = multisegment_row("train", segment_texts=("a", "b"))
    dev = multisegment_row("dev", document_id="dev-document", split="dev")
    if leak == "document":
        dev["document_id"] = train["document_id"]
    elif leak == "cluster":
        dev["dedup_cluster"] = train["dedup_cluster"]
    else:
        dev["trajectory_id"] = train["trajectory_id"]
    root = write_multisegment(tmp_path, train=[train], dev=[dev])
    with pytest.raises(ValueError, match="source document|dedup cluster|duplicate") as caught:
        load_multisegment_pretraining(root, tokenizer, 3, 5, "multisegment_random_prefix")
    assert caught.value.__notes__ == [f"{root / 'dev.jsonl'}:1"]


@pytest.mark.parametrize(
    "invalid", ["segments", "source", "old_field", "text", "split", "segment_length", "tail_length"]
)
def test_multisegment_row_contract_is_enforced_with_location(tmp_path, tokenizer, invalid):
    row = multisegment_row()
    if invalid == "segments":
        row["segments"][1]["char_span"][0] += 1
    elif invalid == "source":
        row["source"]["row_group"] = -1
    elif invalid == "old_field":
        row["write_token_ends"] = [2, 3]
    elif invalid == "text":
        row["text"] = ""
    elif invalid == "split":
        row["split"] = "dev"
    elif invalid == "segment_length":
        row = multisegment_row(segment_texts=("a b c d e a b c d", "e"))
    else:
        row["continuation"] = row["continuation"][:-1]
    root = write_multisegment(tmp_path, train=[row])
    error_type = TypeError if invalid == "old_field" else ValueError
    with pytest.raises(error_type) as caught:
        load_multisegment_pretraining(root, tokenizer, 2, 5, "multisegment_random_prefix")
    assert caught.value.__notes__ == [f"{root / 'train.jsonl'}:1"]


@pytest.mark.parametrize("continuation_tokens", [0, True, 2.5])
def test_multisegment_invalid_continuation_metadata_fails(tmp_path, tokenizer, continuation_tokens):
    root = write_multisegment(tmp_path)
    metadata = {
        "config": MULTISEGMENT_CONFIG.to_dict()
        | {
            "window": MULTISEGMENT_CONFIG.to_dict()["window"]
            | {"continuation_tokens": continuation_tokens}
        }
    }
    (root / "preparation.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="continuation_tokens"):
        load_multisegment_pretraining(root, tokenizer, 2, 5, "multisegment_random_prefix")


@pytest.mark.parametrize("lm_ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "method", ("icae_single", "icae_multi", "autocompressors", "memory_change", "information_loss")
)
def test_runtime_loads_random_prefix_view_with_one_objective(tmp_path, tokenizer, method, lm_ratio):
    ac = method == "autocompressors"
    parts = ("a b c", "d e")
    root = write_multisegment(
        tmp_path,
        train=[multisegment_row(segment_texts=parts)],
        dev=[multisegment_row("dev", "dev-document", parts, split="dev")],
    )
    config = ExperimentConfig(
        ModelConfig(),
        ObjectiveConfig(method=method, stage="lm" if ac else "pretrain"),
        TrainingConfig(
            dataset_dir=str(root),
            output_dir=str(tmp_path / "output"),
            pretrain_data_view="multisegment_random_prefix",
            min_input_tokens=1,
            max_input_tokens=3,
            seed=456,
            lm_ratio=lm_ratio,
            lm_target_tokens=2,
        ),
    )
    splits, statistics = load_splits(config, tokenizer)
    for split in ("train", "dev"):
        assert len(splits[split]) == 1
        assert splits[split][0].input_ids == (3, 4, 5)
        if ac or lm_ratio == 1.0:
            assert splits[split][0].task == "continuation"
        elif lm_ratio == 0.0:
            assert splits[split][0].task == "ae"
        assert statistics["source_data"][split] == dataset_identity(splits[split])
        assert statistics["splits"][split]["selected"] == 1
        assert statistics["splits"][split]["selected_input_tokens"] == 3
    assert statistics["sampling_seed"] == 456
    assert statistics["lm_ratio"] == (1.0 if ac else lm_ratio)
    assert statistics["lm_target_tokens"] == 2
