from dataclasses import replace
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig
from latent_working_memory.data_preparation.pretrain.text_samples import TextSample
from latent_working_memory.data_preparation.segmentation import SegmentationConfig
from latent_working_memory.v3.config import (
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainingConfig,
)
from latent_working_memory.v3.pretrain_data import (
    load_multisegment_pretraining,
    load_pretraining,
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


def experiment(path, method="icae_single", view="text_samples", **training):
    return ExperimentConfig(
        ModelConfig(memory_slots=4),
        ObjectiveConfig(
            method=method,
            stage="lm" if method == "autocompressors" else "pretrain",
            icae_min_segments=2,
            icae_max_segments=2,
            append_slots=2,
            ac_min_segment_tokens=2,
            ac_max_segment_tokens=2,
            ae_prompt="a b",
            lm_prompt="c",
        ),
        TrainingConfig(
            **{
                "dataset_dir": str(path),
                "pretrain_data_view": view,
                "max_input_tokens": 64,
                "lm_target_tokens": 2,
            }
            | training
        ),
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
    config = experiment(tmp_path, min_input_tokens=3, max_input_tokens=3, lm_target_tokens=7)
    splits, statistics = load_pretraining(config, tokenizer, 64)
    first, second = splits["train"]
    assert first.document_id == second.document_id
    assert first.input_ids == second.input_ids == (3, 4, 5)
    assert first.target_ids is first.input_ids
    assert second.target_ids == (6, 7, 3, 4, 5, 6, 7)
    assert not set(tokenizer.all_special_ids).intersection(first.input_ids + second.target_ids)
    assert (tmp_path / "train.jsonl").read_bytes() == before
    assert isinstance(splits["train"], tuple)
    assert statistics["splits"]["train"]["kept_by_task"] == {"ae": 1, "continuation": 1}
    assert statistics["splits"]["train"]["target_tokens"] == {"min": 3, "max": 7, "total": 10}
    assert statistics["splits"]["test"]["input_tokens"] == {"min": None, "max": None, "total": 0}
    json.dumps(statistics)


def test_length_filter_checks_complete_body_and_never_crops(tmp_path, tokenizer):
    rows = [
        sample("short", text="a"),
        sample("lower", text="a b"),
        sample("upper", text="a b c d"),
        sample("long", text="a b c d e"),
    ]
    write_splits(tmp_path, train=rows)
    config = experiment(tmp_path, min_input_tokens=2, max_input_tokens=4)
    splits, statistics = load_pretraining(config, tokenizer, 64)
    assert [example.sample_id for example in splits["train"]] == ["lower", "upper"]
    assert [len(example.input_ids) for example in splits["train"]] == [2, 4]
    assert statistics["input_token_interval"] == [2, 4]
    counts = statistics["splits"]["train"]
    assert counts["read"] == 4
    assert counts["kept"] == 2
    assert counts["filtered_too_short"] == counts["filtered_too_long"] == 1
    assert counts["read_by_task"] == {"ae": 4, "continuation": 0}
    assert counts["kept_by_task"] == {"ae": 2, "continuation": 0}
    assert counts["input_tokens"] == {"min": 2, "max": 4, "total": 6}
    assert counts["target_tokens"] == {"min": 2, "max": 4, "total": 6}


@pytest.mark.parametrize("configured_limit", [None, 64])
def test_model_window_limits_body_independently_of_tokenizer_metadata(
    tmp_path, tokenizer, configured_limit
):
    write_splits(tmp_path, train=[sample("fits"), sample("long", text="a b c d")])
    config = experiment(tmp_path, max_input_tokens=configured_limit)
    splits, statistics = load_pretraining(config, tokenizer, 3)
    assert [example.sample_id for example in splits["train"]] == ["fits"]
    assert statistics["input_token_interval"] == [1, 3]
    assert tokenizer.model_max_length == 2


@pytest.mark.parametrize("model_window", [0, True, 4.0])
def test_invalid_model_window_fails_before_reading(tmp_path, tokenizer, model_window):
    with pytest.raises(ValueError, match="model_window"):
        load_pretraining(experiment(tmp_path), tokenizer, model_window)


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
        load_pretraining(experiment(tmp_path, min_input_tokens=3), tokenizer, 64)


def test_duplicate_sample_in_one_split_is_rejected(tmp_path, tokenizer):
    row = sample("same")
    write_splits(tmp_path, train=[row, row])
    with pytest.raises(ValueError, match="duplicate") as caught:
        load_pretraining(experiment(tmp_path), tokenizer, 64)
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
        load_pretraining(experiment(tmp_path), tokenizer, 64)
    assert caught.value.__notes__ == [f"{tmp_path / 'train.jsonl'}:1"]


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


def multisegment_experiment(path, **training):
    return experiment(path, view="multisegment_full_text", **training)


@pytest.mark.parametrize("view", ("text_samples", "multisegment_full_text"))
@pytest.mark.parametrize("error_type", (ValueError, KeyError, TypeError))
def test_pretraining_preserves_tokenizer_error_and_notes_location(
    tmp_path, tokenizer, monkeypatch, view, error_type
):
    if view == "text_samples":
        write_splits(tmp_path, train=[sample("sample")])
        root = tmp_path
        load = load_pretraining
    else:
        root = write_multisegment(tmp_path, train=[multisegment_row()])
        load = load_multisegment_pretraining
    error = error_type("tokenizer failed")

    def failing_encode(*args, **kwargs):
        raise error

    monkeypatch.setattr(tokenizer, "encode", failing_encode)
    with pytest.raises(error_type) as caught:
        load(experiment(root, view=view), tokenizer, 64)
    assert caught.value is error
    assert caught.value.__notes__ == [f"{root / 'train.jsonl'}:1"]


@pytest.mark.parametrize("include_construction_settings", (False, True))
def test_multisegment_loader_uses_only_saved_window_configuration(
    tmp_path, tokenizer, include_construction_settings
):
    root = write_multisegment(tmp_path, train=[multisegment_row()])
    config = multisegment_experiment(root)
    expected = load_multisegment_pretraining(config, tokenizer, 64)
    saved_config = {"window": MULTISEGMENT_CONFIG.to_dict()["window"]}
    if include_construction_settings:
        saved_config.update(
            source_dir="",
            source_batch_size=0,
            source_seed=-1,
            selection_seed=-1,
            split_counts={"train": 0},
        )
    (root / "preparation.json").write_text(json.dumps({"config": saved_config}), encoding="utf-8")
    assert load_multisegment_pretraining(config, tokenizer, 64) == expected


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
    config = multisegment_experiment(directory)
    if overlap != "none":
        with pytest.raises(ValueError, match="overlapping source windows"):
            load_multisegment_pretraining(config, tokenizer, 64)
    else:
        splits, _ = load_multisegment_pretraining(config, tokenizer, 64)
        assert len(splits["train"]) == 2
        assert len({row.document_id for row in splits["train"]}) == 1
        assert len({row.sample_id for row in splits["train"]}) == 2


@pytest.mark.parametrize("lm_ratio,task", [(0.0, "ae"), (1.0, "continuation")])
def test_multisegment_samples_one_objective_on_the_complete_body(
    tmp_path, tokenizer, lm_ratio, task
):
    row = multisegment_row(segment_texts=("b c", "d"), continuation="e a")
    root = write_multisegment(tmp_path, train=[row])
    before = (root / "train.jsonl").read_bytes()
    config = multisegment_experiment(root, lm_ratio=lm_ratio)
    splits, statistics = load_multisegment_pretraining(config, tokenizer, 64)
    assert len(splits["train"]) == 1
    example = splits["train"][0]
    assert example.input_ids == (4, 5, 6)
    assert example.task == task
    assert example.sample_id == f"sample:{task}"
    assert example.document_id == "document"
    assert example.target_ids == (example.input_ids if task == "ae" else (7, 3))
    counts = statistics["splits"]["train"]
    assert counts["read"] == counts["kept"] == 1
    assert counts["kept_by_task"][task] == 1
    assert counts["source_segments"] == {"min": 2, "max": 2, "total": 2}
    assert statistics["sampling_seed"] == 20261004
    assert statistics["lm_ratio"] == lm_ratio
    assert (root / "train.jsonl").read_bytes() == before
    json.dumps(statistics)


@pytest.mark.parametrize("target_tokens", [1, 2])
def test_training_lm_target_length_uses_only_saved_continuation(tmp_path, tokenizer, target_tokens):
    root = write_multisegment(
        tmp_path, train=[multisegment_row(segment_texts=("a b", "c d", "e a"))]
    )
    before = (root / "preparation.json").read_bytes()
    config = multisegment_experiment(root, lm_ratio=1.0, lm_target_tokens=target_tokens)
    splits, statistics = load_multisegment_pretraining(config, tokenizer, 64)
    example = splits["train"][0]
    assert example.input_ids == (3, 4, 5, 6, 7, 3)
    assert example.task == "continuation"
    assert example.target_ids == (6, 7)[:target_tokens]
    assert statistics["lm_target_tokens"] == target_tokens
    assert (root / "preparation.json").read_bytes() == before


def test_long_complete_body_is_filtered_without_partial_prefix(tmp_path, tokenizer):
    rows = [
        multisegment_row("fits", "short-document", ("a", "b")),
        multisegment_row("long", "long-document", ("a b c d", "e a")),
    ]
    root = write_multisegment(tmp_path, train=rows)
    config = multisegment_experiment(root, max_input_tokens=4, lm_ratio=0.0)
    splits, statistics = load_multisegment_pretraining(config, tokenizer, 64)
    assert [row.sample_id for row in splits["train"]] == ["fits:ae"]
    assert splits["train"][0].input_ids == (3, 4)
    counts = statistics["splits"]["train"]
    assert counts["read"] == 2
    assert counts["kept"] == counts["filtered_too_long"] == 1
    assert "cropped_source_samples" not in counts
    assert "prefix_segments" not in counts


def test_multi_filters_bodies_too_short_for_any_configured_chunk_count(tmp_path, tokenizer):
    root = write_multisegment(
        tmp_path,
        train=[
            multisegment_row("short", "short-document", ("a b", "c")),
            multisegment_row("fits", "fits-document", ("a b", "c d")),
        ],
    )
    config = multisegment_experiment(root, method="icae_multi", max_input_tokens=8, lm_ratio=0.0)
    config = replace(config, objective=replace(config.objective, icae_max_segments=4))
    splits, statistics = load_multisegment_pretraining(config, tokenizer, 64)
    assert [row.sample_id for row in splits["train"]] == ["fits:ae"]
    assert splits["train"][0].input_ids == (3, 4, 5, 6)
    assert statistics["input_token_interval"] == [4, 8]
    assert statistics["splits"]["train"]["filtered_too_short"] == 1


@pytest.mark.parametrize(
    "method", ["icae_single", "icae_multi", "autocompressors", "memory_change", "information_loss"]
)
def test_body_limit_excludes_memory_prompt_and_lm_target(tmp_path, tokenizer, method):
    root = write_multisegment(tmp_path, train=[multisegment_row()])
    config = multisegment_experiment(root, method=method, max_input_tokens=3, lm_ratio=1.0)
    config = replace(config, objective=replace(config.objective, lm_prompt="a " * 20))
    splits, statistics = load_multisegment_pretraining(config, tokenizer, 64)
    assert len(splits["train"]) == 1
    assert splits["train"][0].input_ids == (3, 4, 5)
    assert splits["train"][0].target_ids == (6, 7)
    assert statistics["input_token_interval"] == [1 if method != "icae_multi" else 2, 3]


def test_task_sampling_is_stable_and_never_changes_body(tmp_path, tokenizer):
    rows = [
        multisegment_row(f"sample-{i}", f"document-{i}", ("a b", "c d", "e a")) for i in range(32)
    ]
    root = write_multisegment(tmp_path, train=rows)

    def load(seed=123, lm_ratio=0.5):
        return load_multisegment_pretraining(
            multisegment_experiment(root, seed=seed, lm_ratio=lm_ratio), tokenizer, 64
        )

    splits, statistics = load()
    assert load() == (splits, statistics)
    originals = {row.document_id: row for row in splits["train"]}
    assert {row.task for row in originals.values()} == {"ae", "continuation"}
    assert all(row.input_ids == (3, 4, 5, 6, 7, 3) for row in originals.values())
    write_splits(root, train=list(reversed(rows)))
    reordered, _ = load()
    assert {row.document_id: row for row in reordered["train"]} == originals
    changed, _ = load(seed=456)
    assert any(originals[row.document_id].task != row.task for row in changed["train"])
    assert all(originals[row.document_id].input_ids == row.input_ids for row in changed["train"])


@pytest.mark.parametrize("lm_only", [False, True])
def test_short_saved_continuation_falls_back_only_for_ae_lm_methods(tmp_path, tokenizer, lm_only):
    row = multisegment_row(segment_texts=("a b c", "d"), continuation="e")
    root = write_multisegment(tmp_path, train=[row])
    method = "autocompressors" if lm_only else "icae_single"
    config = multisegment_experiment(root, method=method, lm_ratio=1.0, lm_target_tokens=3)
    splits, statistics = load_multisegment_pretraining(config, tokenizer, 64)
    example = splits["train"][0]
    assert example.input_ids == (3, 4, 5, 6)
    assert example.task == ("continuation" if lm_only else "ae")
    assert example.target_ids == ((7,) if lm_only else example.input_ids)
    counts = statistics["splits"]["train"]
    assert counts["short_continuation_sources"] == 1
    assert counts["lm_to_ae_sources"] == (0 if lm_only else 1)
    assert counts["read_by_task"] == {"ae": 0, "continuation": 1}
    assert counts["kept_by_task"] == (
        {"ae": 0, "continuation": 1} if lm_only else {"ae": 1, "continuation": 0}
    )


def test_default_lm_target_is_independent_of_construction_estimate(tmp_path, tokenizer):
    root = write_multisegment(tmp_path, train=[multisegment_row()])
    config = multisegment_experiment(root, lm_ratio=1.0, lm_target_tokens=512)
    splits, statistics = load_multisegment_pretraining(config, tokenizer, 64)
    assert splits["train"][0].task == "ae"
    assert statistics["lm_target_tokens"] == 512
    assert statistics["splits"]["train"]["lm_to_ae_sources"] == 1


@pytest.mark.parametrize("reserve_ratio,segment_chars", [(1.5, 6), (1.1, 5)])
def test_complete_body_is_tokenized_once_across_original_segment_boundaries(
    tmp_path, reserve_ratio, segment_chars
):
    config = replace(
        MULTISEGMENT_CONFIG,
        window=replace(MULTISEGMENT_CONFIG.window, content_reserve_ratio=reserve_ratio),
    )
    first, second = "a" * (segment_chars - 1) + "b", "c" * (segment_chars - 1) + "d"
    word_backend = Tokenizer(models.WordLevel({"<unk>": 0, first: 1, second: 2}, unk_token="<unk>"))
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
    root = write_multisegment(tmp_path, config=config, train=[row])
    before = (root / "train.jsonl").read_bytes()
    runtime_config = multisegment_experiment(root, lm_ratio=0.0)
    word_splits, _ = load_multisegment_pretraining(runtime_config, word_tokenizer, 64)
    char_splits, _ = load_multisegment_pretraining(runtime_config, char_tokenizer, 64)
    word_ae, char_ae = word_splits["train"][0], char_splits["train"][0]
    assert word_ae.input_ids == (0,)
    assert char_ae.input_ids == tuple(char_tokenizer.encode(row["text"]))
    assert len(char_ae.input_ids) == segment_chars * 2
    assert word_ae.target_ids is word_ae.input_ids
    assert char_ae.target_ids is char_ae.input_ids
    assert (root / "train.jsonl").read_bytes() == before


def test_multisegment_ac_receives_each_complete_source_once_and_ignores_lm_ratio(
    tmp_path, tokenizer
):
    root = write_multisegment(tmp_path, train=[multisegment_row()])
    config = multisegment_experiment(root, method="autocompressors", lm_ratio=0.0)
    splits, statistics = load_multisegment_pretraining(config, tokenizer, 64)
    assert len(splits["train"]) == 1
    example = splits["train"][0]
    assert example.task == "continuation"
    assert example.input_ids + example.target_ids == (3, 4, 5, 6, 7)
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
        load_multisegment_pretraining(
            multisegment_experiment(root, min_input_tokens=3), tokenizer, 64
        )
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
        load_multisegment_pretraining(multisegment_experiment(root), tokenizer, 64)
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
        load_multisegment_pretraining(multisegment_experiment(root), tokenizer, 64)


@pytest.mark.parametrize("lm_ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "method", ("icae_single", "icae_multi", "autocompressors", "memory_change", "information_loss")
)
def test_runtime_loads_full_text_view_with_one_objective(tmp_path, tokenizer, method, lm_ratio):
    parts = ("a b c", "d e")
    root = write_multisegment(
        tmp_path,
        train=[multisegment_row(segment_texts=parts)],
        dev=[multisegment_row("dev", "dev-document", parts, split="dev")],
    )
    config = multisegment_experiment(root, method=method, seed=456, lm_ratio=lm_ratio)
    splits, statistics = load_splits(config, tokenizer, 64)
    for split in ("train", "dev"):
        assert len(splits[split]) == 1
        assert splits[split][0].input_ids == (3, 4, 5, 6, 7)
        if method == "autocompressors" or lm_ratio == 1.0:
            assert splits[split][0].task == "continuation"
        elif lm_ratio == 0.0:
            assert splits[split][0].task == "ae"
        assert statistics["source_data"][split] == dataset_identity(splits[split])
        assert statistics["splits"][split]["selected"] == 1
        assert statistics["splits"][split]["selected_input_tokens"] == 5
    assert statistics["sampling_seed"] == 456
    assert statistics["lm_ratio"] == (1.0 if method == "autocompressors" else lm_ratio)
    assert statistics["lm_target_tokens"] == 2


def test_train_sample_limit_selects_whole_sources_after_length_filtering(tmp_path, tokenizer):
    root = write_multisegment(
        tmp_path,
        train=[
            multisegment_row("first", "first-document", ("a b", "c")),
            multisegment_row("second", "second-document", ("a", "b")),
            multisegment_row("long", "long-document", ("a b c d", "e")),
        ],
    )
    config = multisegment_experiment(root, max_input_tokens=3, max_train_samples=1, lm_ratio=0.0)
    one, first_stats = load_splits(config, tokenizer, 64)
    two, second_stats = load_splits(
        replace(config, training=replace(config.training, max_train_samples=2)), tokenizer, 64
    )
    assert len(one["train"]) == 1
    assert len(two["train"]) == 2
    assert one["train"][0] in two["train"]
    assert [row.input_ids for row in two["train"]] == [(3, 4, 5), (3, 4)]
    assert first_stats["source_data"] == second_stats["source_data"]
    for statistics, selected in ((first_stats, 1), (second_stats, 2)):
        assert statistics["splits"]["train"]["kept"] == 2
        assert statistics["splits"]["train"]["filtered_too_long"] == 1
        assert statistics["splits"]["train"]["selected"] == selected
