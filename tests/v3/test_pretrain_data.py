import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

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
    with pytest.raises(ValueError, match="train.jsonl:2: duplicate"):
        load_pretraining(tmp_path, tokenizer, 1, 4)


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
    with pytest.raises(ValueError, match="train.jsonl:1"):
        load_pretraining(tmp_path, tokenizer, 1, 4)


@pytest.mark.parametrize("minimum,maximum", [(0, 3), (4, 3), (True, 4), (1, 4.0)])
def test_invalid_input_interval_fails_before_reading(tmp_path, tokenizer, minimum, maximum):
    with pytest.raises(ValueError, match="input token interval"):
        load_pretraining(tmp_path, tokenizer, minimum, maximum)


def multisegment_row(
    sample_id="sample", document_id="document", cuts=(3,), text="a b c d e a b c d e", **changes
):
    row = {
        "sample_id": sample_id,
        "document_id": document_id,
        "dedup_cluster": f"{document_id}:cluster",
        "text": text,
        "write_token_ends": list(cuts),
        "source": {
            "file": "data/raw/source.parquet",
            "row_group": 0,
            "row_index": 0,
            "char_span": [100, 100 + len(text)],
        },
    }
    return row | changes


def write_multisegment(path, **splits):
    root = path / "multisegment"
    root.mkdir(exist_ok=True)
    (root / "preparation.json").write_text(
        json.dumps({"config": {"capacity": 512, "continuation_tokens": 2}}), encoding="utf-8"
    )
    write_splits(root, **splits)
    return root


def test_multisegment_ae_and_lm_share_text_without_raw_source(tmp_path, tokenizer):
    root = write_multisegment(tmp_path, train=[multisegment_row(text="b c d e a b c d")])
    splits, statistics = load_multisegment_pretraining(root, tokenizer, 3, 3, "multisegment_full")
    ae, lm = splits["train"]
    assert ae.input_ids == (4, 5, 6)
    assert ae.input_ids is ae.target_ids is lm.input_ids
    assert lm.target_ids == (7, 3)
    assert (ae.sample_id, lm.sample_id) == ("sample:ae", "sample:continuation")
    assert ae.document_id == "document"
    assert ae.dedup_cluster == "document:cluster"
    assert statistics["splits"]["train"]["read_by_task"] == {"ae": 1, "continuation": 1}
    assert statistics["splits"]["train"]["kept_source_samples"] == 1
    assert statistics["splits"]["test"]["input_tokens"]["total"] == 0
    assert sorted(item.name for item in root.iterdir()) == [
        "dev.jsonl",
        "preparation.json",
        "test.jsonl",
        "train.jsonl",
    ]
    assert not (tmp_path / "raw").exists()
    json.dumps(statistics)


def test_multisegment_first_write_needs_only_its_prefix_and_continuation(tmp_path, tokenizer):
    # A tokenizer change can shorten the full text below its saved final cut.
    root = write_multisegment(tmp_path, train=[multisegment_row(cuts=(2, 5, 8), text="a b c d")])
    splits, statistics = load_multisegment_pretraining(
        root, tokenizer, 2, 2, "multisegment_first_write"
    )
    ae, lm = splits["train"]
    assert ae.input_ids == (3, 4)
    assert lm.target_ids == (5, 6)
    assert statistics["view"] == "multisegment_first_write"
    assert statistics["splits"]["train"]["cropped_source_samples"] == 0


@pytest.mark.parametrize("view", ["multisegment_full", "multisegment_first_write"])
def test_multisegment_crop_moves_lm_target_to_actual_endpoint(tmp_path, tokenizer, view):
    cuts = (2, 6) if view == "multisegment_full" else (6, 9)
    root = write_multisegment(tmp_path, train=[multisegment_row(cuts=cuts)])
    splits, statistics = load_multisegment_pretraining(root, tokenizer, 2, 4, view)
    ae, lm = splits["train"]
    assert ae.input_ids is lm.input_ids
    assert ae.input_ids == ae.target_ids == (3, 4, 5, 6)
    assert lm.target_ids == (7, 3)
    counts = statistics["splits"]["train"]
    assert counts["cropped_source_samples"] == 1
    assert counts["cropped_source_tokens"] == 2
    assert counts["original_source_prefix_tokens"] == {"min": 6, "max": 6, "total": 6}
    assert counts["actual_source_prefix_tokens"] == {"min": 4, "max": 4, "total": 4}
    assert counts["input_tokens"] == {"min": 4, "max": 4, "total": 8}


def test_multisegment_crop_requires_only_actual_prefix_not_original_cut(tmp_path, tokenizer):
    root = write_multisegment(tmp_path, train=[multisegment_row(cuts=(8,), text="a b c d e")])
    splits, statistics = load_multisegment_pretraining(root, tokenizer, 3, 3, "multisegment_full")
    assert [row.target_ids for row in splits["train"]] == [(3, 4, 5), (6, 7)]
    assert statistics["splits"]["train"]["cropped_source_tokens"] == 5


def test_multisegment_ac_receives_each_source_once(tmp_path, tokenizer):
    root = write_multisegment(tmp_path, train=[multisegment_row()])
    splits, statistics = load_multisegment_pretraining(
        root, tokenizer, 1, 10, "multisegment_full", lm_only=True
    )
    assert len(splits["train"]) == 1
    example = splits["train"][0]
    assert example.task == "continuation"
    assert example.input_ids + example.target_ids == (3, 4, 5, 6, 7)
    assert statistics["splits"]["train"]["kept_by_task"] == {"ae": 0, "continuation": 1}


def test_multisegment_filters_unavailable_tokens_and_keeps_pairs(tmp_path, tokenizer):
    rows = [
        multisegment_row("short", cuts=(1,)),
        multisegment_row("kept", cuts=(2,)),
        multisegment_row("long", cuts=(6,)),
        multisegment_row("content", cuts=(5,), text="a b c d"),
        multisegment_row("continuation", cuts=(3,), text="a b c d"),
    ]
    root = write_multisegment(tmp_path, train=rows)
    splits, statistics = load_multisegment_pretraining(root, tokenizer, 2, 5, "multisegment_full")
    assert [row.sample_id for row in splits["train"]] == [
        "kept:ae",
        "kept:continuation",
        "long:ae",
        "long:continuation",
    ]
    counts = statistics["splits"]["train"]
    assert counts["source_samples"] == 5
    assert counts["kept_source_samples"] == 2
    assert counts["read"] == 10
    assert counts["kept_by_task"] == {"ae": 2, "continuation": 2}
    assert counts["cropped_source_samples"] == 1
    for reason in ("too_short", "content_length", "continuation_length"):
        assert counts[f"filtered_{reason}"] == 2


@pytest.mark.parametrize("leak", ("document", "cluster", "sample_id"))
def test_multisegment_source_isolation_before_filtering(tmp_path, tokenizer, leak):
    train = multisegment_row("train", cuts=(1,))
    dev = multisegment_row("dev", document_id="dev-document")
    if leak == "document":
        dev["document_id"] = train["document_id"]
    elif leak == "cluster":
        dev["dedup_cluster"] = train["dedup_cluster"]
    else:
        dev["sample_id"] = train["sample_id"]
    root = write_multisegment(tmp_path, train=[train], dev=[dev])
    with pytest.raises(ValueError, match="dev.jsonl:1:.*(source document|dedup cluster|duplicate)"):
        load_multisegment_pretraining(root, tokenizer, 2, 5, "multisegment_full")


@pytest.mark.parametrize("invalid", ["cuts", "source", "old_field", "text"])
def test_multisegment_row_contract_is_enforced_with_location(tmp_path, tokenizer, invalid):
    row = multisegment_row()
    if invalid == "cuts":
        row["write_token_ends"] = [3, 2]
    elif invalid == "source":
        row["source"]["row_group"] = -1
    elif invalid == "old_field":
        row["capacity"] = 512
    else:
        row["text"] = ""
    root = write_multisegment(tmp_path, train=[row])
    with pytest.raises(ValueError, match="train.jsonl:1"):
        load_multisegment_pretraining(root, tokenizer, 2, 5, "multisegment_full")


@pytest.mark.parametrize("continuation_tokens", [0, True, 2.5])
def test_multisegment_invalid_continuation_metadata_fails(tmp_path, tokenizer, continuation_tokens):
    root = write_multisegment(tmp_path)
    (root / "preparation.json").write_text(
        json.dumps({"config": {"capacity": 512, "continuation_tokens": continuation_tokens}})
    )
    with pytest.raises(ValueError, match="continuation_tokens"):
        load_multisegment_pretraining(root, tokenizer, 2, 5, "multisegment_full")


def test_multisegment_preserves_jsonl_order(tmp_path, tokenizer):
    root = write_multisegment(
        tmp_path,
        train=[
            multisegment_row("one"),
            multisegment_row("two", document_id="second"),
            multisegment_row("three", cuts=(4,)),
        ],
    )
    splits, _ = load_multisegment_pretraining(
        root, tokenizer, 2, 5, "multisegment_full", lm_only=True
    )
    assert [row.sample_id for row in splits["train"]] == [
        "one:continuation",
        "two:continuation",
        "three:continuation",
    ]


@pytest.mark.parametrize(
    "method", ("icae_single", "icae_multi", "autocompressors", "memory_change")
)
def test_runtime_loads_selected_multisegment_view(tmp_path, tokenizer, method):
    dynamic = method == "memory_change"
    ac = method == "autocompressors"
    view = "multisegment_first_write" if dynamic else "multisegment_full"
    cuts = (3, 5, 8) if dynamic else (3,)
    root = write_multisegment(
        tmp_path,
        train=[multisegment_row(cuts=cuts)],
        dev=[multisegment_row("dev", "dev-document", cuts)],
    )
    config = ExperimentConfig(
        ModelConfig(),
        ObjectiveConfig(method=method, stage="lm" if ac else "pretrain"),
        TrainingConfig(
            dataset_dir=str(root),
            output_dir=str(tmp_path / "output"),
            pretrain_data_view=view,
            min_input_tokens=3,
            max_input_tokens=3,
        ),
    )
    splits, statistics = load_splits(config, tokenizer)
    for split in ("train", "dev"):
        assert {row.task for row in splits[split]} == (
            {"continuation"} if ac else {"ae", "continuation"}
        )
        assert statistics["source_data"][split] == dataset_identity(splits[split])
        assert statistics["splits"][split]["selected"] == (1 if ac else 2)
        assert statistics["splits"][split]["selected_input_tokens"] == (3 if ac else 6)
