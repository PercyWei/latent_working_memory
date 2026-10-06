import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from latent_working_memory.data_preparation.pretrain.text_samples import TextSample
from latent_working_memory.v3.pretrain_data import load_pretraining


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
