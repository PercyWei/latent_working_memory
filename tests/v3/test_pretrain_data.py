import json

import pyarrow as pa
import pyarrow.parquet as pq
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
    load_reconstruction_pretraining,
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


def reconstruction_row(sample_id="index", document_id="document", cuts=(3,), **changes):
    row = {
        "sample_id": sample_id,
        "document_id": document_id,
        "dedup_cluster": f"{document_id}:cluster",
        "source_file": "../raw/source.parquet",
        "row_group": 0,
        "row_index": 0,
        "char_start": 0,
        "char_end": len("a b c d e a b c d e"),
        "write_token_ends": list(cuts),
        "capacity": 512,
    }
    return row | changes


def write_reconstruction(path, directory="single", documents=None, **splits):
    root = path / "reconstruction"
    root.mkdir(exist_ok=True)
    (root / "preparation.json").write_text(
        json.dumps({"config": {"continuation_tokens": 2}}), encoding="utf-8"
    )
    index_root = root / directory
    index_root.mkdir(exist_ok=True)
    write_splits(index_root, **splits)
    raw = path / "raw"
    raw.mkdir(exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(documents or [{"id": "document", "text": "a b c d e a b c d e"}]),
        raw / "source.parquet",
        row_group_size=1,
    )
    return root


def test_reconstruction_ae_and_lm_use_contiguous_current_tokens(tmp_path, tokenizer):
    # The stored capacity is unrelated to this test's three-token prefix.
    row = reconstruction_row(char_start=2, char_end=17)
    root = write_reconstruction(tmp_path, train=[row])
    splits, statistics = load_reconstruction_pretraining(
        root, tokenizer, 3, 3, "reconstruction_single"
    )
    ae, lm = splits["train"]
    assert ae.input_ids == (4, 5, 6)
    assert ae.input_ids is ae.target_ids is lm.input_ids
    assert lm.target_ids == (7, 3)
    assert (ae.sample_id, lm.sample_id) == ("index:ae", "index:continuation")
    assert ae.document_id == "document"
    assert ae.dedup_cluster == "document:cluster"
    assert statistics["splits"]["train"]["read_by_task"] == {"ae": 1, "continuation": 1}
    assert statistics["splits"]["train"]["kept_source_indices"] == 1
    assert statistics["splits"]["test"]["input_tokens"]["total"] == 0
    assert sorted(item.name for item in root.iterdir()) == ["preparation.json", "single"]
    json.dumps(statistics)


def test_reconstruction_first_write_ignores_final_cut_and_uses_next_tokens(tmp_path, tokenizer):
    root = write_reconstruction(
        tmp_path, directory="multi", train=[reconstruction_row(cuts=(2, 5, 8))]
    )
    splits, statistics = load_reconstruction_pretraining(
        root, tokenizer, 2, 2, "reconstruction_first_write"
    )
    ae, lm = splits["train"]
    assert ae.input_ids == (3, 4)
    assert lm.target_ids == (5, 6)
    assert statistics["index_directory"] == "multi"


def test_reconstruction_ac_receives_each_source_once(tmp_path, tokenizer):
    root = write_reconstruction(tmp_path, train=[reconstruction_row()])
    splits, statistics = load_reconstruction_pretraining(
        root, tokenizer, 1, 10, "reconstruction_single", lm_only=True
    )
    assert len(splits["train"]) == 1
    example = splits["train"][0]
    assert example.task == "continuation"
    assert example.input_ids + example.target_ids == (3, 4, 5, 6, 7)
    assert statistics["splits"]["train"]["kept_by_task"] == {"ae": 0, "continuation": 1}


def test_reconstruction_filters_unavailable_tokens_and_length_selection(tmp_path, tokenizer):
    rows = [
        reconstruction_row("short", cuts=(1,)),
        reconstruction_row("kept", cuts=(2,)),
        reconstruction_row("long", cuts=(6,)),
        reconstruction_row("content", cuts=(5,), char_end=7),
        reconstruction_row("continuation", cuts=(3,), char_end=7),
    ]
    root = write_reconstruction(tmp_path, train=rows)
    splits, statistics = load_reconstruction_pretraining(
        root, tokenizer, 2, 5, "reconstruction_single"
    )
    assert [row.sample_id for row in splits["train"]] == ["kept:ae", "kept:continuation"]
    counts = statistics["splits"]["train"]
    assert counts["source_index_candidates"] == 5
    assert counts["kept_source_indices"] == 1
    assert counts["read"] == 10
    for reason in ("too_short", "too_long", "content_length", "continuation_length"):
        assert counts[f"filtered_{reason}"] == 2


@pytest.mark.parametrize("leak", ("document", "cluster", "sample_id"))
def test_reconstruction_source_isolation_before_filtering(tmp_path, tokenizer, leak):
    train = reconstruction_row("train", cuts=(1,))
    dev = reconstruction_row("dev", document_id="dev-document")
    if leak == "document":
        dev["document_id"] = train["document_id"]
    elif leak == "cluster":
        dev["dedup_cluster"] = train["dedup_cluster"]
    else:
        dev["sample_id"] = train["sample_id"]
    root = write_reconstruction(tmp_path, train=[train], dev=[dev])
    with pytest.raises(ValueError, match="dev.jsonl:1:.*(source document|dedup cluster|duplicate)"):
        load_reconstruction_pretraining(root, tokenizer, 2, 5, "reconstruction_single")


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"document_id": "wrong"}, "does not match document_id"),
        ({"char_start": -1}, "character interval"),
        ({"char_end": 100}, "character interval"),
        ({"write_token_ends": [3, 2]}, "target token cuts"),
        ({"source_file": "/absolute/path"}, "relative source path"),
        ({"row_group": -1}, "nonnegative row location"),
    ],
)
def test_reconstruction_invalid_source_fails(tmp_path, tokenizer, changes, match):
    root = write_reconstruction(tmp_path, train=[reconstruction_row(cuts=(3,), **changes)])
    with pytest.raises(ValueError, match=f"train.jsonl:1:.*{match}"):
        load_reconstruction_pretraining(root, tokenizer, 2, 5, "reconstruction_single")


def test_reconstruction_length_filter_does_not_read_unneeded_source(tmp_path, tokenizer):
    root = write_reconstruction(
        tmp_path,
        directory="multi",
        train=[reconstruction_row(cuts=(1, 4, 7), source_file="../raw/not-needed.parquet")],
    )
    splits, statistics = load_reconstruction_pretraining(
        root, tokenizer, 2, 3, "reconstruction_first_write"
    )
    assert not splits["train"]
    assert statistics["splits"]["train"]["filtered_too_short"] == 2


def test_reconstruction_preserves_index_order_across_row_groups(tmp_path, tokenizer):
    root = write_reconstruction(
        tmp_path,
        documents=[
            {"id": "document", "text": "a b c d e a b c d e"},
            {"id": "second", "text": "b c d e a b c d e a"},
        ],
        train=[
            reconstruction_row("one"),
            reconstruction_row("two", document_id="second", row_group=1),
            reconstruction_row("three", cuts=(4,)),
        ],
    )
    splits, _ = load_reconstruction_pretraining(
        root, tokenizer, 2, 5, "reconstruction_single", lm_only=True
    )
    assert [row.sample_id for row in splits["train"]] == [
        "one:continuation",
        "two:continuation",
        "three:continuation",
    ]


@pytest.mark.parametrize(
    "method", ("icae_single", "icae_multi", "autocompressors", "memory_change")
)
def test_runtime_loads_selected_reconstruction_view(tmp_path, tokenizer, method):
    dynamic = method == "memory_change"
    ac = method == "autocompressors"
    view = "reconstruction_first_write" if dynamic else "reconstruction_single"
    cuts = (3, 5, 8) if dynamic else (3,)
    root = write_reconstruction(
        tmp_path,
        directory="multi" if dynamic else "single",
        documents=[
            {"id": "document", "text": "a b c d e a b c d e"},
            {"id": "dev-document", "text": "a b c d e a b c d e"},
        ],
        train=[reconstruction_row(cuts=cuts)],
        dev=[reconstruction_row("dev", "dev-document", cuts, row_group=1)],
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
