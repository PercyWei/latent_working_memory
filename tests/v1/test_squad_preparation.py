from copy import deepcopy
import json
import random

import pytest
import torch
from latent_working_memory.data_preparation import squad_migration
from latent_working_memory.data_preparation.squad_migration import migrate_squad, same_payload
from latent_working_memory.v1.checkpoint import save_model_checkpoint, capture_rng_state
from tokenizers.pre_tokenizers import Split
from transformers import AutoTokenizer
from latent_working_memory.v1.dynamic import squad as squad_runtime

from latent_working_memory.data_preparation.squad import (
    RECORD_FIELDS,
    paragraph_lengths,
    split_statistics,
    assign_sources,
    load_articles,
    prepare_squad,
)
from latent_working_memory.v1.dynamic.squad import SquadDataset, sample_reads


def article(index, split="train", title=None, text=None):
    return {
        "document_id": f"squad:{split}:{index}",
        "official_split": split,
        "title": title or f"Article {index}",
        "paragraphs": [{"context": text or f"Text {index}", "qas": []}],
    }


def test_transitive_source_isolation():
    rows = [
        article(0, title="Shared"),
        article(1, title="Shared", text="Duplicate"),
        article(2, "dev", text="Duplicate"),
    ] + [article(i) for i in range(3, 23)]
    mapping = assign_sources(rows, 42)
    assert mapping["squad:train:0"]["split"] == "excluded"
    assert mapping["squad:train:1"]["split"] == "excluded"
    assert mapping["squad:dev:2"]["split"] == "test"
    assert sum(v["split"] == "dev" for v in mapping.values()) == 2
    assert mapping == assign_sources(rows, 42)


def test_wrong_answer_offset_fails(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.1",
                "data": [
                    {
                        "title": "A",
                        "paragraphs": [
                            {
                                "context": "First sentence.",
                                "qas": [
                                    {
                                        "id": "q",
                                        "question": "What?",
                                        "answers": [{"text": "First", "answer_start": 1}],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="answer annotation mismatch"):
        load_articles(path, "train")


@pytest.fixture
def dataset(tmp_path, tokenizer):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    tokenizer.name_or_path = str(tokenizer_dir)
    paths = {}
    for split, indices in (("train", range(12)), ("dev", range(20, 22))):
        data = []
        for i in indices:
            paragraphs = []
            for p in range(10):
                # One paragraph already exceeds the former 512-token cap.
                text = "First " + "One " * (1200 if p == 0 else 10) + f"{i}-{p}."
                paragraphs.append(
                    {
                        "context": text,
                        "qas": [
                            {
                                "id": f"q{i}-{p}-{q}",
                                "question": "What?",
                                "answers": [
                                    {"text": "First", "answer_start": 0},
                                    {"text": "First", "answer_start": 0},
                                    {"text": "One", "answer_start": 6},
                                ],
                            }
                            for q in range(5)
                        ],
                    }
                )
            data.append({"title": f"Title {i}", "paragraphs": paragraphs})
        paths[split] = tmp_path / f"{split}.json"
        paths[split].write_text(json.dumps({"version": "1.1", "data": data}))
    before = {k: p.read_bytes() for k, p in paths.items()}
    output = tmp_path / "squad"
    report = prepare_squad(paths["train"], paths["dev"], tokenizer, output)
    assert before == {k: p.read_bytes() for k, p in paths.items()}
    assert report["splits"]["train"]["articles"] == 11
    assert report["splits"]["dev"]["articles"] == 1
    assert report["splits"]["test"]["articles"] == 2
    assert {p.name for p in output.iterdir()} == {
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "preparation.json",
    }
    with pytest.raises(FileExistsError):
        prepare_squad(paths["train"], paths["dev"], tokenizer, output)
    return SquadDataset(output, tokenizer)


def test_full_article_keeps_all_paragraphs_and_questions(dataset):
    doc = dataset.select("train", 0, 10000)[0]
    episode = dataset.episode(doc)
    assert len(episode.write_ends) == 10
    assert episode.write_ends[0] > 1024
    assert len(episode.reads) == 50
    for source in episode.sources:
        p = source.provenance["paragraph_index"]
        raw = dataset.articles[doc]["paragraphs"][p]
        expected = tuple(
            dataset.tokenizer.encode(raw["context"] + "\n\n", add_special_tokens=False)
        )
        assert episode.input_ids[source.token_start : source.token_end] == expected
        assert source.provenance["questions"] == raw["qas"]
    assert [r.text for r in episode.reads[0].references] == ["First", "One"]
    length = len(episode.input_ids)
    assert doc in dataset.select("train", length, length)
    assert doc not in dataset.select("train", length + 1, length + 100)
    prefix = dataset.episode(doc, paragraph_count=2)
    assert prefix.input_ids == episode.input_ids[: episode.write_ends[1]]
    assert len(prefix.reads) == 10
    assert prefix.episode_id != episode.episode_id


def test_runtime_reads_are_causal_reproducible_and_capped(dataset):
    e = dataset.episode(dataset.select("train", 0, 10000)[0])
    original = e.to_record()
    visits, visits2 = {}, {}
    rng, rng2 = random.Random(42), random.Random(42)
    for boundary in e.write_ends:
        reads = sample_reads(e, boundary, 2, 3, rng, visits, 2)
        assert reads == sample_reads(e, boundary, 2, 3, rng2, visits2, 2)
        assert len(reads) <= 5
        assert len({r.read_id for r in reads}) == len(reads)
        assert all(r.prefix_end == boundary for r in reads)
        assert all(
            end <= boundary for r in reads for ref in r.references for _, end in ref.evidence_spans
        )
    assert max(visits.values()) <= 2
    assert original == e.to_record()
    assert sample_reads(e, e.write_ends[-1], 0, 0, rng, visits, 2) == ()
    with pytest.raises(ValueError, match="committed"):
        sample_reads(e, 1, 2, 2, rng, visits, 2)


def test_stale_length_index_is_rejected(dataset):
    doc = dataset.select("train", 0, 10000)[0]
    dataset.records[doc]["paragraph_tokens"][0] += 1
    with pytest.raises(ValueError, match="rebuild"):
        dataset.episode(doc)


def test_reference_files_have_only_locations_and_attributes(dataset):
    root = dataset.dataset_dir
    meta = json.loads((root / "preparation.json").read_text())
    assert set(p.name for p in root.iterdir()) == {
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "preparation.json",
    }
    assert meta["reference_tokenizer"]["backend_fingerprint"]
    assert meta["serialization"]["paragraph_suffix"] == "\n\n"
    assert meta["serialization"]["add_special_tokens"] is False
    for split in ("train", "dev", "test"):
        rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        for row in rows:
            assert set(row) == RECORD_FIELDS
            assert (
                not {"text", "context", "qas", "question", "answers", "split", "input_tokens"}
                & row.keys()
            )
            assert row["reference_input_tokens"] == sum(row["reference_paragraph_tokens"])
            assert row["paragraph_count"] == len(row["reference_paragraph_tokens"])
            assert dataset.records[row["document_id"]]["split"] == split


def test_same_backend_reuses_lengths_without_loading_reference_tokenizer(dataset, monkeypatch):
    other = deepcopy(dataset.tokenizer)
    other.name_or_path = "another-location-for-the-same-tokenizer"

    def forbidden(*args, **kwargs):
        pytest.fail(
            "matching reference lengths should not require remeasurement or tokenizer loading"
        )

    monkeypatch.setattr(squad_runtime, "paragraph_lengths", forbidden)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbidden)
    loaded = SquadDataset(dataset.dataset_dir, other)
    assert loaded.reference_lengths
    assert loaded.records == dataset.records
    assert loaded.index == dataset.index
    doc = loaded.select("train", 0, 10000)[0]
    assert loaded.episode(doc) == dataset.episode(doc)


def test_same_name_and_vocab_with_changed_backend_recomputes_lengths(dataset):
    tokenizer = deepcopy(dataset.tokenizer)
    tokenizer.backend_tokenizer.pre_tokenizer = Split("", behavior="isolated")
    assert tokenizer.name_or_path == dataset.tokenizer.name_or_path
    assert tokenizer.get_vocab() == dataset.tokenizer.get_vocab()
    before = {p.name: p.read_bytes() for p in dataset.dataset_dir.iterdir()}
    other = SquadDataset(dataset.dataset_dir, tokenizer)
    assert not other.reference_lengths
    assert other.index != dataset.index
    doc = next(iter(other.records))
    expected = paragraph_lengths(other.articles[doc], tokenizer)
    assert other.records[doc]["paragraph_tokens"] == expected
    assert expected != dataset.records[doc]["reference_paragraph_tokens"]
    length = sum(expected)
    assert other.records[doc]["input_tokens"] == length
    assert len(other.episode(doc).input_ids) == length
    assert doc in other.select(other.records[doc]["split"], length, length)
    assert before == {p.name: p.read_bytes() for p in dataset.dataset_dir.iterdir()}


def test_reference_serialization_mismatch_forces_remeasurement(dataset, monkeypatch):
    meta_path = dataset.dataset_dir / "preparation.json"
    meta = json.loads(meta_path.read_text())
    meta["serialization"]["paragraph_suffix"] = ""
    meta_path.write_text(json.dumps(meta))
    calls = []
    original = squad_runtime.paragraph_lengths

    def measure(article, tokenizer):
        calls.append(article["document_id"])
        return original(article, tokenizer)

    monkeypatch.setattr(squad_runtime, "paragraph_lengths", measure)
    other = SquadDataset(dataset.dataset_dir, dataset.tokenizer)
    assert not other.reference_lengths
    assert set(calls) == set(other.records)


def test_exclusions_and_split_ownership_are_preserved(dataset):
    meta = dataset.preparation
    source_train = dataset.dataset_dir / meta["source_files"]["train"]
    original = json.loads(source_train.read_text())
    original["data"][0]["title"] = "Title 20"
    source_train.write_text(json.dumps(original))
    output = dataset.dataset_dir.parent / "isolated"
    prepared = prepare_squad(
        source_train,
        dataset.dataset_dir / meta["source_files"]["dev"],
        dataset.tokenizer,
        output,
        meta["seed"],
    )
    assert len(prepared["excluded"]) == 1
    excluded = prepared["excluded"][0]
    assert excluded["document_id"] == "squad:train:0"
    assert excluded["reason"] == "source_group_overlaps_official_dev"
    loaded = SquadDataset(output, dataset.tokenizer)
    assert excluded["document_id"] not in loaded.records
    assert excluded["document_id"] not in loaded.articles
    for split in ("train", "dev", "test"):
        assert excluded["document_id"] not in (output / f"{split}.jsonl").read_text()
    # Even consistent-looking statistics cannot authorize moving a source to another split.
    train_rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    dev_rows = [json.loads(line) for line in (output / "dev.jsonl").read_text().splitlines()]
    dev_rows.append(train_rows.pop())
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        (output / f"{split}.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
        prepared["splits"][split] = split_statistics(rows)
    (output / "preparation.json").write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match="group or split"):
        SquadDataset(output, dataset.tokenizer)


def test_partial_or_unknown_dataset_files_are_rejected(dataset):
    extra = dataset.dataset_dir / "tokenizer_index.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="requires"):
        SquadDataset(dataset.dataset_dir, dataset.tokenizer)
    extra.unlink()
    (dataset.dataset_dir / "test.jsonl").unlink()
    with pytest.raises(ValueError, match="requires"):
        SquadDataset(dataset.dataset_dir, dataset.tokenizer)


def legacy_index(dataset, root):
    root.mkdir()
    meta = dataset.preparation
    rows = []
    for split in ("train", "dev", "test", "excluded"):
        selected = (
            meta["excluded"]
            if split == "excluded"
            else [r for r in dataset.records.values() if r["split"] == split]
        )
        for row in selected:
            rows.append(
                {
                    "document_id": row["document_id"],
                    "official_split": row["official_split"],
                    "article_index": row["article_index"],
                    "title": row["title"],
                    "group": row["group_id"],
                    "split": split,
                    "input_tokens": row["reference_input_tokens"],
                    "paragraph_tokens": row["reference_paragraph_tokens"],
                    "questions": row["question_count"],
                }
            )
    original = {
        "created_at": meta["created_at"],
        "seed": meta["seed"],
        "tokenizer": dataset.tokenizer.name_or_path,
        "source_files": {
            name: str((dataset.dataset_dir / path).resolve())
            for name, path in meta["source_files"].items()
        },
        "articles": rows,
    }
    index = root / "reference_index.json"
    index.write_text(json.dumps(original))
    return index, original


def test_explicit_migration_preserves_checkpoint_and_plan(dataset, tmp_path, tiny_config):
    index, original = legacy_index(dataset, tmp_path / "legacy")
    plan = {"data_index": original, "selection": {"seed": 42}, "panels": {"test": []}, "reads": {}}
    plan_path = tmp_path / "evaluation-plan.json"
    plan_path.write_text(json.dumps(plan))
    checkpoint = tmp_path / "dynamic.pt"
    progress = {
        "identity": {"evaluation_plan": plan, "world_size": 2},
        "next_step": 7,
        "rank_rng_states": [torch.get_rng_state()],
    }
    save_model_checkpoint(
        checkpoint,
        "dynamic",
        tiny_config,
        {"weight": torch.arange(5)},
        {"momentum": torch.arange(3)},
        progress,
        capture_rng_state(),
    )
    before = torch.load(checkpoint, weights_only=True)
    report = migrate_squad(index, tmp_path / "migration-record", [plan_path], [checkpoint])
    loaded = SquadDataset(index.parent, dataset.tokenizer)
    assert report["source_order_splits_and_reference_lengths_unchanged"]
    assert not index.exists()
    after = torch.load(checkpoint, weights_only=True)
    before["progress"]["identity"]["evaluation_plan"]["data_index"] = loaded.index
    assert same_payload(before, after)
    plan["data_index"] = loaded.index
    assert json.loads(plan_path.read_text()) == plan
    assert json.loads((tmp_path / "migration-record/original-index.json").read_text()) == original
    for doc in dataset.records:
        assert loaded.episode(doc) == dataset.episode(doc)


def test_migration_rejects_changed_lengths_before_mutation(dataset, tmp_path):
    index, original = legacy_index(dataset, tmp_path / "legacy")
    original["articles"][0]["input_tokens"] += 1
    index.write_text(json.dumps(original))
    before = index.read_bytes()
    with pytest.raises(ValueError, match="lengths changed"):
        migrate_squad(index, tmp_path / "migration-record")
    assert index.read_bytes() == before
    assert list(index.parent.iterdir()) == [index]
    assert not list(tmp_path.glob(".squad-format-*"))


def test_migration_rolls_back_dataset_plan_and_checkpoint(
    dataset, tmp_path, tiny_config, monkeypatch
):
    index, original = legacy_index(dataset, tmp_path / "legacy")
    plan = {"data_index": original}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    checkpoint = tmp_path / "dynamic.pt"
    save_model_checkpoint(
        checkpoint,
        "dynamic",
        tiny_config,
        {},
        {},
        {"identity": {"evaluation_plan": plan}},
        capture_rng_state(),
    )
    before = {p: p.read_bytes() for p in (index, plan_path, checkpoint)}

    def fail(*args):
        raise OSError("failed checkpoint save")

    monkeypatch.setattr(squad_migration, "_atomic_torch_save", fail)
    with pytest.raises(OSError, match="failed checkpoint save"):
        migrate_squad(index, tmp_path / "migration-record", [plan_path], [checkpoint])
    assert before == {p: p.read_bytes() for p in before}
    assert list(index.parent.iterdir()) == [index]
