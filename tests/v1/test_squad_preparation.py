import json
import random

import pytest

from latent_working_memory.data_preparation.squad import (
    assign_sources,
    load_articles,
    prepare_squad,
)
from latent_working_memory.v1.squad import SquadDataset, sample_reads


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
    output = tmp_path / "records" / "test-tokenizer_index.json"
    report = prepare_squad(paths["train"], paths["dev"], tokenizer, output)
    assert before == {k: p.read_bytes() for k, p in paths.items()}
    assert report["splits"]["train"]["articles"] == 11
    assert report["splits"]["dev"]["articles"] == 1
    assert report["splits"]["test"]["articles"] == 2
    assert list(output.parent.iterdir()) == [output]
    with pytest.raises(FileExistsError):
        prepare_squad(paths["train"], paths["dev"], tokenizer, output)
    return SquadDataset(output)


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
