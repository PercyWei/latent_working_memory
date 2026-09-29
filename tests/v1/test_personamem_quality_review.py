import json

from latent_working_memory.data_preparation.personamem import quality_review
from latent_working_memory.data_preparation.personamem.quality_review import (
    exact_groups,
    near_pairs,
    normalized,
)


def test_shared_answer_does_not_make_distinct_facts_duplicates():
    records = [
        dict(id="a", split="train", text="Which city did Maya visit?", answer="Paris"),
        dict(id="b", split="test", text="Which city did Noah leave?", answer="Paris"),
    ]
    assert exact_groups(records, lambda r: normalized(r["text"]) + normalized(r["answer"])) == []


def test_cross_split_material_match_ignores_ids_case_and_punctuation():
    records = [
        dict(id="user1/m1", split="train", text="Maya took the blue notebook to the station."),
        dict(id="user2/m9", split="test", text="MAYA took the blue notebook to the station!"),
        dict(id="user3/m5", split="train", text="A different document in the same split."),
    ]
    groups = exact_groups(records, lambda r: normalized(r["text"]), min_words=8)
    assert len(groups) == 1
    assert groups[0]["ids"] == ["user1/m1", "user2/m9"]


def test_near_match_detects_contained_material_without_equal_answers():
    passage = "Maya visited the old riverside market and bought a blue notebook before sunset"
    records = [
        dict(id="a", split="train", text=passage),
        dict(
            id="b",
            split="test",
            text="Original letter: " + passage + " The next day was different.",
        ),
        dict(id="c", split="train", text="Original letter: " + passage),
    ]
    pairs, _ = near_pairs(records, 3, 8, 0.9, 0.85, 6)
    assert any(p["left"] == "a" and p["right"] == "b" for p in pairs)
    assert not any({p["left"], p["right"]} == {"a", "c"} for p in pairs)


def test_blind_probe_sends_only_question_and_records_numeric_scores(tmp_path, monkeypatch):
    qa = dict(
        qa_id="q1", split="test", question="Which hall hosted the event?", answer="Harbor Hall"
    )
    monkeypatch.setattr(quality_review, "load_dataset", lambda path: ([qa], {}, {}, []))
    (tmp_path / "blind-probe-selection.json").write_text(json.dumps({"qa_ids": ["q1"]}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": "review-model", "concurrency": 1}))
    payloads = []

    def call(self, key, stage, instructions, content, schema):
        payloads.append(content)
        return dict(answer="Harbor Hall", confidence=0.5, reason="A guess")

    monkeypatch.setattr(quality_review.AnnotationClient, "call", call)
    quality_review.blind_probes(tmp_path, tmp_path, config)
    result = json.loads((tmp_path / "blind-probes.json").read_text())["results"][0]
    assert payloads == [{"question": qa["question"]}]
    assert result["em"] == 1.0 and result["f1"] == 1.0
