import copy
import csv
from io import BytesIO
import json
import random
from urllib.error import HTTPError

import pytest
from tokenizers.pre_tokenizers import WhitespaceSplit

from latent_working_memory.data_preparation.dynamic import (
    evaluation_identity,
    load_evaluation_plan,
    prepare_dynamic,
)
from latent_working_memory.data_preparation.personamem import construction
from latent_working_memory.data_preparation.personamem import sources
from latent_working_memory.data_preparation.personamem.construction import AnnotationClient, locate
from latent_working_memory.data_preparation.personamem.audit import scores
from latent_working_memory.data_preparation.personamem.blocks import evidence_blocks
from latent_working_memory.v1.dynamic_config import DynamicConfig
from latent_working_memory.v1.personamem import PersonaMemDataset


@pytest.fixture
def factual_example():
    candidate = dict(
        history_id="h1",
        persona_id="1",
        split="train",
        candidate_id="h1:0:2",
        source_rows=[5],
        message_start=0,
        messages=[
            dict(message_id="m0001", role="user", content="In my draft, Maya visits Blue Note."),
            dict(message_id="m0002", role="assistant", content="The draft sounds clear."),
        ],
    )
    qa = dict(
        question="In the draft, where does Maya visit?",
        answer="Blue Note",
        answer_message_id="m0001",
        evidence_quote="Maya visits Blue Note.",
        subject="Maya, the draft character",
        subject_type="text_character",
        temporal_scope="In the original draft",
        fact_type="place",
        evidence_start_message_id="m0001",
        evidence_end_message_id="m0001",
    )
    return candidate, dict(qas=[qa], skip_reason="")


def test_evidence_keeps_original_message_and_character_offsets(factual_example):
    candidate, generated = factual_example
    good, rejected = locate(candidate, generated)
    assert not rejected
    qa = good[0]
    original = candidate["messages"][0]["content"]
    assert original[qa["answer_char_start"] : qa["answer_char_end_exclusive"]] == "Blue Note"
    assert qa["evidence_messages"] == candidate["messages"][:1]
    assert qa["evidence_message_end_exclusive"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("answer", "Invented location"),
        ("evidence_start_message_id", "m0002"),
        ("answer_message_id", "m9999"),
        ("temporal_scope", ""),
    ],
)
def test_invalid_or_uncontained_evidence_is_rejected(factual_example, field, value):
    candidate, generated = factual_example
    generated["qas"][0][field] = value
    good, rejected = locate(candidate, generated)
    assert not good
    assert len(rejected) == 1


def test_rewording_same_answer_span_does_not_add_a_fact(factual_example):
    candidate, generated = factual_example
    duplicate = copy.deepcopy(generated["qas"][0])
    duplicate["question"] = "Which place is visited by Maya in that draft?"
    generated["qas"].append(duplicate)
    good, rejected = locate(candidate, generated)
    assert len(good) == len(rejected) == 1


def test_empty_output_must_explain_skip(factual_example):
    candidate, _ = factual_example
    with pytest.raises(ValueError, match="inconsistent"):
        locate(candidate, dict(qas=[], skip_reason=""))


def test_source_preparation_needs_only_raw_data_and_explicit_exclusions(tmp_path, monkeypatch):
    cache = tmp_path / "raw_cache"
    cache.mkdir()
    users = [str(i) for i in range(20)]
    shuffled = list(users)
    random.Random(42).shuffle(shuffled)
    excluded = shuffled[16]
    rows = []
    for user in users:
        message = dict(role="user", content=f"In this account, my name is Person {user}.")
        (cache / f"{user}.json").write_text(
            json.dumps(
                {
                    "metadata": {"persona_id": int(user)},
                    "chat_history": [message],
                }
            )
        )
        rows.append(
            dict(
                persona_id=user,
                chat_history_32k_link=f"data/persona{user}.json",
                related_conversation_snippet=json.dumps([message]),
                pref_type="neutral_preferences",
                conversation_scenario="chat_message",
            )
        )
    csv_path = cache / "persona_train.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def forbid_download(*args, **kwargs):
        raise AssertionError("all selected histories should use the formal raw cache")

    monkeypatch.setattr(sources, "urlopen", forbid_download)
    output = tmp_path / "dataset"
    config = dict(
        dataset_dir=str(output),
        source_csv=str(csv_path),
        history_cache=str(cache),
        source_url="https://example.invalid",
        seed=42,
        split_users={"train": 1, "dev": 1, "test": 1},
        excluded_from_evaluation=[excluded],
    )
    result = sources.prepare(config)
    selection = json.loads((output / "selection.json").read_text())
    assert len(result["histories"]) == 3
    assert selection["excluded_from_evaluation"] == [excluded]
    assert all(h["persona_id"] != excluded for h in result["histories"] if h["split"] != "train")
    assert sources.prepare(config) == result
    candidate_order = (output / "candidates.json").read_text()
    (output / "sources.json").unlink()
    sources.prepare(config)
    assert (output / "candidates.json").read_text() == candidate_order


def write_factual_dataset(path, candidate, qas):
    (path / "histories").mkdir(parents=True)
    identity = {k: candidate[k] for k in ("history_id", "persona_id", "split")}
    (path / "sources.json").write_text(json.dumps({"histories": [identity]}))
    (path / "histories" / f"{candidate['persona_id']}.json").write_text(
        json.dumps(dict(identity, messages=candidate["messages"]))
    )
    (path / "qas.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qas))


def test_evidence_blocks_and_episode_preserve_earliest_legal_read(
    factual_example, tokenizer, tmp_path
):
    candidate, generated = factual_example
    good, _ = locate(candidate, generated)
    blocks = evidence_blocks(candidate, good)
    assert len(blocks) == 2
    assert len(blocks[0]["qas"]) == 1 and not blocks[1]["qas"]
    lengths = [
        len(tokenizer.encode(b["context"] + "\n\n", add_special_tokens=False)) for b in blocks
    ]
    dataset = tmp_path / "dataset"
    write_factual_dataset(dataset, candidate, good)
    before = {p.relative_to(dataset): p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
    data = PersonaMemDataset(dataset, tokenizer)
    episode = data.episode("h1")
    assert episode.reads[0].prefix_end == episode.write_ends[0]
    assert episode.reads[0].references[0].evidence_spans == ((0, lengths[0]),)
    assert data.episode("h1", 1, 1).reads == ()
    assert data.tokenizer is tokenizer
    assert before == {
        p.relative_to(dataset): p.read_bytes() for p in dataset.rglob("*") if p.is_file()
    }


def test_shared_dataset_works_with_different_tokenizers_and_plans_reject_mismatch(
    factual_example, tokenizer, tmp_path
):
    candidate, generated = factual_example
    qas, _ = locate(candidate, generated)
    dataset = tmp_path / "dataset"
    write_factual_dataset(dataset, candidate, qas)
    other_tokenizer = copy.deepcopy(tokenizer)
    other_tokenizer.backend_tokenizer.pre_tokenizer = WhitespaceSplit()
    first = PersonaMemDataset(dataset, tokenizer)
    second = PersonaMemDataset(dataset, other_tokenizer)
    assert first.articles == second.articles
    assert len(first.episode("h1").input_ids) != len(second.episode("h1").input_ids)
    assert (
        first.episode("h1").reads[0].references[0].text
        == second.episode("h1").reads[0].references[0].text
    )
    recipe = DynamicConfig()
    plan_path = tmp_path / "evaluation-plan.json"
    plan_path.write_text(
        json.dumps(
            dict(data_index=first.index, selection=evaluation_identity(recipe), panels={}, reads={})
        )
    )
    load_evaluation_plan(plan_path, first, recipe)
    with pytest.raises(ValueError, match="shared evaluation data"):
        load_evaluation_plan(plan_path, second, recipe)
    assert not (dataset / "index.json").exists() and not (dataset / "derived").exists()


def test_experiment_preparation_cannot_write_into_shared_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    with pytest.raises(ValueError, match="outside the shared dataset"):
        prepare_dynamic(dataset, None, None, dataset / "derived", dataset="personamem")


def test_overlapping_evidence_is_kept_in_one_block(factual_example):
    candidate, generated = factual_example
    good, _ = locate(candidate, generated)
    second = copy.deepcopy(good[0])
    second.update(
        qa_id="h1:other", evidence_message_end_exclusive=2, evidence_messages=candidate["messages"]
    )
    blocks = evidence_blocks(candidate, good + [second])
    assert len(blocks) == 1 and len(blocks[0]["qas"]) == 2


def test_diagnostic_scoring_distinguishes_unknown_and_partial_answer():
    assert scores("unknown", "Blue Note") == dict(em=0, f1=0)
    assert scores("The Blue Note.", "Blue Note") == dict(em=1, f1=1)
    assert scores("Blue", "Blue Note")["f1"] == pytest.approx(2 / 3)


def test_request_retry_and_resume_preserve_successful_output(tmp_path, monkeypatch):
    response = dict(
        status="completed",
        model="gpt-6-astra",
        reasoning={"effort": "medium"},
        output=[
            dict(
                type="message",
                content=[dict(type="output_text", text=json.dumps({"answer": "Blue Note"}))],
            )
        ],
        usage={},
    )
    calls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(response).encode()

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            if len(calls) == 1:
                raise HTTPError(
                    request.full_url, 429, "limited", {"Retry-After": "0"}, BytesIO(b"limited")
                )
            return Response()

    monkeypatch.setattr(construction, "build_opener", lambda *args: Opener())
    config = dict(
        endpoint="http://127.0.0.1:4141/v1/responses",
        model="gpt-6-astra",
        reasoning_effort="medium",
        max_output_tokens=4096,
        timeout_seconds=120,
        max_attempts=3,
    )
    client = AnnotationClient(config, tmp_path)
    monkeypatch.setattr(client.stop, "wait", lambda seconds: False)
    args = ("one", "generate", "instruction", {"question": "Where?"}, {"type": "object"})
    assert client.call(*args) == {"answer": "Blue Note"}
    assert client.call(*args) == {"answer": "Blue Note"}
    assert len(calls) == 2
    records = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert [r["http_status"] for r in records] == [429, 200]
    with pytest.raises(ValueError, match="request changed"):
        client.call("one", "generate", "different instruction", args[3], args[4])
    assert len(calls) == 2
