import csv
import json
from pathlib import Path
import re
import threading

import pytest

from latent_working_memory.data_preparation.personamem import artifacts, construction, pipeline


@pytest.fixture
def offline_construction(tmp_path, monkeypatch):
    cache = tmp_path / "input"
    cache.mkdir()
    rows = []
    for i in range(20):
        message = dict(role="user", content=f"In Person{i}'s original draft, I visited Place{i}.")
        (cache / f"{i}.json").write_text(
            json.dumps(dict(metadata={"persona_id": i}, chat_history=[message]))
        )
        rows.append(
            dict(
                persona_id=str(i),
                chat_history_32k_link=f"data/persona{i}.json",
                related_conversation_snippet=json.dumps([message]),
                pref_type="neutral_preferences",
                conversation_scenario="chat_message",
            )
        )
    source_csv = cache / "persona_train.csv"
    with source_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = dict(
        endpoint="http://127.0.0.1:4141/v1/responses",
        model="gpt-6-astra",
        reasoning_effort="medium",
        max_output_tokens=4096,
        timeout_seconds=120,
        concurrency=8,
        max_attempts=3,
        seed=42,
        split_users={"train": 2, "dev": 2, "test": 2},
        pilot_candidates=2,
        diagnostic_questions=1,
        target_qas={"train": 2, "dev": 2, "test": 2},
        dataset_dir=str(tmp_path / "dataset"),
        artifacts_dir=str(tmp_path / "records"),
        source_csv=str(source_csv),
        history_cache=str(cache),
        source_url="https://example.invalid/dataset/resolve/main",
        excluded_from_evaluation=[],
    )
    calls, lock = [], threading.Lock()

    class Response:
        status = 200

        def __init__(self, payload, output):
            self.payload, self.output = payload, output

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                dict(
                    status="completed",
                    model=self.payload["model"],
                    reasoning=self.payload["reasoning"],
                    output=[
                        dict(
                            type="message",
                            content=[dict(type="output_text", text=json.dumps(self.output))],
                        )
                    ],
                    usage={"input_tokens": 10, "output_tokens": 5},
                )
            ).encode()

    class Opener:
        def open(self, request, timeout):
            payload = json.loads(request.data)
            stage = payload["text"]["format"]["name"].removeprefix("personamem_")
            content = json.loads(payload["input"][0]["content"])
            with lock:
                calls.append(stage)
            if stage == "generate":
                m = content["messages"][0]
                person = re.search(r"Person\d+", m["content"])[0]
                answer = re.search(r"Place\d+", m["content"])[0]
                qa = dict(
                    question=f"In {person}'s original draft, which place was visited?",
                    answer=answer,
                    answer_message_id=m["message_id"],
                    evidence_quote=m["content"],
                    subject=person,
                    subject_type="text_character",
                    temporal_scope="The original draft",
                    fact_type="place",
                    evidence_start_message_id=m["message_id"],
                    evidence_end_message_id=m["message_id"],
                )
                output = {"qas": [qa], "skip_reason": ""}
            elif stage == "verify":
                output = {
                    "decisions": [
                        dict(
                            qa_id=q["qa_id"],
                            accepted=True,
                            revisit_safe=True,
                            reason="Fixture evidence supports the historical question.",
                        )
                        for q in content["qas"]
                    ]
                }
            elif stage in {"gold", "wrong", "question_only"}:
                match = re.search(r"Place\d+", content["evidence"])
                output = {"answer": match[0] if match else "unknown"}
            elif stage == "review":
                output = {
                    "decisions": [
                        dict(
                            qa_id=q["qa_id"],
                            accepted=True,
                            reason="Fixture supported.",
                            blind_answer_correct=True if "blind_gold_answer" in q else None,
                        )
                        for q in content["qas"]
                    ]
                }
            elif stage == "dedup":
                output = {"removals": []}
            else:
                raise AssertionError(stage)
            return Response(payload, output)

    monkeypatch.setattr(construction, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(construction.signal, "signal", lambda *args: None)
    return config, calls


def test_complete_pipeline_publishes_only_construction_artifacts_and_reruns_offline(
    offline_construction,
):
    config, calls = offline_construction
    result = pipeline.run_all(config)
    assert result == dict(users=6, qas=6, artifact_contract="passed")
    dataset = Path(config["dataset_dir"])
    run = Path(config["artifacts_dir"])
    assert {p.name for p in dataset.iterdir()} == {
        "raw",
        "histories",
        "sources.json",
        "selection.json",
        "source_exclusions.json",
        "candidates.json",
        "qas.provisional.jsonl",
        "qas.jsonl",
        "README.md",
        "metadata.json",
    }
    assert (run / "completion.json").exists() and (
        run / "final_diagnostic/diagnostic_results.json"
    ).exists()
    assert not (run / "manual_review.json").exists()
    assert not any(
        (root / forbidden).exists()
        for root in [dataset, run]
        for forbidden in ["derived", "compatible_plan", "evaluation-plan.json"]
    )
    before = {
        str(p): p.read_bytes() for root in [dataset, run] for p in root.rglob("*") if p.is_file()
    }
    request_count = len(calls)
    assert pipeline.run_all(config) == result
    assert len(calls) == request_count
    assert before == {
        str(p): p.read_bytes() for root in [dataset, run] for p in root.rglob("*") if p.is_file()
    }
    (dataset / "README.md").unlink()
    (dataset / "metadata.json").unlink()
    artifacts.finalize(config)
    assert (dataset / "README.md").exists() and artifacts.verify(config)["qas"] == 6
    assert len(calls) == request_count


def test_unexpected_artifact_fails_before_any_model_request(offline_construction):
    config, calls = offline_construction
    path = Path(config["dataset_dir"]) / "derived"
    path.mkdir(parents=True)
    with pytest.raises(ValueError, match="unexpected=.*derived"):
        pipeline.run_all(config)
    assert calls == []


def test_missing_review_cannot_be_declared_completed(offline_construction):
    config, calls = offline_construction
    pipeline.run_all(config)
    run = Path(config["artifacts_dir"])
    (run / "source_review.json").unlink()
    request_count = len(calls)
    with pytest.raises(ValueError, match="missing=.*source_review"):
        pipeline.run_all(config)
    assert len(calls) == request_count


def test_missing_raw_response_cannot_pass_completion_check(offline_construction):
    config, calls = offline_construction
    pipeline.run_all(config)
    response = next((Path(config["artifacts_dir"]) / "requests").rglob("*.response.json"))
    response.unlink()
    request_count = len(calls)
    with pytest.raises(ValueError, match="request artifact mismatch"):
        artifacts.verify(config)
    assert len(calls) == request_count
