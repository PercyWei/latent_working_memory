from copy import deepcopy
import json
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from latent_working_memory.data_preparation.fineweb_qa import pipeline


def panel_fixture():
    selection = {"trajectories": [{"trajectory_id": f"doc-{i}"} for i in range(24)]}
    documents = {
        t["trajectory_id"]: {
            "assembly": {
                "accepted_qas": [
                    {
                        "qa_id": f"{t['trajectory_id']}:s{s}:q1",
                        "segment_id": s,
                        "question": "What is recorded?",
                        "answer": "Gold",
                        "evidence_quote": "Gold is recorded.",
                    }
                    for s in range(1, 9)
                ]
            }
        }
        for t in selection["trajectories"]
    }
    return selection, documents


def test_panel_covers_every_document_and_disperses_segment_positions():
    selection, documents = panel_fixture()
    config = {
        "source": {"seed": 20260928},
        "text": {"segments": 8},
        "review": {"question_limit": 96},
    }
    panel = pipeline.select_panel(selection, documents, config)
    assert len(panel) == 96
    for trajectory in selection["trajectories"]:
        items = [item for item in panel if item["trajectory_id"] == trajectory["trajectory_id"]]
        assert len(items) == 4
        assert len({item["qa"]["segment_id"] for item in items}) == 4
    assert {item["qa"]["segment_id"] for item in panel} == set(range(1, 9))


def test_diagnostic_requests_never_contain_gold_or_fact_statement(tmp_path, monkeypatch):
    selection, documents = panel_fixture()
    selection["trajectories"] = selection["trajectories"][:1]
    pipeline.save_json(tmp_path / "selection.json", selection)
    pipeline.save_json(tmp_path / "documents" / "doc-0.json", documents["doc-0"])
    prompt = tmp_path / "answer.txt"
    prompt.write_text("Answer the question.")
    captured = []

    class Client:
        def __init__(self, config):
            self.calls = []
            self.stop_event = Event()

        def call(self, stage, instructions, content, schema):
            captured.append(content)
            return {"answer": "Gold"}

    monkeypatch.setattr(pipeline, "AnnotationClient", Client)
    config = {
        "source": {"seed": 20260928},
        "text": {"segments": 8},
        "review": {
            "question_limit": 4,
            "conditions": ["evidence", "question_only"],
            "subagents": 2,
        },
        "annotation": {"concurrency": 2, "prompts": {"answer": str(prompt)}},
        "output": {"artifacts_dir": str(tmp_path)},
    }
    pipeline.save_json(tmp_path / "config.json", config)
    pipeline.save_json(tmp_path / "prompts.json", pipeline.load_prompts(config))
    result = pipeline.diagnose(config)
    assert len(captured) == 8
    assert sum(set(c) == {"question"} for c in captured) == 4
    assert sum(set(c) == {"question", "evidence"} for c in captured) == 4
    assert all(entry["evidence"]["em"] == 1 for entry in result["panel"])
    assert len(json.loads((tmp_path / "review-panel-1.json").read_text())) == 2


def test_review_merges_cross_stage_duplicate_groups_without_reviving_rejections():
    automatic = {
        "decisions": [{"qa_id": q, "accepted": q != "b", "reason": "review"} for q in "abcd"],
        "same_fact_groups": [["a", "b"]],
    }
    resolved = [{"qa_id": "c", "accepted": False, "reason": "Ambiguous", "same_fact_with": ["b"]}]
    result = pipeline.combine_review(automatic, resolved, set("abcd"))
    assert result["same_fact_groups"] == [["a", "b", "c"]]
    assert {r["qa_id"] for r in result["decisions"] if not r["accepted"]} == {"b", "c"}


@pytest.fixture
def frozen_batch(tmp_path, monkeypatch):
    prompt = tmp_path / "answer.txt"
    prompt.write_text("Answer using only the supplied context.")
    config = {
        "source": {"seed": 20260928, "trajectory_limit": 24},
        "text": {"segments": 8, "min_segment_chars": 3072},
        "annotation": {
            "endpoint": "http://localhost:4141/v1/responses",
            "model": "gpt-6-sol",
            "reasoning_effort": "medium",
            "concurrency": 4,
            "candidate_counts": [15, 10, 10, 10, 10, 10, 10, 5],
            "max_output_tokens": {"answer": 2048},
            "prompts": {"answer": str(prompt)},
        },
        "review": {"question_limit": 96, "conditions": ["evidence", "question_only"]},
        "output": {
            "artifacts_dir": str(tmp_path / "artifacts"),
            "dataset_dir": str(tmp_path / "dataset"),
            "request_cache_dir": str(tmp_path / "cache"),
        },
    }
    monkeypatch.setattr(
        pipeline,
        "prepare_sources",
        lambda _: {"trajectories": [], "statistics": {}, "source_provenance": {}},
    )
    pipeline.prepare(config)
    return config


def test_prepare_freezes_prompts_and_reuses_selection_with_new_endpoint(frozen_batch, monkeypatch):
    original = deepcopy(frozen_batch)
    frozen_batch["annotation"]["endpoint"] = "http://localhost:5141/v1/responses"

    def unexpected_source_read(config):
        pytest.fail("A frozen selection must not scan sources again")

    monkeypatch.setattr(pipeline, "prepare_sources", unexpected_source_read)
    assert pipeline.prepare(frozen_batch)["trajectories"] == []
    saved_config, prompts = pipeline.load_frozen_batch(frozen_batch)
    assert saved_config == original
    assert prompts == {"answer": "Answer using only the supplied context."}


@pytest.mark.parametrize(
    ("section", "field", "changed"),
    [
        ("source", "seed", 1),
        ("text", "segments", 7),
        ("annotation", "model", "different-model"),
        ("annotation", "reasoning_effort", "high"),
        ("annotation", "candidate_counts", [12] * 8),
        ("annotation", "max_output_tokens", {"answer": 1024}),
        ("review", "question_limit", 48),
        ("output", "dataset_dir", "another-output"),
    ],
)
@pytest.mark.parametrize("stage", ["prepare", "annotate", "diagnose", "finalize"])
def test_every_stage_rejects_changed_batch_settings(frozen_batch, section, field, changed, stage):
    frozen_batch[section][field] = changed
    args = (frozen_batch, 4) if stage == "annotate" else (frozen_batch,)
    with pytest.raises(ValueError, match="Frozen batch settings differ"):
        getattr(pipeline, stage)(*args)


@pytest.mark.parametrize("stage", ["prepare", "annotate", "diagnose", "finalize"])
def test_every_stage_rejects_prompt_content_changes(frozen_batch, stage):
    Path(frozen_batch["annotation"]["prompts"]["answer"]).write_text("Changed instructions.")
    args = (frozen_batch, 4) if stage == "annotate" else (frozen_batch,)
    with pytest.raises(ValueError, match="Frozen batch prompts differ"):
        getattr(pipeline, stage)(*args)


def test_prepare_recovery_keeps_frozen_identity_after_source_failure(tmp_path, monkeypatch):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Stable prompt")
    config = {
        "annotation": {"model": "gpt-6-sol", "prompts": {"answer": str(prompt)}},
        "output": {"artifacts_dir": str(tmp_path)},
    }

    def failed_source_read(_):
        raise OSError("source temporarily unavailable")

    monkeypatch.setattr(pipeline, "prepare_sources", failed_source_read)
    with pytest.raises(OSError, match="source temporarily unavailable"):
        pipeline.prepare(config)
    assert pipeline.load_frozen_batch(config) == (config, {"answer": "Stable prompt"})
    config["annotation"]["model"] = "different-model"
    with pytest.raises(ValueError, match="Frozen batch settings differ"):
        pipeline.prepare(config)


def test_finalize_records_frozen_provenance_and_preserves_cost_statistics(frozen_batch):
    directory = Path(frozen_batch["output"]["artifacts_dir"])
    original = deepcopy(frozen_batch)
    pipeline.save_json(directory / "diagnostics.json", {"panel": []})
    pipeline.save_json(directory / "resolved-review.json", [])
    call = {
        "stage": "generate",
        "cache_hit": False,
        "attempts": [
            {"ok": True, "elapsed_seconds": 1.5, "usage": {"input_tokens": 10, "output_tokens": 20}}
        ],
    }
    (directory / "requests.jsonl").write_text(json.dumps(call) + "\n")
    frozen_batch["annotation"]["endpoint"] = "http://localhost:5141/v1/responses"
    summary = pipeline.finalize(frozen_batch)
    assert summary["config"] == original
    assert summary["prompts"] == {"answer": "Answer using only the supplied context."}
    assert summary["requests_by_stage"]["generate"]["input_tokens"] == 10
    assert summary["requests_by_stage"]["generate"]["attempt_elapsed_seconds"] == 1.5


def test_bounded_execution_preserves_input_order(tmp_path):
    client = SimpleNamespace(stop_event=Event(), calls=[])
    release_first = Event()

    def process(index):
        if index == 0:
            assert release_first.wait(2)
        elif index == 1:
            release_first.set()
        return index * 2

    config = {"annotation": {"concurrency": 2}, "output": {"artifacts_dir": str(tmp_path)}}
    assert pipeline.run_bounded(
        process, list(range(8)), [str(i) for i in range(8)], client, config, "test"
    ) == list(range(0, 16, 2))
    assert not client.stop_event.is_set()


@pytest.mark.parametrize("failed_index", [0, 3])
def test_bounded_execution_stops_refilling_on_first_failure(tmp_path, failed_index):
    client = SimpleNamespace(stop_event=Event(), calls=[])
    first_group = Barrier(4)
    started = []

    def process(index):
        started.append(index)
        first_group.wait(timeout=2)
        if index == failed_index:
            raise ValueError("invalid response protocol")
        # Issued requests may finish; there must be no next task or next API call.
        assert client.stop_event.wait(2)
        return index

    config = {"annotation": {"concurrency": 4}, "output": {"artifacts_dir": str(tmp_path)}}
    with pytest.raises(ValueError, match="invalid response protocol"):
        pipeline.run_bounded(
            process, list(range(20)), [str(i) for i in range(20)], client, config, "annotate"
        )
    assert sorted(started) == [0, 1, 2, 3]
    failure = json.loads((tmp_path / "failure-annotate.json").read_text())
    assert failure["submitted_tasks"] == 4
    assert failure["unsubmitted_tasks"] == 16
    assert len(failure["running_tasks_at_stop"]) <= 3
    assert failure["issued_requests_withdrawn"] is False
    assert failure["running_tasks_drained"] is True
