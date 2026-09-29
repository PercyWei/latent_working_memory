"""Offline checks for the frozen FineWeb QA pilot stages."""

import copy
import json
import re
import threading

import pytest

from latent_working_memory.data_preparation.fineweb_qa import pipeline
from latent_working_memory.data_preparation.fineweb_qa.assembly import validate_trajectory


SEGMENT_QA_COUNTS = [12, 8, 8, 8, 8, 8, 8, 4]
PROMPT_STAGES = ("generate", "verify", "document_review", "answer", "subagent_review")


def _selection():
    parts, segments = [], []
    offset = 0
    for segment_index, count in enumerate(SEGMENT_QA_COUNTS):
        segment_text = "".join(
            f"Fact-{segment_index}-{qa_index} has answer-{segment_index}-{qa_index}.\n"
            for qa_index in range(count)
        )
        parts.append(segment_text)
        segments.append(
            {
                "segment_id": f"seg{segment_index}",
                "char_span": [offset, offset + len(segment_text)],
            }
        )
        offset += len(segment_text)
    text = "".join(parts)
    return {
        "documents": [
            {
                "trajectory_id": "trajectory-0",
                "document_id": "document-0",
                "dedup_cluster": "example.org/article",
                "split": "train",
                "source": {"file": "sample/000_00000.parquet", "row_group": 3, "row_index": 5},
                "window_char_span": [100, 100 + len(text)],
                "text": text,
                "segments": segments,
            }
        ],
        "statistics": {"scanned_documents": 1, "frozen_documents": 1},
    }


def _config(tmp_path):
    prompt_paths = {}
    for stage in PROMPT_STAGES:
        path = tmp_path / f"{stage}.txt"
        path.write_text(f"Pilot {stage} instructions.\n", encoding="utf-8")
        prompt_paths[stage] = str(path)
    return {
        "source": {"dataset": "sample-10BT", "source_seed": 17},
        "window": {"segment_count": 8},
        "qa": {
            "candidate_per_segment": [15, 10, 10, 10, 10, 10, 10, 5],
            "max_answer_chars": 128,
            "train_per_segment": [4] * 8,
            "gate_per_segment": [8, 4, 4, 4, 4, 4, 4, 0],
            "role_seed": 29,
            "review_sample": 8,
            "review_seed": 31,
            "max_supplement_rounds": 3,
            "supplement_surplus": 2,
        },
        "annotation": {
            "endpoint": "http://127.0.0.1:4141/v1/responses",
            "model": "gpt-6-sol",
            "reasoning_effort": "medium",
            "timeout_seconds": 30,
            "max_attempts": 3,
            "concurrency": 4,
            "max_output_tokens": {
                "generate": 8192,
                "verify": 4096,
                "document_review": 8192,
                "answer": 2048,
            },
        },
        "prompts": prompt_paths,
        "artifacts_dir": str(tmp_path / "artifacts"),
        "dataset_dir": str(tmp_path / "dataset"),
        "cache_dir": str(tmp_path / "cache"),
    }


class FakeAnnotationClient:
    calls = []
    lock = threading.Lock()

    def __init__(self, annotation_config, batch_root, cache_root, prompts):
        self.stop = threading.Event()
        self.batch_root = batch_root

    def call(self, stage, content, schema):
        with self.lock:
            self.calls.append((stage, copy.deepcopy(content)))
        if stage == "generate":
            qas = []
            for line in content["segment_text"].splitlines():
                match = re.fullmatch(r"Fact-(\d+)-(\d+) has (answer-\d+-\d+)\.", line)
                assert match is not None
                fact = f"Fact-{match[1]}-{match[2]}"
                qas.append(
                    {
                        "fact_statement": f"{fact} has {match[3]}",
                        "question": f"What does {fact} have?",
                        "answer": match[3],
                        "evidence_quote": line,
                    }
                )
            qas = [qa for qa in qas if qa["fact_statement"] not in content["existing_facts"]]
            qas = qas[: content["candidate_limit"]]
            return {"qas": qas, "skip_reason": ""}
        if stage == "verify":
            return {
                "decisions": [
                    {"qa_id": qa["qa_id"], "accepted": True, "reason": ""} for qa in content["qas"]
                ]
            }
        if stage == "document_review":
            return {
                "decisions": [
                    {
                        "qa_id": qa["qa_id"],
                        "accepted": True,
                        "reason": "",
                        "fact_group_id": qa["qa_id"],
                    }
                    for qa in content["qas"]
                ]
            }
        if stage == "answer":
            if "evidence" not in content:
                return {"answer": "unknown"}
            match = re.search(r"answer-\d+-\d+", content["evidence"])
            assert match is not None
            return {"answer": match[0]}
        raise AssertionError(f"unexpected stage: {stage}")


def _write_resolved_review(root, panel, rejected_id=None):
    decisions = []
    for qa in panel:
        accepted = qa["qa_id"] != rejected_id
        decisions.append(
            {
                "qa_id": qa["qa_id"],
                "accepted": accepted,
                "reason": "" if accepted else "Evidence does not fully support the answer",
                "same_fact_with": [],
                "evidence_prediction_correct": True,
            }
        )
    (root / "resolved-review.json").write_text(
        json.dumps({"decisions": decisions}), encoding="utf-8"
    )


def test_offline_pipeline_resumes_and_finalizes_with_fixed_evidence_schedule(tmp_path, monkeypatch):
    config = _config(tmp_path)
    prepared_calls = []

    def select_once(actual_config):
        prepared_calls.append(actual_config)
        return _selection()

    FakeAnnotationClient.calls = []
    monkeypatch.setattr(pipeline, "prepare_selection", select_once)
    monkeypatch.setattr(pipeline, "AnnotationClient", FakeAnnotationClient)

    prepared = pipeline.prepare(config)
    root = tmp_path / "artifacts"
    assert prepared["frozen_documents"] == 1
    assert json.loads((root / "config.json").read_text()) == config
    assert json.loads((root / "prompts.json").read_text())["generate"].startswith("Pilot")
    pipeline.prepare(config)
    assert len(prepared_calls) == 1

    first = pipeline.annotate(config)
    assert first["annotated_documents"] == first["complete_quota_documents"] == 1
    assert first["generated_candidates"] == 64
    assert len(FakeAnnotationClient.calls) == 8 + 8 + 1
    document_result = json.loads((root / "documents" / "doc-000.json").read_text())
    assert document_result["assembly"]["ok"]
    assert len(document_result["local_candidates"]) == 64
    assert pipeline.annotate(config)["complete_quota_documents"] == 1
    assert len(FakeAnnotationClient.calls) == 17

    diagnosis = pipeline.diagnose(config)
    assert diagnosis["panel_count"] == 8
    assert diagnosis["evidence"] == {"em": 1.0, "f1": 1.0}
    assert diagnosis["question_only"] == {"em": 0.0, "f1": 0.0}
    answer_calls = [content for stage, content in FakeAnnotationClient.calls if stage == "answer"]
    assert len(answer_calls) == 16
    assert sum("evidence" in content for content in answer_calls) == 8
    assert sum("evidence" not in content for content in answer_calls) == 8
    pipeline.diagnose(config)
    assert len(FakeAnnotationClient.calls) == 33

    panel = json.loads((root / "diagnostic-panel.json").read_text())
    assert {row["segment_id"] for row in panel} == {f"seg{i}" for i in range(8)}
    assert len(json.loads((root / "review-panel-1.json").read_text())["qas"]) == 4
    assert len(json.loads((root / "review-panel-2.json").read_text())["qas"]) == 4
    _write_resolved_review(root, panel)
    unresolved = json.loads((root / "resolved-review.json").read_text())
    unresolved["decisions"][0]["evidence_prediction_correct"] = None
    (root / "resolved-review.json").write_text(json.dumps(unresolved), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic review decision"):
        pipeline.finalize(config)
    _write_resolved_review(root, panel)
    finalized = pipeline.finalize(config)
    assert finalized["summary"]["complete_quota_documents_after_review"] == 1
    train_lines = (tmp_path / "dataset" / "train.jsonl").read_text().splitlines()
    assert len(train_lines) == 1
    trajectory = json.loads(train_lines[0])
    assert len(trajectory["qas"]) == 64
    assert len(trajectory["usage"][1]["gate_qa_ids"]) == 8
    validate_trajectory(trajectory, config["qa"])
    for qa in trajectory["qas"]:
        evidence = trajectory["text"][slice(*qa["evidence_char_span"])]
        answer = trajectory["text"][slice(*qa["answer_char_span"])]
        assert qa["answer"] == answer in evidence
        assert "evidence_quote" not in qa
    preparation = json.loads((tmp_path / "dataset" / "preparation.json").read_text())
    assert preparation["summary"]["final_qas"] == 64
    assert preparation["stage_counts"]["seg0"]["distinct_facts"] == 12
    assert len(FakeAnnotationClient.calls) == 33

    _write_resolved_review(root, panel, rejected_id=panel[0]["qa_id"])
    with pytest.raises(ValueError, match="resolved review changed"):
        pipeline.finalize(config)


def test_prepare_freezes_config_and_prompt_but_allows_endpoint_change(tmp_path, monkeypatch):
    config = _config(tmp_path)
    prepared_calls = []
    monkeypatch.setattr(
        pipeline,
        "prepare_selection",
        lambda actual_config: (prepared_calls.append(actual_config), _selection())[1],
    )
    pipeline.prepare(config)

    moved_endpoint = copy.deepcopy(config)
    moved_endpoint["annotation"]["endpoint"] = "http://127.0.0.1:5151/v1/responses"
    pipeline.prepare(moved_endpoint)
    assert len(prepared_calls) == 1

    changed_seed = copy.deepcopy(config)
    changed_seed["qa"]["role_seed"] += 1
    with pytest.raises(ValueError, match="configuration changed"):
        pipeline.prepare(changed_seed)

    (tmp_path / "generate.txt").write_text("Revised generation prompt.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompts changed"):
        pipeline.prepare(config)


def test_request_usage_counts_raw_cache_recovery_once(tmp_path):
    root = tmp_path / "pilot"
    root.mkdir()
    records = [
        {
            "stage": "generate",
            "request_id": "request-1",
            "network_attempts": 1,
            "network_seconds": 2.0,
            "cache_hit": False,
            "ok": False,
            "usage": None,
        },
        {
            "stage": "generate",
            "request_id": "request-1",
            "network_attempts": 0,
            "network_seconds": 0.0,
            "cache_hit": True,
            "ok": True,
            "usage": {"input_tokens": 101, "output_tokens": 42},
        },
        {
            "stage": "generate",
            "request_id": "request-1",
            "network_attempts": 0,
            "network_seconds": 0.0,
            "cache_hit": True,
            "ok": True,
            "usage": None,
        },
    ]
    (root / "requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    usage = pipeline._request_statistics(root)["generate"]
    assert usage["new_logical_requests"] == 1
    assert usage["network_attempts"] == 1
    assert usage["cache_hits"] == 2
    assert usage["input_tokens"] == 101
    assert usage["output_tokens"] == 42
    assert usage["usage_missing"] == 0


def _prepare_fake(tmp_path, monkeypatch, client_type):
    config = _config(tmp_path)
    client_type.calls = []
    monkeypatch.setattr(pipeline, "prepare_selection", lambda _: _selection())
    monkeypatch.setattr(pipeline, "AnnotationClient", client_type)
    pipeline.prepare(config)
    return config, tmp_path / "artifacts"


class DelayedFactsClient(FakeAnnotationClient):
    def call(self, stage, content, schema):
        response = super().call(stage, content, schema)
        if stage == "generate" and content["segment_id"] == "seg0":
            if content["round_index"] == 0:
                response["qas"] = response["qas"][:-2]
            elif content["round_index"] == 1:
                # A paraphrase of an existing fact must not fill either missing slot.
                response["qas"] = [
                    {
                        "fact_statement": "Fact-0-0 has answer-0-0",
                        "question": "Which answer is associated with Fact-0-0?",
                        "answer": "answer-0-0",
                        "evidence_quote": "Fact-0-0 has answer-0-0.",
                    }
                ]
        if stage == "document_review":
            for qa, decision in zip(content["qas"], response["decisions"], strict=True):
                decision["fact_group_id"] = qa["fact_statement"]
        return response


def test_topup_targets_only_short_segments_and_deduplicates_across_rounds(tmp_path, monkeypatch):
    config, root = _prepare_fake(tmp_path, monkeypatch, DelayedFactsClient)
    summary = pipeline.annotate(config)
    result = json.loads((root / "documents/doc-000.json").read_text())
    assert summary["complete_quota_documents"] == 1
    assert len(result["rounds"]) == 3
    assert [r["net_eligible_change"] for r in result["rounds"]] == [62, 0, 2]
    assert summary["supplementation"]["initial_complete_documents"] == 0
    assert summary["supplementation"]["latest_complete_documents"] == 1
    assert len({q["qa_id"] for q in result["candidates"]}) == 65
    topups = [
        c for stage, c in DelayedFactsClient.calls if stage == "generate" and c["round_index"] > 0
    ]
    assert [(c["segment_id"], c["candidate_limit"]) for c in topups] == [("seg0", 4)] * 2
    assert all(len(c["existing_facts"]) == 62 for c in topups)
    count = len(DelayedFactsClient.calls)
    pipeline.annotate(config)
    assert len(DelayedFactsClient.calls) == count
    assert pipeline._stage_counts([result])["seg0"]["generated"] == 13


class RepairedCandidateClient(FakeAnnotationClient):
    def call(self, stage, content, schema):
        response = super().call(stage, content, schema)
        if stage == "verify" and content["round_index"] == 0 and content["segment_id"] == "seg0":
            response["decisions"][0].update(accepted=False, reason="Question needs event scope")
        return response


def test_topup_supplies_rejection_feedback_and_allows_a_new_verified_attempt(tmp_path, monkeypatch):
    config, root = _prepare_fake(tmp_path, monkeypatch, RepairedCandidateClient)
    pipeline.annotate(config)
    topup = next(
        c
        for stage, c in RepairedCandidateClient.calls
        if stage == "generate" and c["round_index"] == 1
    )
    assert topup["candidate_limit"] == 3
    assert topup["rejected_candidates"][0]["reason"] == "Question needs event scope"
    result = json.loads((root / "documents/doc-000.json").read_text())
    assert result["assembly"]["ok"]
    assert "trajectory-0:seg0:round0:qa0" not in {q["qa_id"] for q in result["local_candidates"]}
    assert "trajectory-0:seg0:round1:qa0" in {q["qa_id"] for q in result["local_candidates"]}


class NoMoreFactsClient(FakeAnnotationClient):
    def call(self, stage, content, schema):
        response = super().call(stage, content, schema)
        if stage == "generate":
            if content["round_index"] == 0 and content["segment_id"] == "seg0":
                response["qas"] = response["qas"][:-1]
            elif content["round_index"] > 0:
                return {"qas": [], "skip_reason": "No additional supported facts"}
        return response


def test_three_round_budget_is_exhausted_once_across_annotate_and_finalize(tmp_path, monkeypatch):
    config, root = _prepare_fake(tmp_path, monkeypatch, NoMoreFactsClient)
    pipeline.annotate(config)
    result = json.loads((root / "documents/doc-000.json").read_text())
    assert len(result["rounds"]) == 4
    assert not result["assembly"]["ok"]
    assert all(r["segments"][0]["generated_count"] == 0 for r in result["rounds"][1:])
    pipeline.diagnose(config)
    panel = json.loads((root / "diagnostic-panel.json").read_text())
    _write_resolved_review(root, panel)
    before = len(NoMoreFactsClient.calls)
    for _ in range(2):
        pipeline.annotate(config)
        final = pipeline.finalize(config)
    assert len(NoMoreFactsClient.calls) == before
    assert final["summary"]["final_qas"] == 0
    doc = json.loads((root / "final-documents/doc-000.json").read_text())
    assert doc["supplement_rounds_used"] == 3
    assert doc["shortfalls"][0]["missing_total"] == 1


class InterruptedTopupClient(DelayedFactsClient):
    fail = True

    def call(self, stage, content, schema):
        if stage == "generate" and content["round_index"] == 2 and self.fail:
            raise RuntimeError("injected transient interruption")
        return super().call(stage, content, schema)


def test_resume_keeps_completed_rounds_and_the_remaining_budget(tmp_path, monkeypatch):
    InterruptedTopupClient.fail = True
    config, root = _prepare_fake(tmp_path, monkeypatch, InterruptedTopupClient)
    with pytest.raises(RuntimeError, match="injected transient interruption"):
        pipeline.annotate(config)
    checkpoint = json.loads((root / "documents/doc-000.json").read_text())
    assert len(checkpoint["rounds"]) == 2
    with pytest.raises(ValueError, match="annotation is incomplete"):
        pipeline.diagnose(config)
    old_calls = len(InterruptedTopupClient.calls)
    InterruptedTopupClient.fail = False
    assert pipeline.annotate(config)["complete_quota_documents"] == 1
    assert len(InterruptedTopupClient.calls) == old_calls + 3
    resumed = json.loads((root / "documents/doc-000.json").read_text())
    assert resumed["rounds"][:2] == checkpoint["rounds"]
    assert not (root / "documents/doc-000.error.json").exists()


def test_final_review_topup_shares_budget_and_keeps_the_diagnostic_panel(tmp_path, monkeypatch):
    config, root = _prepare_fake(tmp_path, monkeypatch, DelayedFactsClient)
    pipeline.annotate(config)  # Uses supplementary rounds 1 and 2.
    pipeline.diagnose(config)
    original = (root / "documents/doc-000.json").read_bytes()
    panel_bytes = (root / "diagnostic-panel.json").read_bytes()
    panel = json.loads(panel_bytes)
    rejected = next(q["qa_id"] for q in panel if q["segment_id"] == "seg0")
    _write_resolved_review(root, panel, rejected_id=rejected)
    final = pipeline.finalize(config)
    assert final["summary"]["final_qas"] == 64
    reviewed = json.loads((root / "reviewed-documents/doc-000.json").read_text())
    assert len(reviewed["rounds"]) == 4
    assert reviewed["rounds"][-1]["phase"] == "finalize"
    assert rejected not in {q["qa_id"] for q in reviewed["assembly"]["eligible_qas"]}
    assert (root / "documents/doc-000.json").read_bytes() == original
    count = len(DelayedFactsClient.calls)
    pipeline.finalize(config)
    pipeline.diagnose(config)
    assert len(DelayedFactsClient.calls) == count
    assert (root / "diagnostic-panel.json").read_bytes() == panel_bytes


@pytest.mark.parametrize(
    "field,value",
    [("max_supplement_rounds", -1), ("max_supplement_rounds", True), ("supplement_surplus", -1)],
)
def test_invalid_supplement_settings_fail_before_freezing(tmp_path, field, value):
    config = _config(tmp_path)
    config["qa"][field] = value
    with pytest.raises(ValueError, match=field):
        pipeline.prepare(config)
    assert not (tmp_path / "artifacts/config.json").exists()
