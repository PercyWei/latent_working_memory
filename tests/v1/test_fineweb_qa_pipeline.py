"""Offline checks for the frozen FineWeb QA batch stages."""

import copy
import json
import re
import threading

import pytest

from latent_working_memory.data_preparation.fineweb_qa import pipeline
from latent_working_memory.data_preparation.fineweb_qa.annotation import (
    ContentFilteredError,
    AnnotationContractError,
)
from latent_working_memory.data_preparation.fineweb_qa.assembly import validate_trajectory


PROMPT_STAGES = ("generate", "verify", "document_review", "answer", "review", "adjudicate")


def _selection(segment_count=8, split="train", document_index=0):
    parts, segments = [], []
    offset = 0
    for segment_index, count in enumerate([12] + [8] * (segment_count - 2) + [4]):
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
        "source_pool_id": "pool-test",
        "batch_index": 0,
        "source_pool_config": {},
        "ranges": {},
        "documents": [
            {
                "trajectory_id": f"trajectory-{document_index}",
                "document_id": f"document-{document_index}",
                "dedup_cluster": f"example.org/article-{document_index}",
                "split": split,
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
        "source_pool_dir": str(tmp_path / "pool"),
        "batch_index": 0,
        "review": {"qas_per_segment": 1, "seed": 31},
        "qa": {
            "max_answer_chars": 128,
            "role_seed": 29,
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
                "review": 8192,
                "adjudicate": 8192,
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
        if stage in ("review", "adjudicate"):
            return {
                "decisions": [
                    {
                        "qa_id": q["qa_id"],
                        "accepted": True,
                        "reason": "",
                        "same_fact_with": [],
                        "evidence_prediction_correct": True,
                    }
                    for q in content["qas"]
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
    assert len(json.loads((root / "review-inputs/doc-000.json").read_text())["qas"]) == 8
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
            "attempts_total": 1,
            "network_attempts": 1,
            "network_seconds": 2.0,
            "cache_hit": False,
            "ok": False,
            "usage": None,
        },
        {
            "stage": "generate",
            "request_id": "request-1",
            "attempts_total": 1,
            "network_attempts": 0,
            "network_seconds": 0.0,
            "cache_hit": True,
            "ok": True,
            "usage": {"input_tokens": 101, "output_tokens": 42},
        },
        {
            "stage": "generate",
            "request_id": "request-1",
            "attempts_total": 1,
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


class FlaggedReviewClient(FakeAnnotationClient):
    def call(self, stage, content, schema):
        output = super().call(stage, content, schema)
        if stage == "review":
            output["decisions"][0].update(
                accepted=False,
                reason="Check scope and duplicate target",
                same_fact_with=["old-rejected-candidate"],
            )
        return output


def test_review_adjudicates_flags_and_completed_documents_resume(tmp_path, monkeypatch):
    config, root = _prepare_fake(tmp_path, monkeypatch, FlaggedReviewClient)
    pipeline.annotate(config)
    pipeline.diagnose(config)
    summary = pipeline.review(config)
    assert summary["reviewed_qas"] == 8 and summary["rejected_qas"] == 0
    raw = json.loads((root / "reviews/doc-000/raw-review.json").read_text())
    resolved = json.loads((root / "resolved-review.json").read_text())
    assert not raw["decisions"][0]["accepted"]
    assert raw["decisions"][0]["same_fact_with"]
    assert resolved["decisions"][0]["accepted"]
    assert not resolved["decisions"][0]["same_fact_with"]
    adjudication = next(c for stage, c in FlaggedReviewClient.calls if stage == "adjudicate")
    assert len(adjudication["qas"]) == 1
    reviewer = next(c for stage, c in FlaggedReviewClient.calls if stage == "review")
    assert "candidate_decisions" not in reviewer
    assert "candidate_decisions" in adjudication
    calls = len(FlaggedReviewClient.calls)
    pipeline.review(config)
    assert len(FlaggedReviewClient.calls) == calls
    assert pipeline.finalize(config)["summary"]["final_qas"] == 64


class InvalidAdjudicationClient(FlaggedReviewClient):
    def call(self, stage, content, schema):
        result = super().call(stage, content, schema)
        if stage == "adjudicate":
            result["decisions"][0]["same_fact_with"] = ["nonexistent"]
        return result


def test_invalid_adjudication_is_preserved_and_prevents_publication(tmp_path, monkeypatch):
    config, root = _prepare_fake(tmp_path, monkeypatch, InvalidAdjudicationClient)
    pipeline.annotate(config)
    pipeline.diagnose(config)
    with pytest.raises(ValueError, match="unknown or rejected"):
        pipeline.review(config)
    assert (root / "reviews/doc-000/adjudication.json").exists()
    assert (root / "reviews/doc-000/error.json").exists()
    assert not (root / "resolved-review.json").exists()
    with pytest.raises(FileNotFoundError):
        pipeline.finalize(config)
    assert not (tmp_path / "dataset/train.jsonl").exists()


class InterruptedReviewClient(FakeAnnotationClient):
    fail = True

    def call(self, stage, content, schema):
        if stage == "review" and content["trajectory_id"] == "trajectory-1" and self.fail:
            raise RuntimeError("injected review interruption")
        return super().call(stage, content, schema)


def test_review_resumes_per_document_without_repeating_completed_work(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["annotation"]["concurrency"] = 1
    selection = _selection()
    selection["documents"] += _selection(document_index=1)["documents"]
    monkeypatch.setattr(pipeline, "prepare_selection", lambda _: selection)
    monkeypatch.setattr(pipeline, "AnnotationClient", InterruptedReviewClient)
    InterruptedReviewClient.fail = True
    InterruptedReviewClient.calls = []
    pipeline.prepare(config)
    pipeline.annotate(config)
    pipeline.diagnose(config)
    with pytest.raises(RuntimeError, match="review interruption"):
        pipeline.review(config)
    root = tmp_path / "artifacts"
    assert (root / "reviews/doc-000/resolved.json").exists()
    assert not (root / "resolved-review.json").exists()
    InterruptedReviewClient.fail = False
    assert pipeline.review(config)["reviewed_qas"] == 16
    calls = [c["trajectory_id"] for stage, c in InterruptedReviewClient.calls if stage == "review"]
    assert calls == ["trajectory-0", "trajectory-1"]
    assert not (root / "reviews/doc-001/error.json").exists()


class InterruptedSupplementReviewClient(DelayedFactsClient):
    fail = True

    def call(self, stage, content, schema):
        if stage == "review" and content["phase"] == "finalize_review" and self.fail:
            raise RuntimeError("injected supplement review interruption")
        return super().call(stage, content, schema)


def test_final_supplement_review_resume_preserves_round_budget_and_fixed_panel(
    tmp_path, monkeypatch
):
    InterruptedSupplementReviewClient.fail = True
    config, root = _prepare_fake(tmp_path, monkeypatch, InterruptedSupplementReviewClient)
    pipeline.annotate(config)
    pipeline.diagnose(config)
    panel_bytes = (root / "diagnostic-panel.json").read_bytes()
    panel = json.loads(panel_bytes)
    rejected = next(q["qa_id"] for q in panel if q["segment_id"] == "seg0")
    _write_resolved_review(root, panel, rejected_id=rejected)
    with pytest.raises(RuntimeError, match="supplement review interruption"):
        pipeline.finalize(config)
    state = json.loads((root / "reviewed-documents/doc-000.json").read_text())
    assert len(state["rounds"]) == 4 and state["pending_review"] and not state["finished"]
    generated = sum(stage == "generate" for stage, _ in InterruptedSupplementReviewClient.calls)
    InterruptedSupplementReviewClient.fail = False
    final = pipeline.finalize(config)
    assert final["summary"]["final_qas"] == 64
    assert final["summary"]["supplement_reviewed_qas"] == 1
    assert (
        sum(stage == "generate" for stage, _ in InterruptedSupplementReviewClient.calls)
        == generated
    )
    state = json.loads((root / "reviewed-documents/doc-000.json").read_text())
    assert not state["pending_review"] and state["finished"]
    assert len(state["rounds"]) == 4
    assert (root / "diagnostic-panel.json").read_bytes() == panel_bytes


class RejectSupplementClient(DelayedFactsClient):
    def call(self, stage, content, schema):
        result = super().call(stage, content, schema)
        if stage in ("review", "adjudicate") and content["phase"] == "finalize_review":
            for d in result["decisions"]:
                d.update(accepted=False, reason="Unsupported repaired candidate")
        return result


def test_supplement_review_rejections_cannot_escape_the_shared_budget(tmp_path, monkeypatch):
    config, root = _prepare_fake(tmp_path, monkeypatch, RejectSupplementClient)
    pipeline.annotate(config)
    pipeline.diagnose(config)
    panel = json.loads((root / "diagnostic-panel.json").read_text())
    _write_resolved_review(root, panel, rejected_id=panel[0]["qa_id"])
    final = pipeline.finalize(config)
    assert final["summary"]["final_qas"] == 0
    doc = json.loads((root / "final-documents/doc-000.json").read_text())
    assert doc["supplement_rounds_used"] == 3 and doc["supplement_reviewed_qas"] == 1
    assert (tmp_path / "dataset/train.jsonl").read_text() == ""


def test_complete_variable_length_batch_writes_all_three_splits(tmp_path, monkeypatch):
    config = _config(tmp_path)
    selection = _selection(6, "train", 0)
    selection["documents"] += (
        _selection(8, "dev", 1)["documents"] + _selection(10, "test", 2)["documents"]
    )
    monkeypatch.setattr(pipeline, "prepare_selection", lambda _: selection)
    monkeypatch.setattr(pipeline, "AnnotationClient", FakeAnnotationClient)
    FakeAnnotationClient.calls = []
    pipeline.prepare(config)
    pipeline.annotate(config)
    assert pipeline.diagnose(config)["panel_count"] == 24
    assert pipeline.review(config)["reviewed_qas"] == 24
    result = pipeline.finalize(config)
    assert result["summary"]["final_qas"] == 192
    for split, n in [("train", 6), ("dev", 8), ("test", 10)]:
        records = [
            json.loads(x) for x in (tmp_path / f"dataset/{split}.jsonl").read_text().splitlines()
        ]
        assert len(records) == 1 and records[0]["split"] == split
        assert len(records[0]["qas"]) == n * 8
        expected_role = "train" if split == "train" else "evaluation"
        assert {q["role"] for q in records[0]["qas"]} == {expected_role, "gate"}
        assert result["summary"]["by_split"][split]["qas"] == n * 8
    assert set(result["summary"]["by_segment_count"]) == {"6", "8", "10"}


def test_valid_manual_resolution_is_allowed_before_but_not_after_finalization(
    tmp_path, monkeypatch
):
    config, root = _prepare_fake(tmp_path, monkeypatch, FakeAnnotationClient)
    pipeline.annotate(config)
    pipeline.diagnose(config)
    pipeline.review(config)
    path = root / "reviews/doc-000/resolved.json"
    resolved = json.loads(path.read_text())
    resolved["decisions"][0]["evidence_prediction_correct"] = False
    resolved["decisions"][0]["reason"] = "Manual semantic correction before finalization"
    path.write_text(json.dumps(resolved))
    calls = len(FakeAnnotationClient.calls)
    assert pipeline.review(config)["evidence_semantic_correct"] == 7
    assert len(FakeAnnotationClient.calls) == calls
    pipeline.finalize(config)
    resolved["decisions"][0]["evidence_prediction_correct"] = True
    path.write_text(json.dumps(resolved))
    with pytest.raises(ValueError, match="after finalization started"):
        pipeline.review(config)


@pytest.mark.parametrize("error_type", [ContentFilteredError, AnnotationContractError])
@pytest.mark.parametrize("filtered_stage", PROMPT_STAGES)
def test_filtered_document_is_excluded_across_stages_and_resume(
    tmp_path, monkeypatch, filtered_stage, error_type
):
    config = _config(tmp_path)
    config["annotation"]["concurrency"] = 1
    selection = _selection(6)
    selection["documents"] += [
        _selection(6, split, i)["documents"][0] for i, split in [(1, "dev"), (2, "test")]
    ]
    original = FakeAnnotationClient.call
    blocked = []

    def call(self, stage, content, schema):
        if stage == filtered_stage and not blocked:
            blocked.append(stage)
            raise error_type(stage, "filtered-request", "cache/response.raw.json")
        result = original(self, stage, content, schema)
        if filtered_stage == "adjudicate" and stage == "review":
            result["decisions"][0]["reason"] = "Check this proposal"
        return result

    FakeAnnotationClient.calls = []
    monkeypatch.setattr(pipeline, "prepare_selection", lambda _: selection)
    monkeypatch.setattr(pipeline, "AnnotationClient", FakeAnnotationClient)
    monkeypatch.setattr(FakeAnnotationClient, "call", call)
    pipeline.prepare(config)
    pipeline.annotate(config)
    diagnosis = pipeline.diagnose(config)
    review = pipeline.review(config)
    final = pipeline.finalize(config)
    root = tmp_path / "artifacts"
    failure = json.loads((root / "failed-documents/doc-000.json").read_text())
    assert failure["reason"] == error_type.reason and failure["stage"] == filtered_stage
    assert failure["request_id"] == "filtered-request"
    assert final["summary"]["frozen_documents"] == 3
    assert final["summary"][f"{error_type.reason}_documents"] == 1
    assert final["summary"]["complete_quota_documents_after_review"] == 2
    assert final["summary"]["quota_failed_documents"] == 0
    assert final["summary"]["trajectory_success_rate"] == 2 / 3
    assert final["summary"]["by_split"]["train"][f"{error_type.reason}_documents"] == 1
    assert (tmp_path / "dataset/train.jsonl").read_text() == ""
    assert final["summary"]["final_qas"] == 96
    assert (
        len(json.loads((tmp_path / "dataset/preparation.json").read_text())["document_failures"])
        == 1
    )
    if filtered_stage == "answer":
        assert diagnosis["panel_count"] == 18 and diagnosis["scored_panel_count"] == 12
        assert (
            len(json.loads((root / "diagnostics.json").read_text())["excluded_panel_qa_ids"]) == 6
        )
    if filtered_stage in ("review", "adjudicate"):
        assert review["excluded_panel_qas"] == 6 and review["reviewed_qas"] == 12
    before = len(FakeAnnotationClient.calls)
    for stage in (pipeline.annotate, pipeline.diagnose, pipeline.review, pipeline.finalize):
        stage(config)
    assert len(FakeAnnotationClient.calls) == before
    assert json.loads((root / "failed-documents/doc-000.json").read_text()) == failure


@pytest.mark.parametrize("error_type", [ContentFilteredError, AnnotationContractError])
@pytest.mark.parametrize("filtered_stage", PROMPT_STAGES)
def test_content_filter_during_final_supplement_excludes_whole_document(
    tmp_path, monkeypatch, filtered_stage, error_type
):
    config = _config(tmp_path)
    config["annotation"]["concurrency"] = 1
    selection = _selection(6)
    selection["documents"] += _selection(8, "test", 1)["documents"]
    FakeAnnotationClient.calls = []
    monkeypatch.setattr(pipeline, "prepare_selection", lambda _: selection)
    monkeypatch.setattr(pipeline, "AnnotationClient", FakeAnnotationClient)
    pipeline.prepare(config)
    pipeline.annotate(config)
    pipeline.diagnose(config)
    root = tmp_path / "artifacts"
    panel = json.loads((root / "diagnostic-panel.json").read_text())
    _write_resolved_review(root, panel, panel[0]["qa_id"])
    original = FakeAnnotationClient.call
    blocked = []

    def call(self, stage, content, schema):
        if stage == filtered_stage and not blocked:
            blocked.append(stage)
            raise error_type(stage, "supplement-filtered", "cache/response.raw.json")
        result = original(self, stage, content, schema)
        if filtered_stage == "adjudicate" and stage == "review":
            result["decisions"][0]["reason"] = "Check this proposal"
        return result

    monkeypatch.setattr(FakeAnnotationClient, "call", call)
    final = pipeline.finalize(config)
    assert final["summary"][f"{error_type.reason}_documents"] == 1
    assert final["summary"]["complete_quota_documents_after_review"] == 1
    assert final["summary"]["final_qas"] == 64
    failure = json.loads((root / "failed-documents/doc-000.json").read_text())
    assert failure["phase"] == "finalize" and failure["stage"] == filtered_stage
    assert (tmp_path / "dataset/train.jsonl").read_text() == ""
    before = len(FakeAnnotationClient.calls)
    assert pipeline.finalize(config)["summary"] == final["summary"]
    assert len(FakeAnnotationClient.calls) == before


def test_all_documents_filtered_produces_auditable_empty_dataset(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(pipeline, "prepare_selection", lambda _: _selection())
    monkeypatch.setattr(pipeline, "AnnotationClient", FakeAnnotationClient)

    def filtered(self, stage, content, schema):
        raise ContentFilteredError(stage, "all-filtered", "cache/response.raw.json")

    monkeypatch.setattr(FakeAnnotationClient, "call", filtered)
    pipeline.prepare(config)
    assert pipeline.annotate(config)["content_filtered_documents"] == 1
    assert pipeline.diagnose(config)["scored_panel_count"] == 0
    assert pipeline.review(config)["reviewed_qas"] == 0
    final = pipeline.finalize(config)
    assert final["summary"]["trajectory_success_rate"] == 0
    assert final["summary"]["content_filtered_documents"] == 1
    assert final["summary"]["quota_failed_documents"] == 0
    assert all(
        (tmp_path / "dataset" / f"{split}.jsonl").read_text() == ""
        for split in ("train", "dev", "test")
    )


def test_annotation_filter_after_saved_round_preserves_history_but_excludes_qa(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    config["annotation"]["concurrency"] = 1
    selection = _selection()
    original = FakeAnnotationClient.call

    def call(self, stage, content, schema):
        if stage == "generate" and content["round_index"] == 1:
            raise ContentFilteredError(stage, "round-one-filter", "cache/response.raw.json")
        result = original(self, stage, content, schema)
        if stage == "generate" and content["segment_id"] == "seg0":
            result["qas"] = result["qas"][:-1]
        return result

    monkeypatch.setattr(pipeline, "prepare_selection", lambda _: selection)
    monkeypatch.setattr(pipeline, "AnnotationClient", FakeAnnotationClient)
    monkeypatch.setattr(FakeAnnotationClient, "call", call)
    pipeline.prepare(config)
    summary = pipeline.annotate(config)
    assert summary["content_filtered_documents"] == 1 and summary["incomplete_documents"] == 0
    saved = json.loads((tmp_path / "artifacts/documents/doc-000.json").read_text())
    assert len(saved["rounds"]) == 1 and not saved["finished"]
    assert saved["candidates"]
    assert pipeline.diagnose(config)["panel_count"] == 0
    pipeline.review(config)
    final = pipeline.finalize(config)
    assert final["summary"]["annotated_documents"] == 0
    assert final["summary"]["content_filtered_documents"] == 1
    assert final["summary"]["final_qas"] == 0
