"""Offline checks for the frozen FineWeb QA batch stages."""

import copy
import json
import re
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latent_working_memory.data_preparation.fineweb_factqa import pipeline, campaign
from latent_working_memory.data_preparation.fineweb_factqa.storage import load_json, save_json
from latent_working_memory.data_preparation.fineweb_factqa.annotation import (
    ContentFilteredError,
    AnnotationContractError,
)
from latent_working_memory.data_preparation.fineweb_factqa.assembly import validate_trajectory
from latent_working_memory.data_preparation.pretrain.dedup import source_key
from latent_working_memory.data_preparation.pretrain.fineweb import document_split


PROMPT_STAGES = ("generate", "verify", "document_review")


def _selection(segment_count=8, split="train", document_index=0, answer_suffix=""):
    parts, segments = [], []
    offset = 0
    for segment_index, count in enumerate([12] + [8] * (segment_count - 2) + [4]):
        segment_text = "".join(
            f"Fact-{segment_index}-{qa_index} has answer-{segment_index}-{qa_index}{answer_suffix}.\n"
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
        "statistics": {"scanned_documents": 1, "frozen_trajectories": 1},
    }


def _config(tmp_path):
    prompt_paths = {}
    for stage in PROMPT_STAGES:
        path = tmp_path / f"{stage}.txt"
        text = f"Pilot {stage} instructions.\n"
        if stage == "generate":
            text += "Answer limit: {{max_answer_chars}} characters.\n"
        path.write_text(text, encoding="utf-8")
        prompt_paths[stage] = str(path)
    return {
        "source_pool_dir": str(tmp_path / "pool"),
        "batch_index": 0,
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
        raise AssertionError(f"unexpected stage: {stage}")


def _prepare_fake(
    tmp_path, monkeypatch, client_type=FakeAnnotationClient, segments=8, split="train"
):
    config = _config(tmp_path)
    client_type.calls = []
    monkeypatch.setattr(pipeline, "AnnotationClient", client_type)
    monkeypatch.setattr(pipeline, "prepare_selection", lambda _: _selection(segments, split))
    pipeline.prepare(config)
    return config, tmp_path / "artifacts"


def test_annotation_and_qa_ids_distinguish_windows_of_the_same_source(tmp_path, monkeypatch):
    config = _config(tmp_path)
    selection = _selection(segment_count=6)
    first = selection["documents"][0]
    second = copy.deepcopy(first)
    start = first["window_char_span"][1]
    second["window_char_span"] = [start, start + len(second["text"])]
    second["trajectory_id"] = "trajectory-1"
    selection["documents"].append(second)
    FakeAnnotationClient.calls = []
    monkeypatch.setattr(pipeline, "AnnotationClient", FakeAnnotationClient)
    monkeypatch.setattr(pipeline, "prepare_selection", lambda _: selection)
    pipeline.prepare(config)
    assert pipeline.annotate(config)["complete_quota_trajectories"] == 2
    final = pipeline.finalize(config)
    summary = final["summary"]
    assert summary["frozen_trajectories"] == summary["complete_trajectories"] == 2
    assert summary["frozen_source_documents"] == summary["complete_source_documents"] == 1
    rows = [
        json.loads(line) for line in (tmp_path / "dataset/train.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert {row["document_id"] for row in rows} == {first["document_id"]}
    ids = [{qa["qa_id"] for qa in row["qas"]} for row in rows]
    assert ids[0].isdisjoint(ids[1])
    for row in rows:
        validate_trajectory(row, config["qa"])
    before = len(FakeAnnotationClient.calls)
    pipeline.annotate(config)
    assert len(FakeAnnotationClient.calls) == before


@pytest.mark.parametrize("segments", [6, 7, 8, 9, 10])
@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_three_model_stages_cover_every_candidate_and_finalize_without_requests(
    tmp_path, monkeypatch, segments, split
):
    config, root = _prepare_fake(tmp_path, monkeypatch, segments=segments, split=split)
    result = pipeline.annotate(config)
    assert result["complete_quota_trajectories"] == 1
    calls = FakeAnnotationClient.calls
    assert [stage for stage, _ in calls] == [
        s for _ in range(segments) for s in ("generate", "verify")
    ] + ["document_review"]
    verified_ids = {
        q["qa_id"] for stage, content in calls if stage == "verify" for q in content["qas"]
    }
    whole = calls[-1][1]
    assert {q["qa_id"] for q in whole["qas"]} == verified_ids
    assert len(whole["segments"]) == segments
    assert whole["text"] == _selection(segments, split)["documents"][0]["text"]
    before = len(calls)
    monkeypatch.setattr(
        pipeline, "AnnotationClient", lambda *_: pytest.fail("finalization must be local")
    )
    final = pipeline.finalize(config)
    assert final["summary"]["complete_trajectories"] == 1
    assert final["summary"]["final_qas"] == 8 * segments
    rows = [
        json.loads(line)
        for line in (tmp_path / "dataset" / f"{split}.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    validate_trajectory(rows[0], config["qa"])
    assert {q["role"] for q in rows[0]["qas"]} == {
        "train" if split == "train" else "evaluation",
        "gate",
    }
    first = (tmp_path / "dataset" / f"{split}.jsonl").read_bytes()
    pipeline.finalize(config)
    assert (tmp_path / "dataset" / f"{split}.jsonl").read_bytes() == first
    assert len(calls) == before
    metadata = json.loads((tmp_path / "dataset/preparation.json").read_text())
    assert set(metadata["prompts"]) == set(PROMPT_STAGES)
    assert "diagnostics" not in metadata and "review" not in metadata
    assert set(p.name for p in root.iterdir()) == {
        "config.json",
        "prompts.json",
        "selection.json",
        "documents",
        "annotate-summary.json",
        "final-documents",
    }


class RejectedFirstQuestionClient(FakeAnnotationClient):
    rejected_stage = "verify"

    def call(self, stage, content, schema):
        response = super().call(stage, content, schema)
        if stage == self.rejected_stage and content["round_index"] == 0:
            for decision in response["decisions"]:
                if decision["qa_id"].endswith(":seg0:round0:qa0"):
                    decision.update(accepted=False, reason="Question needs a clear time scope")
        return response


@pytest.mark.parametrize("rejected_stage", ["verify", "document_review"])
def test_rejected_candidates_are_replaced_through_both_checks(
    tmp_path, monkeypatch, rejected_stage
):
    RejectedFirstQuestionClient.rejected_stage = rejected_stage
    config, root = _prepare_fake(tmp_path, monkeypatch, RejectedFirstQuestionClient)
    assert pipeline.annotate(config)["complete_quota_trajectories"] == 1
    saved = json.loads((root / "documents/doc-000.json").read_text())
    assert len(saved["rounds"]) == 2
    calls = RejectedFirstQuestionClient.calls
    supplement = [c for stage, c in calls if stage == "generate" and c["round_index"] == 1]
    assert len(supplement) == 1 and supplement[0]["segment_id"] == "seg0"
    assert supplement[0]["candidate_limit"] == 3
    assert supplement[0]["rejected_candidates"]
    assert [s for s, c in calls if c["round_index"] == 1] == [
        "generate",
        "verify",
        "document_review",
    ]
    first_id = "trajectory-0:seg0:round0:qa0"
    final = pipeline.finalize(config)
    assert final["summary"]["final_qas"] == 64
    row = json.loads((tmp_path / "dataset/train.jsonl").read_text())
    assert first_id not in {q["qa_id"] for q in row["qas"]}
    if rejected_stage == "document_review":
        assert (
            next(d for d in saved["review_decisions"] if d["qa_id"] == first_id)["accepted"]
            is False
        )
    before = len(calls)
    pipeline.annotate(config)
    pipeline.finalize(config)
    assert len(calls) == before


class NoMoreFactsClient(FakeAnnotationClient):
    def call(self, stage, content, schema):
        response = super().call(stage, content, schema)
        if stage == "generate":
            if content["round_index"] > 0:
                return {"qas": [], "skip_reason": "No additional supported facts"}
            if content["segment_id"] == "seg0":
                response["qas"].pop()
        return response


def test_three_supplement_rounds_exhaust_without_publishing_partial_trajectory(
    tmp_path, monkeypatch
):
    config, root = _prepare_fake(tmp_path, monkeypatch, NoMoreFactsClient)
    assert pipeline.annotate(config)["complete_quota_trajectories"] == 0
    result = json.loads((root / "documents/doc-000.json").read_text())
    assert len(result["rounds"]) == 4
    for number in (1, 2, 3):
        assert [s for s, c in NoMoreFactsClient.calls if c["round_index"] == number] == [
            "generate",
            "document_review",
        ]
    final = pipeline.finalize(config)
    assert final["summary"]["quota_failed_trajectories"] == 1
    assert final["summary"]["final_qas"] == 0
    assert (tmp_path / "dataset/train.jsonl").read_text() == ""
    failure = json.loads((root / "final-documents/doc-000.json").read_text())
    assert failure["failure_reason"] == "quota_shortfall" and failure["supplement_rounds_used"] == 3


class InterruptedClient(RejectedFirstQuestionClient):
    fail = True
    rejected_stage = "verify"

    def call(self, stage, content, schema):
        if self.fail and content["round_index"] == 1 and stage == "generate":
            raise RuntimeError("injected interruption")
        return super().call(stage, content, schema)


def test_resume_preserves_completed_rounds_and_finalization_requires_completion(
    tmp_path, monkeypatch
):
    InterruptedClient.fail = True
    config, root = _prepare_fake(tmp_path, monkeypatch, InterruptedClient)
    with pytest.raises(FileNotFoundError, match="has not been annotated"):
        pipeline.finalize(config)
    with pytest.raises(RuntimeError, match="injected interruption"):
        pipeline.annotate(config)
    assert len(json.loads((root / "documents/doc-000.json").read_text())["rounds"]) == 1
    with pytest.raises(ValueError, match="annotation is incomplete"):
        pipeline.finalize(config)
    before = len(InterruptedClient.calls)
    InterruptedClient.fail = False
    pipeline.annotate(config)
    assert len(InterruptedClient.calls) == before + 3
    assert pipeline.finalize(config)["summary"]["complete_trajectories"] == 1


@pytest.mark.parametrize("failed_stage", PROMPT_STAGES)
@pytest.mark.parametrize("error_type", [ContentFilteredError, AnnotationContractError])
def test_document_failure_in_each_model_stage_preserves_other_documents(
    tmp_path, monkeypatch, failed_stage, error_type
):
    class FailedClient(FakeAnnotationClient):
        def call(self, stage, content, schema):
            if content["trajectory_id"] == "trajectory-0" and stage == failed_stage:
                raise error_type(stage, "failed-request", "raw.json", "bad output")
            return super().call(stage, content, schema)

    config = _config(tmp_path)
    documents = [_selection(document_index=i)["documents"][0] for i in range(2)]
    selection = dict(_selection(), documents=documents)
    FailedClient.calls = []
    monkeypatch.setattr(pipeline, "prepare_selection", lambda _: selection)
    monkeypatch.setattr(pipeline, "AnnotationClient", FailedClient)
    pipeline.prepare(config)
    pipeline.annotate(config)
    final = pipeline.finalize(config)
    assert final["summary"]["frozen_trajectories"] == 2
    assert final["summary"][error_type.reason + "_trajectories"] == 1
    assert final["summary"]["complete_trajectories"] == 1
    rows = [
        json.loads(line) for line in (tmp_path / "dataset/train.jsonl").read_text().splitlines()
    ]
    assert [r["trajectory_id"] for r in rows] == ["trajectory-1"]
    before = len(FailedClient.calls)
    pipeline.annotate(config)
    assert len(FailedClient.calls) == before


def test_batch_configuration_and_prompts_are_frozen(tmp_path, monkeypatch):
    config, _ = _prepare_fake(tmp_path, monkeypatch)
    changed = copy.deepcopy(config)
    changed["qa"]["role_seed"] += 1
    with pytest.raises(ValueError, match="configuration changed"):
        pipeline.annotate(changed)
    changed = copy.deepcopy(config)
    changed["qa"]["max_answer_chars"] = 256
    with pytest.raises(ValueError, match="configuration changed"):
        pipeline.annotate(changed)
    Path(config["prompts"]["verify"]).write_text("Changed verifier instructions")
    with pytest.raises(ValueError, match="prompts changed"):
        pipeline.annotate(config)


@pytest.mark.parametrize("limit", [8, 256])
def test_generation_prompt_renders_and_freezes_actual_answer_limit(tmp_path, monkeypatch, limit):
    config = _config(tmp_path)
    config["qa"]["max_answer_chars"] = limit
    source = (
        Path(__file__).resolve().parents[2]
        / "configs/data_preparation/prompts/fineweb_factqa/generate.txt"
    )
    Path(config["prompts"]["generate"]).write_text(source.read_text())
    monkeypatch.setattr(pipeline, "prepare_selection", lambda _: _selection())
    pipeline.prepare(config)
    rendered = load_json(Path(config["artifacts_dir"]) / "prompts.json")["generate"]
    assert f"must be at most {limit} characters" in rendered
    assert "{{max_answer_chars}}" not in rendered
    assert '{"qas": [{"fact_statement": "..."' in rendered
    assert "{{max_answer_chars}}" in Path(config["prompts"]["generate"]).read_text()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "128"])
def test_answer_limit_must_be_positive_integer_before_freezing(tmp_path, monkeypatch, limit):
    config = _config(tmp_path)
    config["qa"]["max_answer_chars"] = limit
    monkeypatch.setattr(
        pipeline, "prepare_selection", lambda _: pytest.fail("invalid config read sources")
    )
    with pytest.raises(ValueError, match="qa.max_answer_chars must be a positive integer"):
        pipeline.prepare(config)
    assert not Path(config["artifacts_dir"]).exists()


@pytest.mark.parametrize("limit,complete", [(192, 1), (8, 0)])
def test_pipeline_uses_answer_limit_for_model_prompt_filtering_and_finalization(
    tmp_path, monkeypatch, limit, complete
):
    class LongAnswerClient(FakeAnnotationClient):
        def __init__(self, annotation_config, batch_root, cache_root, prompts):
            super().__init__(annotation_config, batch_root, cache_root, prompts)
            assert f"Answer limit: {limit} characters." in prompts["generate"]

        def call(self, stage, content, schema):
            if stage != "generate":
                return super().call(stage, content, schema)
            qas = []
            for line in content["segment_text"].splitlines():
                fact, answer = line[:-1].split(" has ")
                qas.append(
                    {
                        "fact_statement": f"{fact} has {answer}",
                        "question": f"What does {fact} have?",
                        "answer": answer,
                        "evidence_quote": line,
                    }
                )
            return {"qas": qas[: content["candidate_limit"]], "skip_reason": ""}

    config = _config(tmp_path)
    config["qa"].update(max_answer_chars=limit, max_supplement_rounds=0)
    monkeypatch.setattr(
        pipeline, "prepare_selection", lambda _: _selection(answer_suffix="x" * 140)
    )
    monkeypatch.setattr(pipeline, "AnnotationClient", LongAnswerClient)
    pipeline.prepare(config)
    summary = pipeline.annotate(config)
    assert summary["complete_quota_trajectories"] == complete
    result = pipeline.finalize(config)
    assert result["summary"]["complete_trajectories"] == complete
    rows = [
        json.loads(line)
        for line in Path(config["dataset_dir"]).joinpath("train.jsonl").read_text().splitlines()
    ]
    assert len(rows) == complete
    if complete:
        assert all(128 < len(qa["answer"]) <= limit for qa in rows[0]["qas"])
    else:
        document = load_json(Path(config["artifacts_dir"]) / "documents/doc-000.json")
        assert document["candidates"] == []
        assert all(
            rejection["reason"] == "answer exceeds character limit"
            for segment in document["rounds"][0]["segments"]
            for rejection in segment["program_rejections"]
        )


def test_generation_prompt_must_include_answer_limit_placeholder(tmp_path):
    config = _config(tmp_path)
    Path(config["prompts"]["generate"]).write_text("Use a fixed answer length.")
    with pytest.raises(ValueError, match="generate prompt requires"):
        pipeline._prompt_texts(config)


def test_campaign_executes_real_batch_pipeline_and_publishes_flat_dataset(tmp_path, monkeypatch):
    template = _config(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    records = []
    for document_index, split in enumerate(("train", "train", "dev", "test")):
        for candidate in range(1000):
            url = f"https://example.org/document-{document_index}/{candidate}"
            if document_split(source_key(url), 37, (1 / 3, 1 / 3, 1 / 3)) == split:
                break
        else:
            raise AssertionError("fixture split not found")
        parts = []
        for segment_index, count in enumerate((12, 8, 8, 8, 8, 4)):
            fact = 10 * document_index + segment_index
            text = "".join(
                f"Fact-{fact}-{index} has answer-{fact}-{index}.\n" for index in range(count)
            )
            assert len(text) < 512
            parts.append(text.ljust(512))
        records.append({"id": f"document-{document_index}", "url": url, "text": "".join(parts)})
    pq.write_table(pa.Table.from_pylist(records), raw / "sample.parquet")
    main = {
        "source_dir": str(raw),
        "source_batch_size": 4,
        "source_seed": 37,
        "selection_seed": 31,
        "split_counts": {"train": 1, "dev": 1, "test": 1},
        "window": {
            "capacity": 128,
            "min_segment_ratio": 1,
            "max_segment_ratio": 1,
            "min_segments": 6,
            "max_segments": 6,
            "content_reserve_ratio": 1,
        },
        "batch_split_counts": {"train": 2, "dev": 2, "test": 2},
        "qa": template["qa"],
        "annotation": template["annotation"],
        "prompts_dir": str(tmp_path),
    }
    path = tmp_path / "main.json"
    save_json(path, main)
    config = campaign.run_config(
        campaign.load_config(path), tmp_path / "datasets", tmp_path / "runs", "offline"
    )
    data, run = Path(config["dataset_dir"]), Path(config["artifacts_dir"])

    class ParquetAnnotationClient(FakeAnnotationClient):
        calls = []

        def call(self, stage, content, schema):
            if stage != "generate":
                return super().call(stage, content, schema)
            with self.lock:
                self.calls.append((stage, copy.deepcopy(content)))
            qas = []
            for match in re.finditer(
                r"(Fact-\d+-\d+) has (answer-\d+-\d+)\.", content["segment_text"]
            ):
                qas.append(
                    {
                        "fact_statement": f"{match[1]} has {match[2]}",
                        "question": f"What does {match[1]} have?",
                        "answer": match[2],
                        "evidence_quote": match[0],
                    }
                )
            return {"qas": qas[: content["candidate_limit"]], "skip_reason": ""}

    monkeypatch.setattr(pipeline, "AnnotationClient", ParquetAnnotationClient)
    executed = []

    class Process:
        pid = 100

        def __init__(self, command, **kwargs):
            self.stage, self.config = command[4], load_json(Path(command[-1]))
            for key in ("qa", "annotation"):
                assert self.config[key] == config[key]
            assert self.config["prompts"] == template["prompts"]
            assert self.config["source_pool_dir"] == str(data)
            assert self.config["dataset_dir"] == str(run / "batches/batch-000/dataset")
            assert self.config["cache_dir"] == str(run / "requests")

        def wait(self):
            executed.append(self.stage)
            getattr(pipeline, self.stage)(self.config)
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", Process)
    result = campaign.run(config)
    assert executed == ["prepare", "annotate", "finalize"]
    assert result["dataset_published"]
    assert result["completed_by_split"] == {
        split: {"trajectories": 1, "qas": 48} for split in ("train", "dev", "test")
    }
    assert {p.name for p in data.iterdir()} == {
        "source-pool.json",
        "used-sources.jsonl",
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "preparation.json",
    }
    pool = load_json(data / "source-pool.json")
    assert pool["statistics"]["scanned_documents"] == len(records)
    assert len(pool["documents"]) == 4
    assert all(bounds["selected"] == 1 for bounds in pool["batches"][0]["ranges"].values())
    rows = [load_json(data / f"{split}.jsonl") for split in ("train", "dev", "test")]
    for row in rows:
        original = next(record for record in records if record["id"] == row["document_id"])
        assert row["text"] == original["text"]
        assert row["window_char_span"] == [0, len(original["text"])]
        validate_trajectory(row, config["qa"])
    metadata = load_json(data / "preparation.json")
    assert set(metadata["prompts"]) == {"generate", "verify", "document_review"}
    assert "review" not in metadata and "diagnostics" not in metadata
    assert "used_sources" not in metadata
    assert metadata["used_sources_file"] == "used-sources.jsonl"
    ledger = [
        json.loads(line) for line in (data / metadata["used_sources_file"]).read_text().splitlines()
    ]
    assert len(ledger) == 3
    assert {entry["document_id"] for entry in ledger} == {row["document_id"] for row in rows}
    assert all(set(entry) == {"document_id", "dedup_cluster", "source"} for entry in ledger)
    before = len(ParquetAnnotationClient.calls)
    campaign.run(config)
    assert len(ParquetAnnotationClient.calls) == before
