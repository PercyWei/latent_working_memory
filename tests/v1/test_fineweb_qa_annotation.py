from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import hashlib
import json
import threading
import time
from urllib.error import HTTPError

import pytest

from latent_working_memory.data_preparation.fineweb_qa import annotation, pipeline


@pytest.fixture
def client_config():
    return {
        "endpoint": "http://127.0.0.1:4141/v1/responses",
        "model": "gpt-6-sol",
        "reasoning_effort": "medium",
        "timeout_seconds": 120,
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
    }


@pytest.fixture
def prompts():
    return {stage: f"Instructions for {stage}" for stage in annotation.STAGES}


def response_bytes(parsed):
    return json.dumps(
        {
            "status": "completed",
            "model": "gpt-6-sol",
            "reasoning": {"effort": "medium"},
            "usage": {"input_tokens": 31, "output_tokens": 7},
            "output": [
                {
                    "type": "message",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": json.dumps(parsed)}],
                }
            ],
        }
    ).encode()


class Response:
    status = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.body


def test_payload_identity_and_shared_cache_exclude_endpoint(
    tmp_path, monkeypatch, client_config, prompts
):
    sent = []

    class Opener:
        def open(self, request, timeout):
            sent.append((request.full_url, json.loads(request.data), timeout, request.data))
            return Response(response_bytes({"answer": "Oslo"}))

    monkeypatch.setattr(annotation, "build_opener", lambda *_args: Opener())
    cache = tmp_path / "cache"
    first = annotation.AnnotationClient(client_config, tmp_path / "batch-a", cache, prompts)
    assert first.call("answer", {"question": "Which city?"}, annotation.ANSWER_SCHEMA) == {
        "answer": "Oslo"
    }
    other_endpoint = dict(client_config, endpoint="http://another-host/v1/responses")
    second = annotation.AnnotationClient(other_endpoint, tmp_path / "batch-b", cache, prompts)
    assert second.call("answer", {"question": "Which city?"}, annotation.ANSWER_SCHEMA) == {
        "answer": "Oslo"
    }
    assert len(sent) == 1
    assert sent[0][1]["model"] == "gpt-6-sol"
    assert sent[0][1]["reasoning"] == {"effort": "medium"}
    assert sent[0][1]["text"]["format"]["schema"] == annotation.ANSWER_SCHEMA
    assert sent[0][1]["store"] is False
    assert sent[0][2] == 120

    changed_prompts = dict(prompts, answer="Different answer instructions")
    third = annotation.AnnotationClient(client_config, tmp_path / "batch-c", cache, changed_prompts)
    third.call("answer", {"question": "Which city?"}, annotation.ANSWER_SCHEMA)
    assert len(sent) == 2
    first_log = json.loads((tmp_path / "batch-a/requests.jsonl").read_text().splitlines()[0])
    second_log = json.loads((tmp_path / "batch-b/requests.jsonl").read_text().splitlines()[0])
    assert first_log["network_attempts"] == 1 and not first_log["cache_hit"]
    assert first_log["usage"] == {"input_tokens": 31, "output_tokens": 7}
    assert first_log["request_id"] == hashlib.sha256(sent[0][3]).hexdigest()
    assert second_log["cache_hit"] and second_log["cache_source"] == "parsed"
    assert second_log["network_attempts"] == 0
    assert first_log["request_id"] == second_log["request_id"]


def test_raw_response_recovers_after_local_parse_write_failure(
    tmp_path, monkeypatch, client_config, prompts
):
    sends = []

    class Opener:
        def open(self, request, timeout):
            sends.append(request)
            return Response(response_bytes({"answer": "Oslo"}))

    monkeypatch.setattr(annotation, "build_opener", lambda *_args: Opener())
    original = annotation._atomic_write_json

    def fail_parsed(path, value):
        if path.name == "parsed.json":
            raise OSError("disk write failed")
        return original(path, value)

    cache = tmp_path / "cache"
    with monkeypatch.context() as patch:
        patch.setattr(annotation, "_atomic_write_json", fail_parsed)
        client = annotation.AnnotationClient(client_config, tmp_path / "batch", cache, prompts)
        with pytest.raises(OSError, match="disk write failed"):
            client.call("answer", {"question": "Which city?"}, annotation.ANSWER_SCHEMA)
        assert client.stop.is_set()
    assert len(sends) == 1
    assert len(list(cache.glob("*/*/response.raw.json"))) == 1
    assert not list(cache.glob("*/*/parsed.json"))

    resumed = annotation.AnnotationClient(client_config, tmp_path / "batch", cache, prompts)
    assert resumed.call("answer", {"question": "Which city?"}, annotation.ANSWER_SCHEMA) == {
        "answer": "Oslo"
    }
    assert len(sends) == 1
    records = [
        json.loads(line) for line in (tmp_path / "batch/requests.jsonl").read_text().splitlines()
    ]
    assert not records[0]["ok"] and records[0]["network_attempts"] == 1
    assert records[1]["ok"] and records[1]["cache_source"] == "raw"
    assert records[1]["network_attempts"] == 0


def test_failed_raw_write_never_triggers_a_second_remote_request(
    tmp_path, monkeypatch, client_config, prompts
):
    sends = []

    class Opener:
        def open(self, request, timeout):
            sends.append(request)
            return Response(response_bytes({"answer": "Oslo"}))

    monkeypatch.setattr(annotation, "build_opener", lambda *_args: Opener())
    original = annotation._atomic_write

    def fail_raw(path, contents):
        if path.name == "response.raw.json":
            raise OSError("raw disk write failed")
        return original(path, contents)

    cache = tmp_path / "cache"
    with monkeypatch.context() as patch:
        patch.setattr(annotation, "_atomic_write", fail_raw)
        client = annotation.AnnotationClient(client_config, tmp_path / "batch-a", cache, prompts)
        with pytest.raises(OSError, match="raw disk write failed"):
            client.call("answer", {"question": "Which city?"}, annotation.ANSWER_SCHEMA)
    resumed = annotation.AnnotationClient(client_config, tmp_path / "batch-b", cache, prompts)
    with pytest.raises(RuntimeError, match="outcome is uncertain"):
        resumed.call("answer", {"question": "Which city?"}, annotation.ANSWER_SCHEMA)
    assert len(sends) == 1


def test_temporary_http_error_retries_but_fatal_http_error_stops(
    tmp_path, monkeypatch, client_config, prompts
):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            if len(calls) == 1:
                raise HTTPError(request.full_url, 429, "limited", {"Retry-After": "0"}, BytesIO())
            if len(calls) == 3:
                raise HTTPError(request.full_url, 400, "bad request", {}, BytesIO())
            return Response(response_bytes({"answer": "Oslo"}))

    monkeypatch.setattr(annotation, "build_opener", lambda *_args: Opener())
    client = annotation.AnnotationClient(
        client_config, tmp_path / "batch", tmp_path / "cache", prompts
    )
    monkeypatch.setattr(client.stop, "wait", lambda _seconds: False)
    assert client.call("answer", {"question": "Which city?"}, annotation.ANSWER_SCHEMA) == {
        "answer": "Oslo"
    }
    with pytest.raises(RuntimeError, match="HTTP 400"):
        client.call("answer", {"question": "Which country?"}, annotation.ANSWER_SCHEMA)
    assert client.stop.is_set()
    with pytest.raises(InterruptedError):
        client.call("answer", {"question": "Another?"}, annotation.ANSWER_SCHEMA)
    assert len(calls) == 3
    records = [
        json.loads(line) for line in (tmp_path / "batch/requests.jsonl").read_text().splitlines()
    ]
    assert records[0]["network_attempts"] == 2
    assert records[0]["attempts_total"] == 2
    assert records[1]["network_attempts"] == 1 and not records[1]["ok"]


def test_invalid_structured_response_exhausts_document_budget_and_keeps_raw(
    tmp_path, monkeypatch, client_config, prompts
):
    sends = []

    class Opener:
        def open(self, request, timeout):
            sends.append(request)
            return Response(response_bytes({"answer": 17}))

    monkeypatch.setattr(annotation, "build_opener", lambda *_args: Opener())
    cache = tmp_path / "cache"
    client = annotation.AnnotationClient(client_config, tmp_path / "batch-a", cache, prompts)
    with pytest.raises(annotation.AnnotationContractError, match="must be a string"):
        client.call("answer", {"question": "Which city?"}, annotation.ANSWER_SCHEMA)
    assert not client.stop.is_set()
    another = annotation.AnnotationClient(client_config, tmp_path / "batch-b", cache, prompts)
    with pytest.raises(annotation.AnnotationContractError, match="must be a string"):
        another.call("answer", {"question": "Which city?"}, annotation.ANSWER_SCHEMA)
    assert len(sends) == 3
    assert len(list(cache.glob("*/*/response.attempt-*.raw.json"))) == 3
    assert not list(cache.glob("*/*/parsed.json"))


def test_parallel_clients_coalesce_identical_request_and_log_both_calls(
    tmp_path, monkeypatch, client_config, prompts
):
    sends = []
    send_lock = threading.Lock()

    class Opener:
        def open(self, request, timeout):
            with send_lock:
                sends.append(request)
            time.sleep(0.05)
            return Response(response_bytes({"answer": "Oslo"}))

    monkeypatch.setattr(annotation, "build_opener", lambda *_args: Opener())
    cache, batch = tmp_path / "cache", tmp_path / "batch"
    clients = [annotation.AnnotationClient(client_config, batch, cache, prompts) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                client.call, "answer", {"question": "Which city?"}, annotation.ANSWER_SCHEMA
            )
            for client in clients
        ]
        assert [future.result() for future in futures] == [{"answer": "Oslo"}] * 2
    assert len(sends) == 1
    records = [json.loads(line) for line in (batch / "requests.jsonl").read_text().splitlines()]
    assert len(records) == 2 and all(row["ok"] for row in records)
    assert sorted(row["cache_hit"] for row in records) == [False, True]


def test_candidate_locator_uses_first_answer_match_and_trajectory_offsets():
    text = "Ada and Ada visited Oslo. Ada and Ada visited Paris."
    generated = {
        "qas": [
            {
                "fact_statement": "Ada visited Oslo.",
                "question": "Which city did Ada visit first?",
                "answer": "Ada",
                "evidence_quote": "Ada and Ada visited Oslo.",
            },
            {
                "fact_statement": "The visit involved Ada.",
                "question": "Who visited a city?",
                "answer": "Ada",
                "evidence_quote": "Ada and Ada",
            },
            {
                "fact_statement": "Ada visited a city.",
                "question": "Which city?",
                "answer": "Berlin",
                "evidence_quote": "Ada and Ada visited Paris.",
            },
        ],
        "skip_reason": "",
    }
    valid, rejected = annotation.locate_candidates(
        text, generated, "trajectory-1", 0, 100, 15, 128, 0
    )
    assert len(valid) == 1 and len(rejected) == 2
    assert valid[0]["qa_id"] == "trajectory-1:seg0:round0:qa0"
    assert valid[0]["segment_id"] == "seg0"
    assert valid[0]["evidence_char_span"] == [100, 125]
    assert valid[0]["answer_char_span"] == [100, 103]
    assert "exactly once" in rejected[0]["reason"]
    assert "absent" in rejected[1]["reason"]
    with pytest.raises(ValueError, match="count"):
        annotation.locate_candidates(text, generated, "trajectory-1", 0, 0, 2, 128, 0)


def test_candidate_locator_rejects_answer_over_character_limit():
    text = "The destination was Northern Harbor."
    generated = {
        "qas": [
            {
                "fact_statement": "The destination was Northern Harbor.",
                "question": "What was the destination?",
                "answer": "Northern Harbor",
                "evidence_quote": text,
            }
        ],
        "skip_reason": "",
    }
    valid, rejected = annotation.locate_candidates(text, generated, "t", 7, 10, 5, 8, 0)
    assert not valid
    assert rejected[0]["qa_id"] == "t:seg7:round0:qa0"
    assert "character limit" in rejected[0]["reason"]


def test_candidate_locator_detects_overlapping_quote_occurrences():
    generated = {
        "qas": [
            {
                "fact_statement": "An overlapping quote.",
                "question": "What letter?",
                "answer": "a",
                "evidence_quote": "aa",
            }
        ],
        "skip_reason": "",
    }
    valid, rejected = annotation.locate_candidates("aaa", generated, "t", 0, 0, 15, 128, 0)
    assert not valid and "exactly once" in rejected[0]["reason"]


def test_supplement_round_identity_and_usage_survive_cached_replays(
    tmp_path, monkeypatch, client_config, prompts
):
    sent = []

    class Opener:
        def open(self, request, timeout):
            sent.append(request)
            return Response(response_bytes({"qas": [], "skip_reason": "No new facts"}))

    monkeypatch.setattr(annotation, "build_opener", lambda *_args: Opener())
    root = tmp_path / "batch"
    client = annotation.AnnotationClient(client_config, root, tmp_path / "cache", prompts)
    for document, round_index in [("doc-a", 0), ("doc-a", 0), ("doc-a", 1), ("doc-b", 0)]:
        content = {
            "trajectory_id": document,
            "round_index": round_index,
            "phase": "annotate",
            "segment_text": "Same source",
            "candidate_limit": 3,
        }
        client.call("generate", content, annotation.GENERATE_SCHEMA)
    assert len(sent) == 3
    logs = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    assert logs[0]["request_id"] == logs[1]["request_id"]
    assert logs[0]["request_id"] != logs[2]["request_id"]
    assert logs[2]["round_index"] == 1 and logs[2]["phase"] == "annotate"
    usage = pipeline._request_statistics(root, "doc-a", 0)["generate"]
    assert usage["new_logical_requests"] == usage["network_attempts"] == 1
    assert usage["cache_hits"] == 1
    assert usage["input_tokens"] == 31 and usage["output_tokens"] == 7
    all_usage = pipeline._request_statistics(root)["generate"]
    assert all_usage["input_tokens"] == 93 and all_usage["output_tokens"] == 21


def test_content_filter_is_cached_document_failure_and_does_not_stop_other_requests(
    tmp_path, monkeypatch, client_config, prompts
):
    sends = []
    blocked = json.loads(response_bytes({"answer": "partial"}))
    blocked.update(status="incomplete", incomplete_details={"reason": "content_filter"})
    blocked["output"][0]["content"][0]["text"] = '{"answer":"unfinished'
    raw = json.dumps(blocked).encode()

    def send(self, payload):
        sends.append(payload)
        return 200, raw if len(sends) == 1 else response_bytes({"answer": "Oslo"})

    monkeypatch.setattr(annotation.AnnotationClient, "_send", send)
    cache, root = tmp_path / "cache", tmp_path / "batch"
    for _ in range(2):
        client = annotation.AnnotationClient(client_config, root, cache, prompts)
        with pytest.raises(annotation.ContentFilteredError) as caught:
            client.call("answer", {"question": "Filtered question"}, annotation.ANSWER_SCHEMA)
        assert caught.value.stage == "answer"
        assert Path(caught.value.raw_response_path).read_bytes() == raw
        assert not client.stop.is_set()
    assert len(sends) == 1
    assert not list(cache.glob("*/*/parsed.json"))
    assert client.call(
        "answer", {"question": "A different document"}, annotation.ANSWER_SCHEMA
    ) == {"answer": "Oslo"}
    logs = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    assert logs[0]["failure_reason"] == logs[1]["failure_reason"] == "content_filtered"
    assert logs[1]["cache_source"] == "raw" and logs[1]["network_attempts"] == 0
    assert logs[0]["usage"] == logs[1]["usage"] == blocked["usage"]
    usage = pipeline._request_statistics(root)["answer"]
    assert usage["content_filtered_calls"] == 2
    assert usage["input_tokens"] == 62 and usage["output_tokens"] == 14


@pytest.mark.parametrize("reason", ["unknown", None])
def test_other_incomplete_responses_remain_fatal(
    tmp_path, monkeypatch, client_config, prompts, reason
):
    raw = json.loads(response_bytes({"answer": "partial"}))
    raw.update(status="incomplete", incomplete_details={"reason": reason})
    monkeypatch.setattr(
        annotation.AnnotationClient, "_send", lambda *_: (200, json.dumps(raw).encode())
    )
    client = annotation.AnnotationClient(
        client_config, tmp_path / "batch", tmp_path / "cache", prompts
    )
    with pytest.raises(ValueError, match="not completed"):
        client.call("answer", {"question": "Question"}, annotation.ANSWER_SCHEMA)
    assert client.stop.is_set()


def test_filtered_response_log_write_error_remains_fatal(
    tmp_path, monkeypatch, client_config, prompts
):
    raw = json.loads(response_bytes({"answer": "partial"}))
    raw.update(status="incomplete", incomplete_details={"reason": "content_filter"})
    monkeypatch.setattr(
        annotation.AnnotationClient, "_send", lambda *_: (200, json.dumps(raw).encode())
    )
    client = annotation.AnnotationClient(
        client_config, tmp_path / "batch", tmp_path / "cache", prompts
    )

    def fail_log(record):
        raise OSError("request log write failed")

    monkeypatch.setattr(client, "_append_log", fail_log)
    with pytest.raises(OSError, match="request log write failed"):
        client.call("answer", {"question": "Question"}, annotation.ANSWER_SCHEMA)
    assert client.stop.is_set()


@pytest.mark.parametrize("reason", ["content_filter", "max_output_tokens"])
@pytest.mark.parametrize(
    "field,value",
    [("model", "wrong-model"), ("reasoning", {"effort": "low"}), ("usage", "invalid")],
)
def test_incomplete_response_does_not_hide_invalid_response_contract(
    tmp_path, monkeypatch, client_config, prompts, field, value, reason
):
    raw = json.loads(response_bytes({"answer": "partial"}))
    raw.update(status="incomplete", incomplete_details={"reason": reason})
    raw[field] = value
    monkeypatch.setattr(
        annotation.AnnotationClient, "_send", lambda *_: (200, json.dumps(raw).encode())
    )
    client = annotation.AnnotationClient(
        client_config, tmp_path / "batch", tmp_path / "cache", prompts
    )
    with pytest.raises(ValueError) as caught:
        client.call("answer", {"question": "Question"}, annotation.ANSWER_SCHEMA)
    assert not isinstance(caught.value, annotation.ContentFilteredError)
    assert client.stop.is_set()


def _verification_fixture():
    content = {"qas": [{"qa_id": "q0"}, {"qa_id": "q1"}]}
    output = {
        "decisions": [{"qa_id": q["qa_id"], "accepted": True, "reason": ""} for q in content["qas"]]
    }
    return content, output


@pytest.mark.parametrize(
    "invalid_kind",
    ["unknown_id", "duplicate_id", "missing_id", "extra_id", "empty_reason", "bad_json"],
)
def test_response_contract_retry_precedes_success_cache(
    tmp_path, monkeypatch, client_config, prompts, invalid_kind
):
    content, output = _verification_fixture()
    invalid = json.loads(json.dumps(output))
    if invalid_kind == "unknown_id":
        invalid["decisions"][1]["qa_id"] = "q1-wrong"
    elif invalid_kind == "duplicate_id":
        invalid["decisions"][1]["qa_id"] = "q0"
    elif invalid_kind == "missing_id":
        invalid["decisions"].pop()
    elif invalid_kind == "extra_id":
        invalid["decisions"].append(invalid["decisions"][0].copy())
    elif invalid_kind == "empty_reason":
        invalid["decisions"][1]["accepted"] = False
    raw = response_bytes(invalid)
    if invalid_kind == "bad_json":
        envelope = json.loads(raw)
        envelope["output"][0]["content"][0]["text"] = "{"
        raw = json.dumps(envelope).encode()
    sent = []
    cache, root = tmp_path / "cache", tmp_path / "batch"

    def send(self, payload):
        sent.append(payload)
        schema = payload["text"]["format"]["schema"]["properties"]["decisions"]
        assert schema["minItems"] == schema["maxItems"] == 2
        assert schema["items"]["properties"]["qa_id"]["enum"] == ["q0", "q1"]
        if len(sent) == 2:
            assert not list(cache.glob("*/*/parsed.json"))
            assert len(list(cache.glob("*/*/response.attempt-1.raw.json"))) == 1
        return 200, raw if len(sent) == 1 else response_bytes(output)

    monkeypatch.setattr(annotation.AnnotationClient, "_send", send)
    client = annotation.AnnotationClient(client_config, root, cache, prompts)
    assert client.call("verify", content, annotation.VERIFY_SCHEMA) == output
    assert not client.stop.is_set() and len(sent) == 2
    assert client.call("verify", content, annotation.VERIFY_SCHEMA) == output
    assert len(sent) == 2
    stats = pipeline._request_statistics(root)["verify"]
    assert stats["new_logical_requests"] == 1 and stats["network_attempts"] == 2
    assert stats["input_tokens"] == 62 and stats["output_tokens"] == 14


def test_old_invalid_success_cache_is_revalidated_and_retried_once(
    tmp_path, monkeypatch, client_config, prompts
):
    content, output = _verification_fixture()
    cache, root = tmp_path / "cache", tmp_path / "batch"
    client = annotation.AnnotationClient(client_config, root, cache, prompts)
    payload = client._payload("verify", content, annotation.VERIFY_SCHEMA)
    request_id = hashlib.sha256(annotation._payload_bytes(payload)).hexdigest()
    folder = cache / request_id[:2] / request_id
    folder.mkdir(parents=True)
    invalid = json.loads(json.dumps(output))
    invalid["decisions"][0]["qa_id"] = "q0-wrong"
    (folder / "request.json").write_text(json.dumps(payload))
    (folder / "parsed.json").write_text(json.dumps(invalid))
    (folder / "response.raw.json").write_bytes(response_bytes(invalid))
    (folder / "attempt-1.json").write_text(
        json.dumps({"attempt": 1, "status": "response_received"})
    )
    sent = []

    def send(self, actual):
        sent.append(actual)
        return 200, response_bytes(output)

    monkeypatch.setattr(annotation.AnnotationClient, "_send", send)
    assert client.call("verify", content, annotation.VERIFY_SCHEMA) == output
    assert len(sent) == 1
    assert (folder / "response.attempt-1.raw.json").read_bytes() == response_bytes(invalid)
    assert json.loads((folder / "attempt-1.json").read_text())["status"] == "invalid_response"
    assert (
        json.loads((folder / "attempt-2.json").read_text())["response_schema"]["properties"][
            "decisions"
        ]["maxItems"]
        == 2
    )
    log = json.loads((root / "requests.jsonl").read_text().splitlines()[0])
    assert log["request_id"] == request_id and log["attempts_total"] == 2
    # Only attempt 2 was sent by this batch; attempt 1 may belong to another batch.
    assert pipeline._request_statistics(root)["verify"]["input_tokens"] == 31


@pytest.mark.parametrize("truncated", [False, True])
def test_invalid_response_budget_survives_interruption_and_exhaustion(
    tmp_path, monkeypatch, client_config, prompts, truncated
):
    content, output = _verification_fixture()
    invalid = json.loads(json.dumps(output))
    invalid["decisions"][1]["qa_id"] = "q0"
    raw = response_bytes(invalid)
    if truncated:
        envelope = json.loads(raw)
        envelope.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        raw = json.dumps(envelope).encode()
    sent = []

    def send(self, payload):
        sent.append(payload)
        return 200, raw

    monkeypatch.setattr(annotation.AnnotationClient, "_send", send)
    cache, root = tmp_path / "cache", tmp_path / "batch"
    first = annotation.AnnotationClient(client_config, root, cache, prompts)
    # Stop between attempts, after the invalid response has been durably archived.
    monkeypatch.setattr(first.stop, "wait", lambda _: True)
    with pytest.raises(InterruptedError):
        first.call("verify", content, annotation.VERIFY_SCHEMA)
    second = annotation.AnnotationClient(client_config, root, cache, prompts)
    with pytest.raises(annotation.AnnotationContractError):
        second.call("verify", content, annotation.VERIFY_SCHEMA)
    assert len(sent) == 3 and not second.stop.is_set()
    third = annotation.AnnotationClient(client_config, root, cache, prompts)
    with pytest.raises(annotation.AnnotationContractError):
        third.call("verify", content, annotation.VERIFY_SCHEMA)
    assert len(sent) == 3 and not third.stop.is_set()
    stats = pipeline._request_statistics(root)["verify"]
    assert stats["network_attempts"] == 3 and stats["input_tokens"] == 93


def test_network_and_contract_errors_share_one_attempt_budget(
    tmp_path, monkeypatch, client_config, prompts
):
    content, output = _verification_fixture()
    output["decisions"][1]["qa_id"] = "q0"
    calls = []

    def send(self, payload):
        calls.append(payload)
        if len(calls) == 1:
            raise HTTPError("http://local", 429, "limited", {"Retry-After": "0"}, BytesIO())
        return 200, response_bytes(output)

    monkeypatch.setattr(annotation.AnnotationClient, "_send", send)
    client = annotation.AnnotationClient(
        client_config, tmp_path / "batch", tmp_path / "cache", prompts
    )
    with pytest.raises(annotation.AnnotationContractError):
        client.call("verify", content, annotation.VERIFY_SCHEMA)
    assert len(calls) == 3 and not client.stop.is_set()
    stats = pipeline._request_statistics(tmp_path / "batch")["verify"]
    assert stats["network_attempts"] == 3 and stats["input_tokens"] == 62


def test_invalid_response_archive_write_error_is_fatal(
    tmp_path, monkeypatch, client_config, prompts
):
    content, output = _verification_fixture()
    output["decisions"][0]["qa_id"] = "wrong"
    calls = []

    def send(self, payload):
        calls.append(payload)
        return 200, response_bytes(output)

    monkeypatch.setattr(annotation.AnnotationClient, "_send", send)
    original = annotation._atomic_write

    def fail_archive(path, contents):
        if path.name.startswith("response.attempt-"):
            raise OSError("archive write failed")
        return original(path, contents)

    monkeypatch.setattr(annotation, "_atomic_write", fail_archive)
    client = annotation.AnnotationClient(
        client_config, tmp_path / "batch", tmp_path / "cache", prompts
    )
    with pytest.raises(OSError, match="archive write failed"):
        client.call("verify", content, annotation.VERIFY_SCHEMA)
    assert client.stop.is_set() and len(calls) == 1


@pytest.mark.parametrize(
    "partial_text", ['{"qas":[', '{"qas":[],"skip_reason":"partial"}', '{"qas":[' + " " * 20000]
)
def test_max_output_tokens_retries_at_same_limit_without_salvaging_partial_output(
    tmp_path, monkeypatch, client_config, prompts, partial_text
):
    output = {"qas": [], "skip_reason": "No supported facts"}
    raw = json.loads(response_bytes(output))
    raw.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    raw["usage"]["output_tokens"] = 8192
    raw["output"][0]["content"][0]["text"] = partial_text
    truncated = json.dumps(raw).encode()
    sent = []
    cache, root = tmp_path / "cache", tmp_path / "batch"

    def send(self, payload):
        sent.append(payload)
        if len(sent) == 2:
            assert not list(cache.glob("*/*/parsed.json"))
            archived = list(cache.glob("*/*/response.attempt-1.raw.json"))
            assert len(archived) == 1 and archived[0].read_bytes() == truncated
            attempt = json.loads((archived[0].parent / "attempt-1.json").read_text())
            assert attempt["status"] == "invalid_response"
            assert "max_output_tokens" in attempt["error"]
        return 200, truncated if len(sent) == 1 else response_bytes(output)

    monkeypatch.setattr(annotation.AnnotationClient, "_send", send)
    client = annotation.AnnotationClient(client_config, root, cache, prompts)
    content = {"candidate_limit": 15}
    assert client.call("generate", content, annotation.GENERATE_SCHEMA) == output
    assert len(sent) == 2 and sent[0] == sent[1]
    assert all(p["max_output_tokens"] == 8192 for p in sent)
    assert not client.stop.is_set()
    assert client.call("generate", content, annotation.GENERATE_SCHEMA) == output
    assert len(sent) == 2
    usage = pipeline._request_statistics(root)["generate"]
    assert usage["network_attempts"] == 2
    assert usage["input_tokens"] == 62 and usage["output_tokens"] == 8199
