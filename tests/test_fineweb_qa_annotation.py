from concurrent.futures import ThreadPoolExecutor
import copy
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError, URLError

import pytest

from latent_working_memory.data_preparation.fineweb_qa import annotation
from latent_working_memory.data_preparation.fineweb_qa.annotation import (
    ANSWER_SCHEMA,
    AnnotationClient,
    annotate_segment,
    review_document,
)


@pytest.fixture
def config(tmp_path):
    return {
        "annotation": {
            "endpoint": "http://example.invalid/v1/responses",
            "model": "gpt-6-sol",
            "reasoning_effort": "medium",
            "concurrency": 2,
            "max_attempts": 3,
            "timeout_seconds": 30,
            "max_answer_chars": 128,
            "candidate_counts": [3] * 8,
            "max_output_tokens": {
                "generate": 8192, "verify": 4096, "document_review": 8192, "answer": 2048,
            },
        },
        "output": {
            "request_cache_dir": str(tmp_path / "cache"),
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
    }


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("tests must not contact a model service")

    monkeypatch.setattr(annotation, "build_opener", forbidden)


def response_envelope(value):
    return {
        "id": "response-test",
        "status": "completed",
        "model": "gpt-6-sol",
        "reasoning": {"effort": "medium"},
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        "output": [{
            "type": "message", "phase": "final_answer",
            "content": [{"type": "output_text", "text": json.dumps(value)}],
        }],
    }


class Response(BytesIO):
    status = 200


def install_service(monkeypatch, handler):
    class Opener:
        def open(self, request, timeout):
            assert timeout == 30
            body = handler(json.loads(request.data))
            return Response(json.dumps(body).encode())

    def build_opener(proxy_handler):
        assert proxy_handler.proxies == {}
        return Opener()

    monkeypatch.setattr(annotation, "build_opener", build_opener)


def test_cache_uses_complete_payload_and_is_shared_between_clients(config, monkeypatch):
    requests = []

    def respond(payload):
        requests.append(payload)
        return response_envelope({"answer": "Blue Note"})

    install_service(monkeypatch, respond)
    client = AnnotationClient(config)
    expected = {"answer": "Blue Note"}
    assert client.call("answer", "Read carefully.", {"question": "Where?"}, ANSWER_SCHEMA) == expected
    second_config = copy.deepcopy(config)
    second_config["output"]["artifacts_dir"] += "-second"
    second = AnnotationClient(second_config)
    assert second.call("answer", "Read carefully.", {"question": "Where?"}, ANSWER_SCHEMA) == expected
    assert second.calls[0]["cache_hit"] and second.calls[0]["attempts"] == []
    assert len(requests) == 1
    assert requests[0]["text"]["format"]["strict"] is True
    assert requests[0]["reasoning"] == {"effort": "medium"}
    assert client.calls[0]["usage"]["total_tokens"] == 120
    assert client.calls[0]["attempts"][0]["response_id"] == "response-test"
    assert client.calls[0]["elapsed_seconds"] >= 0

    second.call("answer", "A changed prompt.", {"question": "Where?"}, ANSWER_SCHEMA)
    second.settings["max_output_tokens"]["answer"] += 1
    second.call("answer", "A changed prompt.", {"question": "Where?"}, ANSWER_SCHEMA)
    assert len(requests) == 3
    assert len({record["cache_key"] for record in second.calls}) == 3


@pytest.mark.parametrize("malformation", ["extra_field", "wrong_type", "incomplete", "broken_envelope"])
def test_invalid_structured_output_stops_without_retry(config, monkeypatch, malformation):
    calls = []
    envelope = response_envelope({"answer": "Blue Note"})
    if malformation == "extra_field":
        envelope = response_envelope({"answer": "Blue Note", "explanation": "extra"})
    elif malformation == "wrong_type":
        envelope = response_envelope({"answer": 7})
    elif malformation == "incomplete":
        envelope["status"] = "incomplete"
    else:
        envelope["output"] = [{"type": "message", "content": None}]

    def respond(payload):
        calls.append(payload)
        return envelope

    install_service(monkeypatch, respond)
    client = AnnotationClient(config)
    with pytest.raises(ValueError):
        client.call("answer", "Answer.", {}, ANSWER_SCHEMA)
    with pytest.raises(ValueError, match="requires correction"):
        AnnotationClient(config).call("answer", "Answer.", {}, ANSWER_SCHEMA)
    assert len(calls) == 1
    assert client.calls[0]["usage"]["total_tokens"] == 120
    assert client.calls[0]["attempts"][0]["retryable"] is False


@pytest.mark.parametrize("failed_file", ["attempt-001.response.json.tmp", "parsed.json.tmp"])
def test_local_write_failure_does_not_retry_a_completed_remote_request(config, monkeypatch, failed_file):
    calls = []

    def respond(payload):
        calls.append(payload)
        return response_envelope({"answer": "Paris"})

    install_service(monkeypatch, respond)
    original_write = Path.write_text

    def fail_write(path, *args, **kwargs):
        if path.name == failed_file:
            raise OSError("disk full")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_write)
    client = AnnotationClient(config)
    with pytest.raises(OSError, match="disk full"):
        client.call("answer", "Answer.", {}, ANSWER_SCHEMA)
    assert len(calls) == 1
    assert client.stop_event.is_set()
    assert len(client.calls[0]["attempts"]) == 1
    assert client.calls[0]["attempts"][0]["retryable"] is False


@pytest.mark.parametrize("keep_metadata", [True, False])
def test_saved_raw_response_is_recovered_without_another_post(config, monkeypatch, keep_metadata):
    calls = []

    def respond(payload):
        calls.append(payload)
        return response_envelope({"answer": "Paris"})

    install_service(monkeypatch, respond)
    original_save = annotation._save

    def fail_parsed(path, value):
        if path.name == "parsed.json":
            raise OSError("disk full")
        original_save(path, value)

    monkeypatch.setattr(annotation, "_save", fail_parsed)
    client = AnnotationClient(config)
    with pytest.raises(OSError, match="disk full"):
        client.call("answer", "Answer.", {}, ANSWER_SCHEMA)
    folder = Path(config["output"]["request_cache_dir"]) / client.calls[0]["cache_key"]
    assert (folder / "attempt-001.response.json").exists()
    if not keep_metadata:
        # 模拟响应已落盘、还没来得及保存解析结果和 attempt metadata 时中断。
        (folder / "attempt-001.meta.json").unlink()
    monkeypatch.setattr(annotation, "_save", original_save)
    recovered = AnnotationClient(config)
    assert recovered.call("answer", "Answer.", {}, ANSWER_SCHEMA) == {"answer": "Paris"}
    assert len(calls) == 1
    assert recovered.calls[0]["cache_hit"] is True
    assert recovered.calls[0]["attempts"] == []
    assert recovered.calls[0]["recovered_response"] == "attempt-001.response.json"
    assert json.loads((folder / "parsed.json").read_text()) == {"answer": "Paris"}


def test_batch_stop_prevents_new_calls(config):
    client = AnnotationClient(config)
    client.stop_event.set()
    with pytest.raises(RuntimeError, match="Batch stopped"):
        client.call("answer", "Answer.", {}, ANSWER_SCHEMA)
    assert client.calls == []


def test_batch_stop_is_checked_after_waiting_for_a_request_slot(config):
    client = AnnotationClient(config)

    class StoppedSlot:
        def __enter__(self):
            client.stop_event.set()

        def __exit__(self, *args):
            return False

    client._slots = StoppedSlot()
    with pytest.raises(RuntimeError, match="Batch stopped"):
        client.call("answer", "Answer.", {}, ANSWER_SCHEMA)
    assert client.calls[0]["attempts"] == []
    assert not list(Path(config["output"]["request_cache_dir"]).glob("*/attempt-*.meta.json"))


def test_retryable_errors_are_bounded_and_recorded(config, monkeypatch):
    calls, sleeps = [], []

    def respond(payload):
        calls.append(payload)
        if len(calls) == 1:
            raise HTTPError("url", 429, "rate limited", {"Retry-After": "7"}, BytesIO(b"busy"))
        if len(calls) == 2:
            raise URLError("temporarily unavailable")
        return response_envelope({"answer": "Paris"})

    install_service(monkeypatch, respond)
    monkeypatch.setattr(annotation.time, "sleep", sleeps.append)
    client = AnnotationClient(config)
    assert client.call("answer", "Answer.", {}, ANSWER_SCHEMA) == {"answer": "Paris"}
    assert sleeps == [7, 10]
    assert [attempt["ok"] for attempt in client.calls[0]["attempts"]] == [False, False, True]
    assert client.calls[0]["attempts"][0]["http_status"] == 429


def test_exhausted_budget_does_not_restart_on_rerun(config, monkeypatch):
    calls = []

    def respond(payload):
        calls.append(payload)
        raise URLError("service unavailable")

    install_service(monkeypatch, respond)
    monkeypatch.setattr(annotation.time, "sleep", lambda delay: None)
    client = AnnotationClient(config)
    with pytest.raises(URLError):
        client.call("answer", "Answer.", {}, ANSWER_SCHEMA)
    with pytest.raises(RuntimeError, match="retry budget exhausted"):
        AnnotationClient(config).call("answer", "Answer.", {}, ANSWER_SCHEMA)
    assert len(calls) == config["annotation"]["max_attempts"]


def test_concurrent_calls_share_limit_and_deduplicate_identical_payloads(config, monkeypatch):
    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "calls": 0}

    def respond(payload):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            state["calls"] += 1
        time.sleep(0.02)
        with lock:
            state["active"] -= 1
        return response_envelope({"answer": "Paris"})

    install_service(monkeypatch, respond)
    client = AnnotationClient(config)
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(
            lambda index: client.call("answer", "Answer.", {"index": index}, ANSWER_SCHEMA),
            [0, 0, 0, 1, 2, 3, 4, 5],
        ))
    assert values == [{"answer": "Paris"}] * 8
    assert state["calls"] == 6
    assert state["peak"] <= 2
    assert sum(record["cache_hit"] for record in client.calls) == 2


class ScriptedClient:
    def __init__(self, config, responses):
        self.settings = config["annotation"]
        self.responses = iter(responses)
        self.calls = []

    def call(self, stage, instructions, content, schema):
        self.calls.append({"stage": stage, "content": content})
        result = next(self.responses)
        annotation._check_json(result, schema)
        return result


PROMPTS = {name: name for name in ("generate", "verify", "document_review")}


def candidate(answer="Blue Note", quote="Maya visited Blue Note."):
    return {
        "fact_statement": "Maya visited Blue Note.",
        "question": "Where did Maya visit?",
        "answer": answer,
        "evidence_quote": quote,
    }


def decision(qa_id, accepted=True):
    return {"qa_id": qa_id, "accepted": accepted, "reason": "supported" if accepted else "ambiguous"}


def test_segment_offsets_and_original_candidate_ids_survive_rejection(config):
    prefix = "Earlier history.\n"
    text = "Maya visited Blue Note. Maya stayed in Paris."
    segment = {"segment_id": 2, "char_start": len(prefix), "char_end": len(prefix + text)}
    trajectory = {"trajectory_id": "doc1", "text": prefix + text, "segments": [segment]}
    candidates = [candidate("Missing"), candidate(), candidate("Paris", "Maya stayed in Paris.")]
    client = ScriptedClient(config, [
        {"qas": candidates, "shortfall_reason": ""},
        {"decisions": [decision("doc1:s2:q2"), decision("doc1:s2:q3", False)]},
    ])
    result = annotate_segment(client, trajectory, segment, PROMPTS)
    assert result["counts"] == {"generated": 3, "program_accepted": 2, "accepted": 1}
    assert result["program_rejections"][0]["qa_id"] == "doc1:s2:q1"
    qa = result["accepted"][0]
    assert qa["qa_id"] == "doc1:s2:q2"
    assert qa["segment_id"] == 2
    for field, span in [("evidence_quote", "evidence_span"), ("answer", "answer_span")]:
        start, end = qa[span]
        assert trajectory["text"][start:end] == qa[field]
        assert start >= len(prefix)
    verified = client.calls[1]["content"]
    assert verified["text"] == text
    assert all("fact_statement" not in row for row in verified["qas"])


def test_invalid_spans_and_long_answers_skip_verification(config):
    text = "Maya visited Blue Note. Maya visited Blue Note. " + "x" * 129
    segment = {"segment_id": 1, "char_start": 0, "char_end": len(text)}
    trajectory = {"trajectory_id": "doc1", "text": text, "segments": [segment]}
    client = ScriptedClient(config, [{
        "qas": [candidate(), candidate("x" * 129, "x" * 129)],
        "shortfall_reason": "Only two candidate facts.",
    }])
    result = annotate_segment(client, trajectory, segment, PROMPTS)
    assert result["accepted"] == []
    assert len(result["program_rejections"]) == 2
    assert len(client.calls) == 1


@pytest.mark.parametrize("reason", ["", "   "])
def test_shortfall_requires_explanation(config, reason):
    segment = {"segment_id": 1, "char_start": 0, "char_end": 4}
    trajectory = {"trajectory_id": "doc1", "text": "Text", "segments": [segment]}
    client = ScriptedClient(config, [{"qas": [], "shortfall_reason": reason}])
    with pytest.raises(ValueError, match="shortfall_reason"):
        annotate_segment(client, trajectory, segment, PROMPTS)


@pytest.mark.parametrize("ids", [[], ["unknown"], ["doc1:s1:q1", "doc1:s1:q1"]])
def test_verification_must_cover_exact_ids(config, ids):
    text = "Maya visited Blue Note."
    segment = {"segment_id": 1, "char_start": 0, "char_end": len(text)}
    trajectory = {"trajectory_id": "doc1", "text": text, "segments": [segment]}
    client = ScriptedClient(config, [
        {"qas": [candidate()], "shortfall_reason": "Only one fact."},
        {"decisions": [decision(qa_id) for qa_id in ids]},
    ])
    with pytest.raises(ValueError, match="exactly once"):
        annotate_segment(client, trajectory, segment, PROMPTS)


def document_fixture():
    segments = [
        {"segment_id": index + 1, "char_start": index * 6, "char_end": (index + 1) * 6}
        for index in range(8)
    ]
    trajectory = {"trajectory_id": "doc1", "text": "".join(f"part{i}." for i in range(8)), "segments": segments}
    qas = [{**candidate(), "qa_id": f"q{i}", "segment_id": i} for i in range(1, 4)]
    return trajectory, qas


def test_document_review_receives_all_segments_without_generator_explanations(config):
    trajectory, qas = document_fixture()
    client = ScriptedClient(config, [{
        "decisions": [decision("q1"), decision("q2"), decision("q3", False)],
        "same_fact_groups": [["q1", "q2"]],
    }])
    result = review_document(client, trajectory, qas, PROMPTS)
    assert [qa["qa_id"] for qa in result["accepted"]] == ["q1", "q2"]
    assert result["same_fact_groups"] == [["q1", "q2"]]
    sent = client.calls[0]["content"]
    assert "".join(row["text"] for row in sent["segments"]) == trajectory["text"]
    assert len(sent["segments"]) == 8
    assert all("fact_statement" not in row for row in sent["qas"])


@pytest.mark.parametrize("groups", [[["q1", "unknown"]], [["q1", "q1"]], [["q1", "q2"], ["q2", "q3"]]])
def test_document_groups_must_be_disjoint_known_qa_ids(config, groups):
    trajectory, qas = document_fixture()
    client = ScriptedClient(config, [{
        "decisions": [decision(row["qa_id"]) for row in qas],
        "same_fact_groups": groups,
    }])
    with pytest.raises(ValueError, match="disjoint groups"):
        review_document(client, trajectory, qas, PROMPTS)
