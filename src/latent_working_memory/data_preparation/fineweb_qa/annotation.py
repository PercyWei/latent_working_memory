"""FineWeb 事实 QA 的本地结构化标注、核验与共享请求缓存。"""

from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


STRING = {"type": "string"}
NONEMPTY_STRING = {"type": "string", "minLength": 1}
QA_SCHEMA = object_schema(
    {key: NONEMPTY_STRING for key in ("fact_statement", "question", "answer", "evidence_quote")}
)
DECISION_SCHEMA = object_schema(
    {"qa_id": NONEMPTY_STRING, "accepted": {"type": "boolean"}, "reason": NONEMPTY_STRING}
)
VERIFY_SCHEMA = object_schema({"decisions": {"type": "array", "items": DECISION_SCHEMA}})
DOCUMENT_REVIEW_SCHEMA = object_schema(
    {
        "decisions": {"type": "array", "items": DECISION_SCHEMA},
        "same_fact_groups": {
            "type": "array",
            "items": {"type": "array", "items": NONEMPTY_STRING, "minItems": 2},
        },
    }
)
ANSWER_SCHEMA = object_schema({"answer": NONEMPTY_STRING})


def generation_schema(candidate_limit):
    return object_schema(
        {
            "qas": {"type": "array", "items": QA_SCHEMA, "maxItems": candidate_limit},
            "shortfall_reason": STRING,
        }
    )


def _check_json(value, schema, path="response"):
    """校验本模块实际使用的四种 JSON 类型；不依赖服务端一定遵守 schema。"""
    kind = schema["type"]
    expected = {"object": dict, "array": list, "string": str, "boolean": bool}[kind]
    if type(value) is not expected:
        raise ValueError(f"{path}: expected {kind}")
    if kind == "object":
        if set(value) != set(schema["properties"]):
            raise ValueError(f"{path}: fields differ from schema")
        for key, field_schema in schema["properties"].items():
            _check_json(value[key], field_schema, f"{path}.{key}")
    elif kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            raise ValueError(f"{path}: array length differs from schema")
        for index, item in enumerate(value):
            _check_json(item, schema["items"], f"{path}[{index}]")
    elif kind == "string" and len(value) < schema.get("minLength", 0):
        raise ValueError(f"{path}: empty string")


def _now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _request_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "request.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _parse_response(data, schema):
    if not isinstance(data, dict) or data.get("status") != "completed":
        raise ValueError("Responses request did not complete")
    try:
        texts = [
            part["text"]
            for item in data["output"]
            if item["type"] == "message" and item.get("phase") in {None, "final_answer"}
            for part in item["content"]
            if part["type"] == "output_text"
        ]
        parsed = json.loads("".join(texts))
    except (AttributeError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Responses JSON output") from error
    _check_json(parsed, schema)
    return parsed


class AnnotationClient:
    """一个批次共用一个实例；同请求可跨线程和构造批次复用已成功响应。"""

    def __init__(self, config):
        self.config = config
        self.settings = config["annotation"]
        self.cache_dir = Path(config["output"]["request_cache_dir"])
        self.artifacts_dir = Path(config["output"]["artifacts_dir"])
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._slots = threading.BoundedSemaphore(self.settings["concurrency"])
        self._log_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.calls = []

    def _log(self, record):
        with self._log_lock:
            with (self.artifacts_dir / "requests.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.calls.append(record)

    def call(self, stage, instructions, content, schema):
        if self.stop_event.is_set():
            raise RuntimeError("Batch stopped after another request failed")
        payload = {
            "model": self.settings["model"],
            "reasoning": {"effort": self.settings["reasoning_effort"]},
            "instructions": instructions,
            "input": [{"role": "user", "content": json.dumps(content, ensure_ascii=False, sort_keys=True)}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"fineweb_qa_{stage}",
                    "strict": True,
                    "schema": schema,
                }
            },
            "max_output_tokens": self.settings["max_output_tokens"][stage],
            "store": False,
            "stream": False,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        key = hashlib.sha256(encoded).hexdigest()
        folder = self.cache_dir / key
        started = time.perf_counter()
        record = {
            "stage": stage,
            "cache_key": key,
            "cache_hit": False,
            "started_at": _now(),
            "model": self.settings["model"],
            "attempts": [],
            "usage": None,
        }
        try:
            with _request_lock(folder):
                parsed_path = folder / "parsed.json"
                if parsed_path.exists():
                    record.update(cache_hit=True, ok=True)
                    return json.loads(parsed_path.read_text(encoding="utf-8"))
                raw_responses = sorted(folder.glob("attempt-*.response.json"))
                if raw_responses:
                    record["cache_hit"] = True
                    try:
                        data = json.loads(raw_responses[-1].read_text(encoding="utf-8"))
                        parsed = _parse_response(data, schema)
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f"cached request requires correction: {error}") from error
                    _save(parsed_path, parsed)
                    record.update(ok=True, recovered_response=raw_responses[-1].name)
                    return parsed
                _save(folder / "request.json", payload)
                previous = sorted(folder.glob("attempt-*.meta.json"))
                if previous:
                    last = json.loads(previous[-1].read_text(encoding="utf-8"))
                    if not last["retryable"]:
                        raise ValueError(f"cached request requires correction: {last['error']}")
                if len(previous) >= self.settings["max_attempts"]:
                    raise RuntimeError(f"request retry budget exhausted: {key}")
                for index in range(len(previous), self.settings["max_attempts"]):
                    attempt = {
                        "attempt": index + 1,
                        "endpoint": self.settings["endpoint"],
                        "started_at": _now(),
                        "retryable": False,
                    }
                    request_started = time.perf_counter()
                    request_attempted = False
                    delay = min(5 * 2**index, 60)
                    try:
                        request = Request(
                            self.settings["endpoint"],
                            data=encoded,
                            headers={"Content-Type": "application/json"},
                        )
                        try:
                            with self._slots:
                                if self.stop_event.is_set():
                                    raise RuntimeError("Batch stopped after another request failed")
                                request_attempted = True
                                with build_opener(ProxyHandler({})).open(
                                    request, timeout=self.settings["timeout_seconds"]
                                ) as response:
                                    attempt["http_status"] = response.status
                                    response_bytes = response.read()
                        except HTTPError as error:
                            attempt.update(
                                ok=False,
                                http_status=error.code,
                                error=error.read().decode("utf-8", errors="replace"),
                                retryable=error.code in {408, 429, 500, 502, 503, 504},
                            )
                            retry_after = error.headers.get("Retry-After", "") if error.headers else ""
                            if retry_after.isdigit():
                                delay = min(max(delay, int(retry_after)), 60)
                            if not attempt["retryable"] or index + 1 == self.settings["max_attempts"]:
                                raise
                        except (URLError, TimeoutError, OSError) as error:
                            attempt.update(ok=False, error=f"{type(error).__name__}: {error}", retryable=True)
                            if index + 1 == self.settings["max_attempts"]:
                                raise
                        else:
                            # 本地写入、解析失败不能触发另一笔远端请求。
                            raw = response_bytes.decode("utf-8")
                            raw_path = folder / f"attempt-{index + 1:03d}.response.json"
                            temporary = raw_path.with_suffix(".json.tmp")
                            temporary.write_text(raw, encoding="utf-8")
                            temporary.replace(raw_path)
                            data = json.loads(raw)
                            if isinstance(data, dict):
                                attempt.update(
                                    response_id=data.get("id"),
                                    returned_model=data.get("model"),
                                    reasoning=data.get("reasoning"),
                                    response_status=data.get("status"),
                                    usage=data.get("usage"),
                                )
                                record["usage"] = data.get("usage")
                            parsed = _parse_response(data, schema)
                            _save(parsed_path, parsed)
                            attempt["ok"] = True
                            record.update(ok=True, usage=attempt.get("usage"))
                            return parsed
                    except Exception as error:
                        if "error" not in attempt:
                            attempt.update(ok=False, error=f"{type(error).__name__}: {error}")
                        raise
                    finally:
                        if request_attempted:
                            attempt.update(finished_at=_now(), elapsed_seconds=time.perf_counter() - request_started)
                            _save(folder / f"attempt-{index + 1:03d}.meta.json", attempt)
                            record["attempts"].append(attempt)
                    time.sleep(delay)
        except Exception as error:
            self.stop_event.set()
            record.update(ok=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record.update(finished_at=_now(), elapsed_seconds=time.perf_counter() - started)
            self._log(record)


def _decisions_by_id(output, qas):
    decisions = output["decisions"]
    ids = [decision["qa_id"] for decision in decisions]
    if len(ids) != len(set(ids)) or set(ids) != {qa["qa_id"] for qa in qas}:
        raise ValueError("verification decisions must cover each supplied QA exactly once")
    if any(not decision["reason"].strip() for decision in decisions):
        raise ValueError("verification reason must be non-empty")
    return {decision["qa_id"]: decision for decision in decisions}


def _review_qa(qa):
    return {key: qa[key] for key in ("qa_id", "segment_id", "question", "answer", "evidence_quote")}


def annotate_segment(client, trajectory, segment, prompts):
    segment_id = segment["segment_id"]
    start, end = segment["char_start"], segment["char_end"]
    text = trajectory["text"][start:end]
    limit = client.settings["candidate_counts"][segment_id - 1]
    generated = client.call(
        "generate",
        prompts["generate"],
        {"segment_id": segment_id, "text": text, "candidate_limit": limit},
        generation_schema(limit),
    )
    if bool(generated["shortfall_reason"].strip()) != (len(generated["qas"]) < limit):
        raise ValueError("shortfall_reason must explain exactly the below-limit generations")
    located, rejected = [], []
    for index, candidate in enumerate(generated["qas"], 1):
        qa_id = f"{trajectory['trajectory_id']}:s{segment_id}:q{index}"
        quote, answer = candidate["evidence_quote"], candidate["answer"]
        reason = ""
        if any(not value.strip() for value in candidate.values()):
            reason = "empty QA field"
        elif len(answer) > client.settings["max_answer_chars"]:
            reason = "answer exceeds max_answer_chars"
        elif text.count(quote) != 1:
            reason = "evidence_quote must occur exactly once in its segment"
        elif answer not in quote:
            reason = "answer is not a contiguous substring of evidence_quote"
        if reason:
            rejected.append({"qa_id": qa_id, "reason": reason, "qa": candidate})
            continue
        evidence_start = start + text.index(quote)
        answer_start = evidence_start + quote.index(answer)
        located.append(
            {
                **candidate,
                "qa_id": qa_id,
                "segment_id": segment_id,
                "evidence_span": [evidence_start, evidence_start + len(quote)],
                "answer_span": [answer_start, answer_start + len(answer)],
            }
        )
    verification = {"decisions": []}
    if located:
        verification = client.call(
            "verify",
            prompts["verify"],
            {"segment_id": segment_id, "text": text, "qas": [_review_qa(qa) for qa in located]},
            VERIFY_SCHEMA,
        )
    decisions = _decisions_by_id(verification, located)
    accepted = [qa for qa in located if decisions[qa["qa_id"]]["accepted"]]
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "segment_id": segment_id,
        "candidate_limit": limit,
        "generated": generated,
        "program_rejections": rejected,
        "verification": verification,
        "accepted": accepted,
        "counts": {"generated": len(generated["qas"]), "program_accepted": len(located), "accepted": len(accepted)},
    }


def review_document(client, trajectory, accepted_qas, prompts):
    reviewed = {"decisions": [], "same_fact_groups": []}
    if accepted_qas:
        reviewed = client.call(
            "document_review",
            prompts["document_review"],
            {
                "segments": [
                    {
                        "segment_id": segment["segment_id"],
                        "text": trajectory["text"][segment["char_start"] : segment["char_end"]],
                    }
                    for segment in trajectory["segments"]
                ],
                "qas": [_review_qa(qa) for qa in accepted_qas],
            },
            DOCUMENT_REVIEW_SCHEMA,
        )
    decisions = _decisions_by_id(reviewed, accepted_qas)
    grouped = set()
    for group in reviewed["same_fact_groups"]:
        if len(set(group)) != len(group) or not set(group) <= set(decisions) or grouped.intersection(group):
            raise ValueError("same_fact_groups must contain disjoint groups of supplied QA IDs")
        grouped.update(group)
    return {
        "trajectory_id": trajectory["trajectory_id"],
        **reviewed,
        "accepted": [qa for qa in accepted_qas if decisions[qa["qa_id"]]["accepted"]],
    }
