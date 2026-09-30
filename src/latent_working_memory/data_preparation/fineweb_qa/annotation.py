"""Cached, resumable Responses requests and exact FineWeb QA evidence spans."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from latent_working_memory.data_preparation.fineweb_qa.finalization import validate_document_reviews


def _object_schema(properties: dict[str, dict]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_STRING = {"type": "string"}
_BOOLEAN = {"type": "boolean"}
GENERATE_SCHEMA = _object_schema(
    {
        "qas": {
            "type": "array",
            "items": _object_schema(
                {
                    "fact_statement": _STRING,
                    "question": _STRING,
                    "answer": _STRING,
                    "evidence_quote": _STRING,
                }
            ),
            "maxItems": 15,
        },
        "skip_reason": _STRING,
    }
)
VERIFY_SCHEMA = _object_schema(
    {
        "decisions": {
            "type": "array",
            "items": _object_schema({"qa_id": _STRING, "accepted": _BOOLEAN, "reason": _STRING}),
        }
    }
)
DOCUMENT_REVIEW_SCHEMA = _object_schema(
    {
        "decisions": {
            "type": "array",
            "items": _object_schema(
                {
                    "qa_id": _STRING,
                    "accepted": _BOOLEAN,
                    "reason": _STRING,
                    "fact_group_id": _STRING,
                }
            ),
        }
    }
)
ANSWER_SCHEMA = _object_schema({"answer": _STRING})
REVIEW_SCHEMA = _object_schema(
    {
        "decisions": {
            "type": "array",
            "items": _object_schema(
                {
                    "qa_id": _STRING,
                    "accepted": _BOOLEAN,
                    "reason": _STRING,
                    "same_fact_with": {"type": "array", "items": _STRING},
                    "evidence_prediction_correct": _BOOLEAN,
                }
            ),
        }
    }
)
STAGES = ("generate", "verify", "document_review", "answer", "review", "adjudicate")

_LOCAL_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


def _local_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCAL_LOCKS_GUARD:
        return _LOCAL_LOCKS.setdefault(key, threading.Lock())


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _atomic_write(path: Path, contents: bytes) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    contents = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    _atomic_write(path, contents)


def _payload_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _validate_schema(value: Any, schema: dict, path: str = "$") -> None:
    """Check the small JSON Schema subset used by the annotation and review request stages."""
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(f"{path} has unexpected fields")
        for key, child in properties.items():
            if key in value:
                _validate_schema(value[key], child, f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        if len(value) > schema.get("maxItems", len(value)):
            raise ValueError(f"{path} exceeds maxItems")
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path} is below minItems")
        for index, item in enumerate(value):
            _validate_schema(item, schema["items"], f"{path}[{index}]")
    elif kind == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
    elif kind == "boolean":
        if type(value) is not bool:
            raise ValueError(f"{path} must be a boolean")
    else:
        raise ValueError(f"unsupported schema type: {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is outside enum")


class ResponseContractError(ValueError):
    """A known truncated or invalid model output cannot satisfy the QA contract."""


class DocumentAnnotationError(ValueError):
    """A known terminal annotation outcome excludes one document."""

    reason: str

    def __init__(
        self, stage: str, request_id: str, raw_response_path: str, detail: str = ""
    ) -> None:
        super().__init__(f"{self.reason}: {detail}" if detail else self.reason)
        self.stage = stage
        self.request_id = request_id
        self.raw_response_path = raw_response_path


class ContentFilteredError(DocumentAnnotationError):
    reason = "content_filtered"


class AnnotationContractError(DocumentAnnotationError):
    reason = "annotation_contract_failed"


def response_schema(stage: str, content: dict, schema: dict) -> dict:
    """Bind decision IDs/counts to this request, without changing its logical identity."""
    bound = copy.deepcopy(schema)
    if stage == "generate":
        bound["properties"]["qas"]["maxItems"] = min(
            bound["properties"]["qas"]["maxItems"], content["candidate_limit"]
        )
    if stage in ("verify", "document_review", "review", "adjudicate"):
        ids = [qa["qa_id"] for qa in content["qas"]]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("decision request needs distinct nonempty candidate IDs")
        decisions = bound["properties"]["decisions"]
        decisions.update(minItems=len(ids), maxItems=len(ids))
        decisions["items"]["properties"]["qa_id"] = {"type": "string", "enum": ids}
    return bound


def validate_output(output: dict, stage: str, content: dict, schema: dict) -> None:
    """Validate model JSON and request coverage before a successful cache is written."""
    try:
        _validate_schema(output, schema)
        if stage in ("verify", "document_review", "review", "adjudicate"):
            decisions = output["decisions"]
            ids = [d["qa_id"] for d in decisions]
            expected = [qa["qa_id"] for qa in content["qas"]]
            if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
                raise ValueError(f"{stage} QA IDs do not match the candidate set")
            for decision in decisions:
                if not decision["accepted"] and not decision["reason"].strip():
                    raise ValueError(f"rejected {stage} decision needs a reason")
                if (
                    stage == "document_review"
                    and decision["accepted"]
                    and not decision["fact_group_id"].strip()
                ):
                    raise ValueError("accepted document-review QA needs a fact group")
            if stage == "adjudicate":
                validate_document_reviews(
                    content["qas"], output, {"review_decisions": content["candidate_decisions"]}
                )
    except ValueError as error:
        raise ResponseContractError(str(error)) from error


class AnnotationClient:
    """Share completed requests across batches while logging each batch's actual use."""

    def __init__(
        self,
        annotation_config: dict,
        batch_root: Path,
        cache_root: Path,
        prompts: dict[str, str],
    ) -> None:
        required = {
            "endpoint",
            "model",
            "reasoning_effort",
            "timeout_seconds",
            "max_attempts",
            "concurrency",
            "max_output_tokens",
        }
        if set(annotation_config) != required:
            raise ValueError(f"annotation_config requires exactly {sorted(required)}")
        if not set(STAGES) <= set(prompts) or any(not prompts[s].strip() for s in STAGES):
            raise ValueError("prompts require non-empty text for every annotation stage")
        limits = annotation_config["max_output_tokens"]
        if set(limits) != set(STAGES) or any(
            type(limits[s]) is not int or limits[s] < 1 for s in STAGES
        ):
            raise ValueError("max_output_tokens requires a positive limit for every stage")
        if not 1 <= annotation_config["max_attempts"] <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        if not 1 <= annotation_config["concurrency"] <= 4:
            raise ValueError("concurrency must be between 1 and 4")
        if annotation_config["timeout_seconds"] <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.config = annotation_config
        self.prompts = prompts
        self.batch_root = Path(batch_root)
        self.cache_root = Path(cache_root)
        self.batch_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self._slots = threading.BoundedSemaphore(annotation_config["concurrency"])

    def call(self, stage: str, content: dict, schema: dict) -> dict:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "stage": stage,
            "trajectory_id": content.get("trajectory_id"),
            "round_index": content.get("round_index"),
            "phase": content.get("phase"),
            "started_at": _now(),
            "cache_hit": False,
            "cache_source": None,
            "network_attempts": 0,
            "network_attempt_numbers": [],
            "attempts_total": 0,
            "network_seconds": 0.0,
            "usage": None,
            "ok": False,
        }
        result = None
        error = None
        try:
            if self.stop.is_set():
                raise InterruptedError("annotation stopped after a fatal error")
            payload = self._payload(stage, content, schema)
            request_id = hashlib.sha256(_payload_bytes(payload)).hexdigest()
            record["request_id"] = request_id
            bound_schema = response_schema(stage, content, schema)
            result = self._cached_call(payload, bound_schema, request_id, record)
            record["ok"] = True
        except DocumentAnnotationError as exc:
            record["failure_reason"] = exc.reason
            record["error"] = f"{type(exc).__name__}: {exc}"
            error = exc
        except Exception as exc:
            self.stop.set()
            record["error"] = f"{type(exc).__name__}: {exc}"
            error = exc
        record["finished_at"] = _now()
        record["elapsed_seconds"] = time.perf_counter() - started
        try:
            self._append_log(record)
        except Exception as log_error:
            self.stop.set()
            if error is None or isinstance(error, DocumentAnnotationError):
                raise
            error.add_note(f"request log write also failed: {log_error}")
        if error is not None:
            raise error
        return result

    def _payload(self, stage: str, content: dict, schema: dict) -> dict:
        if stage not in STAGES:
            raise ValueError(f"unknown annotation stage: {stage}")
        if not isinstance(content, dict) or not isinstance(schema, dict):
            raise TypeError("content and schema must be JSON objects")
        return {
            "model": self.config["model"],
            "reasoning": {"effort": self.config["reasoning_effort"]},
            "instructions": self.prompts[stage],
            "input": [
                {
                    "role": "user",
                    "content": json.dumps(
                        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ),
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"fineweb_qa_{stage}",
                    "strict": True,
                    "schema": schema,
                }
            },
            "max_output_tokens": self.config["max_output_tokens"][stage],
            "store": False,
            "stream": False,
        }

    def _cached_call(self, payload: dict, schema: dict, request_id: str, record: dict) -> dict:
        folder = self.cache_root / request_id[:2] / request_id
        folder.mkdir(parents=True, exist_ok=True)
        with _local_lock(folder):
            with (folder / "request.lock").open("a+b") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    return self._locked_call(folder, payload, schema, record)
                except DocumentAnnotationError:
                    raise
                except Exception:
                    self.stop.set()
                    raise
                finally:
                    try:
                        attempts = self._previous_attempts(folder)
                        record["attempts_total"] = len(attempts)
                        record["usage_by_attempt"] = {
                            str(a["attempt"]): a.get("usage") for a in attempts
                        }
                        usages = [a["usage"] for a in attempts if a.get("usage") is not None]
                        if usages:
                            record["usage"] = {
                                key: sum(u[key] for u in usages if type(u.get(key)) is int)
                                for key in ("input_tokens", "output_tokens")
                            }
                    finally:
                        fcntl.flock(handle, fcntl.LOCK_UN)

    def _locked_call(self, folder: Path, payload: dict, schema: dict, record: dict) -> dict:
        if self.stop.is_set():
            raise InterruptedError("annotation stopped after a fatal error")
        request_file = folder / "request.json"
        raw_file = folder / "response.raw.json"
        parsed_file = folder / "parsed.json"
        if request_file.exists():
            if json.loads(request_file.read_text()) != payload:
                raise ValueError("cached request payload differs from its identity")
        else:
            _atomic_write_json(request_file, payload)
        content = json.loads(payload["input"][0]["content"])
        if parsed_file.exists():
            parsed = json.loads(parsed_file.read_text())
            try:
                validate_output(parsed, record["stage"], content, schema)
            except ResponseContractError:
                # Old successful caches may predate request-level coverage checks.
                if not raw_file.exists():
                    raise RuntimeError("invalid parsed cache has no raw response for recovery")
            else:
                record.update(cache_hit=True, cache_source="parsed")
                return parsed
        if raw_file.exists():
            record.update(cache_hit=True, cache_source="raw")
            parsed = self._consume_response(folder, schema, content, record)
            if parsed is not None:
                return parsed

        previous = self._previous_attempts(folder)
        record["attempts_total"] = len(previous)
        if previous:
            last = previous[-1]
            if last["status"] in {"in_flight", "response_received"}:
                raise RuntimeError("request outcome is uncertain and its raw cache is missing")
            if last["status"] == "failed" and not last["retryable"]:
                raise RuntimeError("cached request has a non-retryable failure")
            if last["status"] not in {"started", "failed", "invalid_response"}:
                raise ValueError("cached attempt has an unknown status")
        if len(previous) >= self.config["max_attempts"]:
            if previous[-1]["status"] == "invalid_response":
                raise AnnotationContractError(
                    record["stage"],
                    record["request_id"],
                    str(folder / f"response.attempt-{len(previous)}.raw.json"),
                    previous[-1]["error"],
                )
            raise RuntimeError("request retry budget exhausted")
        wire_payload = copy.deepcopy(payload)
        wire_payload["text"]["format"]["schema"] = schema

        for attempt in range(len(previous) + 1, self.config["max_attempts"] + 1):
            if self.stop.is_set():
                raise InterruptedError("annotation stopped before a retry")
            attempt_file = folder / f"attempt-{attempt}.json"
            metadata = {
                "attempt": attempt,
                "started_at": _now(),
                "status": "started",
                "response_schema": schema,
            }
            _atomic_write_json(attempt_file, metadata)
            record["attempts_total"] = attempt
            while not self._slots.acquire(timeout=0.1):
                if self.stop.is_set():
                    raise InterruptedError("annotation stopped before a request")
            try:
                if self.stop.is_set():
                    raise InterruptedError("annotation stopped before a request")
                metadata["status"] = "in_flight"
                _atomic_write_json(attempt_file, metadata)
                record["network_attempts"] += 1
                record["network_attempt_numbers"].append(attempt)
                sent_at = time.perf_counter()
                try:
                    status, raw = self._send(wire_payload)
                except HTTPError as exc:
                    elapsed = time.perf_counter() - sent_at
                    record["network_seconds"] += elapsed
                    retryable = exc.code in {408, 429} or 500 <= exc.code <= 599
                    metadata.update(
                        status="failed",
                        http_status=exc.code,
                        retryable=retryable,
                        elapsed_seconds=elapsed,
                        error=f"HTTPError: {exc}",
                    )
                    _atomic_write_json(attempt_file, metadata)
                    if not retryable or attempt == self.config["max_attempts"]:
                        raise RuntimeError(f"request failed with HTTP {exc.code}") from exc
                    delay = self._retry_delay(attempt, exc.headers.get("Retry-After"))
                except (URLError, TimeoutError, OSError) as exc:
                    elapsed = time.perf_counter() - sent_at
                    record["network_seconds"] += elapsed
                    metadata.update(
                        status="failed",
                        retryable=True,
                        elapsed_seconds=elapsed,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    _atomic_write_json(attempt_file, metadata)
                    if attempt == self.config["max_attempts"]:
                        raise RuntimeError("request failed after temporary network errors") from exc
                    delay = self._retry_delay(attempt, None)
                else:
                    elapsed = time.perf_counter() - sent_at
                    record["network_seconds"] += elapsed
                    metadata.update(
                        status="response_received", http_status=status, elapsed_seconds=elapsed
                    )
                    _atomic_write_json(attempt_file, metadata)
                    _atomic_write(raw_file, raw)
                    parsed = self._consume_response(folder, schema, content, record)
                    if parsed is not None:
                        return parsed
                    if attempt == self.config["max_attempts"]:
                        rejected = json.loads(attempt_file.read_text())
                        raise AnnotationContractError(
                            record["stage"],
                            record["request_id"],
                            str(folder / f"response.attempt-{attempt}.raw.json"),
                            rejected["error"],
                        )
                    delay = 0
            finally:
                self._slots.release()
            if self.stop.wait(delay):
                raise InterruptedError("annotation stopped before a retry")
        raise RuntimeError("request retry budget exhausted")

    def _consume_response(
        self, folder: Path, schema: dict, content: dict, record: dict
    ) -> dict | None:
        """Archive rejected outputs before clearing their cache; preserve the attempt budget."""
        raw_path = folder / "response.raw.json"
        parsed_path = folder / "parsed.json"
        previous = self._previous_attempts(folder)
        metadata = previous[-1]
        attempt = metadata["attempt"]
        attempt_path = folder / f"attempt-{attempt}.json"
        raw = raw_path.read_bytes()
        record["usage"] = None
        try:
            parsed, usage = self._parse_response(raw, schema, record, raw_path, content)
        except ResponseContractError as error:
            _atomic_write(folder / f"response.attempt-{attempt}.raw.json", raw)
            metadata.update(status="invalid_response", error=str(error), usage=record["usage"])
            _atomic_write_json(attempt_path, metadata)
            parsed_path.unlink(missing_ok=True)
            raw_path.unlink()
            return None
        except Exception:
            metadata["usage"] = record["usage"]
            _atomic_write_json(attempt_path, metadata)
            raise
        metadata["usage"] = usage
        _atomic_write_json(attempt_path, metadata)
        _atomic_write_json(parsed_path, parsed)
        return parsed

    def _send(self, payload: dict) -> tuple[int, bytes]:
        request = Request(
            self.config["endpoint"],
            data=_payload_bytes(payload),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with build_opener(ProxyHandler({})).open(
            request, timeout=self.config["timeout_seconds"]
        ) as response:
            return response.status, response.read()

    def _parse_response(
        self, raw: bytes, schema: dict, record: dict, raw_path: Path, content: dict
    ) -> tuple[dict, dict | None]:
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Responses result was not completed")
        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            raise ValueError("Responses usage must be an object when present")
        record["usage"] = usage
        if data.get("model") != self.config["model"]:
            raise ValueError("Responses model differs from selected model")
        if (data.get("reasoning") or {}).get("effort") != self.config["reasoning_effort"]:
            raise ValueError("Responses reasoning effort differs from selected effort")
        if data.get("status") == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason")
            if reason == "content_filter":
                raise ContentFilteredError(record["stage"], record["request_id"], str(raw_path))
            if reason == "max_output_tokens":
                raise ResponseContractError("Responses result was truncated: max_output_tokens")
        if data.get("status") != "completed":
            raise ValueError("Responses result was not completed")
        output = data.get("output")
        if not isinstance(output, list):
            raise ValueError("Responses output is missing")
        pieces = [
            part["text"]
            for item in output
            if item["type"] == "message" and item.get("phase") in (None, "final_answer")
            for part in item["content"]
            if part["type"] == "output_text"
        ]
        if not pieces or any(not isinstance(piece, str) for piece in pieces):
            raise ValueError("Responses final output_text is missing")
        try:
            parsed = json.loads("".join(pieces))
        except json.JSONDecodeError as error:
            raise ResponseContractError(f"model output is not valid JSON: {error}") from error
        validate_output(parsed, record["stage"], content, schema)
        return parsed, usage

    def _previous_attempts(self, folder: Path) -> list[dict]:
        paths = sorted(folder.glob("attempt-*.json"), key=lambda p: int(p.stem.split("-")[1]))
        if [int(p.stem.split("-")[1]) for p in paths] != list(range(1, len(paths) + 1)):
            raise ValueError("cached attempts are not contiguous")
        return [json.loads(path.read_text()) for path in paths]

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None) -> int:
        header_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 0
        return min(60, max(5 * 2 ** (attempt - 1), header_seconds))

    def _append_log(self, record: dict) -> None:
        path = self.batch_root / "requests.jsonl"
        line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
        with _local_lock(path):
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                remaining = memoryview(line)
                while remaining:
                    remaining = remaining[os.write(fd, remaining) :]
                os.fsync(fd)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)


def locate_candidates(
    segment_text: str,
    generated: dict,
    trajectory_id: str,
    segment_index: int,
    segment_offset: int,
    candidate_limit: int,
    max_answer_chars: int,
    round_index: int,
) -> tuple[list[dict], list[dict]]:
    """Locate generated quotes and answers in unchanged trajectory characters."""
    if not isinstance(generated, dict) or set(generated) != {"qas", "skip_reason"}:
        raise ValueError("generation object schema mismatch")
    qas, skip_reason = generated["qas"], generated["skip_reason"]
    if not isinstance(qas, list) or len(qas) > candidate_limit or not isinstance(skip_reason, str):
        raise ValueError("invalid generated QA count or skip reason")
    if bool(qas) == bool(skip_reason.strip()):
        raise ValueError("skip reason must be present exactly when no QA was generated")
    if segment_index < 0 or segment_offset < 0:
        raise ValueError("segment index and offset must be nonnegative")
    segment_id = f"seg{segment_index}"
    valid, rejected = [], []
    expected_fields = {"fact_statement", "question", "answer", "evidence_quote"}
    for index, candidate in enumerate(qas):
        qa_id = f"{trajectory_id}:{segment_id}:round{round_index}:qa{index}"
        try:
            if not isinstance(candidate, dict) or set(candidate) != expected_fields:
                raise ValueError("QA fields mismatch")
            if any(not isinstance(value, str) or not value.strip() for value in candidate.values()):
                raise ValueError("QA fields must be non-empty strings")
            answer, quote = candidate["answer"], candidate["evidence_quote"]
            if len(answer) > max_answer_chars:
                raise ValueError("answer exceeds character limit")
            first_quote = segment_text.find(quote)
            if first_quote < 0 or segment_text.find(quote, first_quote + 1) >= 0:
                raise ValueError("evidence quote must occur exactly once in segment")
            if answer not in quote:
                raise ValueError("answer is absent from evidence quote")
            quote_start = segment_offset + first_quote
            answer_start = quote_start + quote.index(answer)
            valid.append(
                {
                    "qa_id": qa_id,
                    "segment_id": segment_id,
                    "fact_statement": candidate["fact_statement"],
                    "question": candidate["question"],
                    "answer": answer,
                    "evidence_quote": quote,
                    "evidence_char_span": [quote_start, quote_start + len(quote)],
                    "answer_char_span": [answer_start, answer_start + len(answer)],
                }
            )
        except ValueError as exc:
            rejected.append(
                {"qa_id": qa_id, "candidate_index": index, "reason": str(exc), "qa": candidate}
            )
    return valid, rejected
