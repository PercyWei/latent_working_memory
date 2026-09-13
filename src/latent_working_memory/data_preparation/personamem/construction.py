"""Resumable PersonaMem factual QA annotation with an explicit pilot checkpoint."""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import fcntl
import json
from pathlib import Path
import signal
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from latent_working_memory.data_preparation.personamem.common import (
    GENERATE as BASE_GENERATE,
    QA_SCHEMA,
    STRING,
    VERIFY as BASE_VERIFY,
    object_schema,
    now,
)
from latent_working_memory.data_preparation.personamem.sources import prepare


GENERATE = (
    BASE_GENERATE
    + """
Additional requirements for this dataset:
1. Give the smallest CONTIGUOUS COMPLETE-MESSAGE range that supports the question
and answer, including any message needed to identify a quoted passage, speaker,
negation or time. Specify its first and last message IDs (inclusive); both must
belong to the excerpt and enclose answer_message_id. Do not include an assistant
rewrite if the original user message alone fully supports the question.
2. Ask about what was said or written in the specific exchange/event/passage.
For mutable facts, explicitly anchor the question to the described statement,
event or draft, so later changes do not change the answer to this historical
question. Never ask an unqualified question about the user's current state.
3. Every field must be non-empty. If there is no calendar date, temporal_scope
should describe the exchange, event or original draft; do not invent a date.
4. Generic assertions in an assistant explanation are not eligible facts, even
if they contain names or locations. Concrete user/third-party statements and
facts explicitly attributed to drafted text are eligible. Do not infer biography
from a writing sample or turn the assistant's suggestions into user preferences.
5. Select different facts, not two phrasings of the same relation. Short answers
must be sufficient but avoid unnecessary leading clauses and conjunctions.
6. This is an English-question, English-answer dataset. Select an English answer
verbatim from the evidence; original proper names and titles are allowed. Do not
pair an English question with an ordinary answer phrase in another language.
If only non-English evidence supports the fact, skip it instead of translating
the answer and losing its original span. Ask a question whose requested scope
matches the full answer, e.g. 'which city' for a city, not a broad 'where' if
both city and specific venue could be required. Include every requested item.
Names of dishes, foods and cultural items used in otherwise English prose are
also eligible original names (e.g. batata poha, paprikás, al pastor, pan dulce).
"""
)
VERIFY = (
    BASE_VERIFY
    + """
Also check that the specified evidence range alone, provided separately, contains
all required context. The question must ask about the stated exchange/event/draft
rather than an unqualified current fact; mark revisit_safe true only when later
updates would not change the answer to this historically anchored question.
Reject generic facts merely asserted by the assistant. Reject duplicate facts
within the proposal, retaining at most one question per fact. A valid decision
requires accepted=true AND revisit_safe=true. Do not repair or invent facts.
Both questions and ordinary answer phrases must be English, except original
proper names/titles. Reject non-English common-noun or descriptive answers.
Check that the short answer covers ALL requested items, properties and scope;
do not accept an incomplete substring just because it appears in the evidence.
Names of dishes, foods and cultural items in English prose may retain their
original names; these are not ineligible translated descriptive answers.
"""
)
QA_PROPERTIES = copy.deepcopy(QA_SCHEMA["properties"])
QA_PROPERTIES.update(evidence_start_message_id=STRING, evidence_end_message_id=STRING)
GEN_SCHEMA = object_schema(
    {
        "qas": {"type": "array", "items": object_schema(QA_PROPERTIES), "maxItems": 2},
        "skip_reason": STRING,
    }
)
VER_SCHEMA = object_schema(
    {
        "decisions": {
            "type": "array",
            "items": object_schema(
                {
                    "qa_id": STRING,
                    "accepted": {"type": "boolean"},
                    "revisit_safe": {"type": "boolean"},
                    "reason": STRING,
                }
            ),
        }
    }
)


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def normalize(text):
    return " ".join(text.casefold().split()).strip(" .?!")


class AnnotationClient:
    def __init__(self, config, root):
        self.config, self.root = config, root
        self.lock = threading.Lock()
        self.active, self.peak = 0, 0
        self.stop = threading.Event()

    def call(self, key, stage, instructions, content, schema):
        folder = self.root / "requests" / key
        folder.mkdir(parents=True, exist_ok=True)
        payload = dict(
            model=self.config["model"],
            reasoning={"effort": self.config["reasoning_effort"]},
            instructions=instructions,
            input=[{"role": "user", "content": json.dumps(content, ensure_ascii=False)}],
            text={
                "format": dict(
                    type="json_schema", name=f"personamem_{stage}", strict=True, schema=schema
                )
            },
            max_output_tokens=self.config["max_output_tokens"],
            store=False,
            stream=False,
        )
        request_file, parsed_file = (
            folder / f"{stage}.request.json",
            folder / f"{stage}.parsed.json",
        )
        if request_file.exists():
            if json.loads(request_file.read_text()) != payload:
                raise ValueError(f"request changed on resume: {key}/{stage}")
            if parsed_file.exists():
                return json.loads(parsed_file.read_text())
        else:
            save(request_file, payload)
        # Attempt records are append-only across resumes; accepted outputs are never regenerated.
        previous = list(folder.glob(f"{stage}.attempt*.meta.json"))
        for attempt in range(len(previous), self.config["max_attempts"]):
            if self.stop.is_set():
                raise InterruptedError("construction stop requested")
            started = time.perf_counter()
            record = dict(key=key, stage=stage, attempt=attempt + 1, started_at=now())
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                record["active_requests_at_start"] = self.active
            retryable = False
            try:
                request = Request(
                    self.config["endpoint"],
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with build_opener(ProxyHandler({})).open(
                    request, timeout=self.config["timeout_seconds"]
                ) as response:
                    record["http_status"] = response.status
                    raw = response.read().decode()
                (folder / f"{stage}.attempt{attempt + 1}.response.json").write_text(raw)
                data = json.loads(raw)
                record.update(
                    model=data.get("model"),
                    reasoning=data.get("reasoning"),
                    response_status=data.get("status"),
                    usage=data.get("usage"),
                    response_id=data.get("id"),
                )
                if data.get("status") != "completed":
                    raise ValueError(f"response not completed: {data.get('status')}")
                if data.get("model") != self.config["model"]:
                    raise ValueError("response model differs from selected model")
                if (data.get("reasoning") or {}).get("effort") != self.config["reasoning_effort"]:
                    raise ValueError("response reasoning effort differs from selected effort")
                text = "".join(
                    p["text"]
                    for item in data["output"]
                    if item["type"] == "message" and item.get("phase") in {None, "final_answer"}
                    for p in item["content"]
                    if p["type"] == "output_text"
                )
                parsed = json.loads(text)
                save(parsed_file, parsed)
                record["ok"] = True
                return parsed
            except HTTPError as error:
                record.update(
                    ok=False, http_status=error.code, error=error.read().decode(errors="replace")
                )
                retryable = error.code in {408, 429, 500, 502, 503, 504}
                delay = max(
                    5 * 2**attempt,
                    int(error.headers.get("Retry-After", "0"))
                    if error.headers.get("Retry-After", "0").isdigit()
                    else 0,
                )
            except (URLError, TimeoutError, OSError) as error:
                record.update(ok=False, error=f"{type(error).__name__}: {error}")
                retryable, delay = True, 5 * 2**attempt
            except (ValueError, KeyError, TypeError) as error:
                record.update(ok=False, error=f"{type(error).__name__}: {error}")
            finally:
                record.update(finished_at=now(), elapsed_seconds=time.perf_counter() - started)
                save(folder / f"{stage}.attempt{attempt + 1}.meta.json", record)
                with self.lock:
                    self.active -= 1
                    with (self.root / "requests.jsonl").open("a") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not retryable or attempt + 1 == self.config["max_attempts"]:
                raise RuntimeError(f"request failed: {key}/{stage}: {record.get('error')}")
            if self.stop.wait(min(delay, 60)):
                raise InterruptedError("construction stop requested")
        raise RuntimeError(f"retry budget exhausted: {key}/{stage}")


def locate(candidate, output):
    if not isinstance(output, dict) or set(output) != {"qas", "skip_reason"}:
        raise ValueError("generation object schema mismatch")
    qas, reason = output["qas"], output["skip_reason"]
    if not isinstance(qas, list) or len(qas) > 2 or not isinstance(reason, str):
        raise ValueError("invalid QA list")
    if bool(qas) == bool(reason.strip()):
        raise ValueError("inconsistent empty QA/skip reason")
    messages = candidate["messages"]
    indices = {m["message_id"]: i for i, m in enumerate(messages)}
    valid, rejected, seen = [], [], set()
    for i, qa in enumerate(qas):
        try:
            if not isinstance(qa, dict) or set(qa) != set(QA_PROPERTIES):
                raise ValueError("QA field schema mismatch")
            if any(not isinstance(v, str) or not v.strip() for v in qa.values()):
                raise ValueError("empty QA field")
            for field in ("subject_type", "fact_type"):
                if qa[field] not in QA_PROPERTIES[field]["enum"]:
                    raise ValueError(f"invalid {field}")
            a, b = indices[qa["evidence_start_message_id"]], indices[qa["evidence_end_message_id"]]
            answer_index = indices[qa["answer_message_id"]]
            if not a <= answer_index <= b:
                raise ValueError("answer outside evidence range")
            text = messages[answer_index]["content"]
            quote, answer = qa["evidence_quote"], qa["answer"]
            if text.count(quote) != 1 or quote.count(answer) != 1:
                raise ValueError("quote/answer not uniquely located")
            if len(answer.split()) > 20:
                raise ValueError("answer longer than 20 words")
            start = text.index(quote) + quote.index(answer)
            end = start + len(answer)
            fact = (qa["answer_message_id"], start, end)
            question = normalize(qa["question"])
            if fact in seen or question in seen:
                raise ValueError("duplicate answer span or question")
            seen.update((fact, question))
            valid.append(
                dict(
                    qa,
                    qa_id=f"{candidate['candidate_id']}:qa{i}",
                    history_id=candidate["history_id"],
                    persona_id=candidate["persona_id"],
                    split=candidate["split"],
                    candidate_id=candidate["candidate_id"],
                    source_rows=candidate["source_rows"],
                    answer_char_start=start,
                    answer_char_end_exclusive=end,
                    evidence_message_start=candidate["message_start"] + a,
                    evidence_message_end_exclusive=candidate["message_start"] + b + 1,
                    evidence_messages=messages[a : b + 1],
                    revisit_validity="pending_semantic_verification",
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            rejected.append(dict(index=i, reason=str(error), qa=qa))
    return valid, rejected


def annotate(client, candidate, index):
    key = f"candidate-{index:05d}"
    path = client.root / "results" / f"{key}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["candidate_id"] != candidate["candidate_id"]:
            raise ValueError("candidate identity changed on resume")
        if result["ok"]:
            return result
    started = time.perf_counter()
    result = dict(
        candidate_id=candidate["candidate_id"],
        persona_id=candidate["persona_id"],
        split=candidate["split"],
        candidate_index=index,
        started_at=now(),
        accepted=[],
    )
    try:
        generated = client.call(
            key, "generate", GENERATE, {"messages": candidate["messages"]}, GEN_SCHEMA
        )
        result["generated"] = generated
        valid, rejected = locate(candidate, generated)
        result.update(program_valid=len(valid), program_rejections=rejected)
        if valid:
            verification = client.call(
                key, "verify", VERIFY, {"excerpt": candidate["messages"], "qas": valid}, VER_SCHEMA
            )
            decisions = verification["decisions"]
            ids = [d["qa_id"] for d in decisions]
            if len(ids) != len(set(ids)) or set(ids) != {q["qa_id"] for q in valid}:
                raise ValueError("verifier IDs mismatch")
            if any(
                set(d) != {"qa_id", "accepted", "revisit_safe", "reason"}
                or type(d["accepted"]) is not bool
                or type(d["revisit_safe"]) is not bool
                or not isinstance(d["reason"], str)
                or not d["reason"].strip()
                for d in decisions
            ):
                raise ValueError("verifier decision schema mismatch")
            by_id = {d["qa_id"]: d for d in decisions}
            result["verification"] = verification
            for qa in valid:
                decision = by_id[qa["qa_id"]]
                if decision["accepted"] and decision["revisit_safe"]:
                    result["accepted"].append(
                        dict(
                            qa,
                            revisit_validity="historically_anchored",
                            verification_reason=decision["reason"],
                        )
                    )
        result["ok"] = True
    except Exception as error:
        result.update(ok=False, error=f"{type(error).__name__}: {error}")
    result.update(finished_at=now(), elapsed_seconds=time.perf_counter() - started)
    save(path, result)
    return result


def collect(config):
    root = Path(config["artifacts_dir"])
    results = [json.loads(p.read_text()) for p in sorted((root / "results").glob("*.json"))]
    accepted, duplicate_log, content_rejections, seen = [], [], [], {}
    pilot_review_file = root / "pilot_review.json"
    manual_review_file = root / "manual_review.json"
    manual_keeps = set()
    if manual_review_file.exists():
        manual_keeps = {
            d["qa_id"]
            for d in json.loads(manual_review_file.read_text())["decisions"]
            if d["accepted"]
        }
    pilot_rejections = {}
    if pilot_review_file.exists():
        pilot_rejections = {
            d["qa_id"]: d["reason"]
            for d in json.loads(pilot_review_file.read_text())["decisions"]
            if not d["accepted"] and d["qa_id"] not in manual_keeps
        }
    for result in results:
        for qa in result["accepted"]:
            if qa["qa_id"] in pilot_rejections:
                content_rejections.append(
                    dict(qa_id=qa["qa_id"], reason=pilot_rejections[qa["qa_id"]])
                )
                continue
            if "keep existing turns word-by-word identical" in normalize(qa["evidence_quote"]):
                content_rejections.append(
                    dict(
                        qa_id=qa["qa_id"],
                        reason="QA targets a repeated conversation-construction template",
                    )
                )
                continue
            keys = [
                (
                    qa["history_id"],
                    qa["answer_message_id"],
                    qa["answer_char_start"],
                    qa["answer_char_end_exclusive"],
                ),
                (qa["history_id"], normalize(qa["question"])),
            ]
            duplicate = next((seen[k] for k in keys if k in seen), None)
            if duplicate:
                duplicate_log.append(
                    dict(
                        qa_id=qa["qa_id"],
                        duplicate_of=duplicate,
                        reason="same answer span or normalized question",
                    )
                )
                continue
            accepted.append(qa)
            for key in keys:
                seen[key] = qa["qa_id"]
    summary = dict(
        updated_at=now(),
        processed_candidates=len(results),
        completed_candidates=sum(r["ok"] for r in results),
        failed_candidates=sum(not r["ok"] for r in results),
        generated_qas=sum(len(r.get("generated", {}).get("qas", [])) for r in results),
        program_rejections=sum(len(r.get("program_rejections", [])) for r in results),
        semantic_rejections=sum(
            not (d["accepted"] and d["revisit_safe"])
            for r in results
            for d in r.get("verification", {}).get("decisions", [])
        ),
        duplicates=len(duplicate_log),
        content_rejections=len(content_rejections),
        accepted_qas=len(accepted),
        by_split=dict(Counter(q["split"] for q in accepted)),
        by_subject_type=dict(Counter(q["subject_type"] for q in accepted)),
        productive_users=len({q["persona_id"] for q in accepted}),
    )
    save(root / "progress.json", summary)
    save(root / "exact_duplicates.json", duplicate_log)
    save(root / "content_rejections.json", content_rejections)
    dataset = Path(config["dataset_dir"])
    path = dataset / "qas.provisional.jsonl"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for qa in accepted:
            handle.write(json.dumps(qa, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return summary


def run(config, stage, lock_acquired=False):
    if not 1 <= config["concurrency"] <= 8:
        raise ValueError("concurrency must be between 1 and 8")
    root = Path(config["artifacts_dir"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    with (root / "construction.lock").open("w") as lock:
        if not lock_acquired:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = root / "config.json"
        if snapshot.exists() and json.loads(snapshot.read_text()) != config:
            raise ValueError("construction config changed; use another run directory")
        save(snapshot, config)
        candidates = json.loads((Path(config["dataset_dir"]) / "candidates.json").read_text())
        if stage == "full":
            gate = json.loads((root / "pilot_gate.json").read_text())
            if not gate["passed"]:
                raise ValueError("pilot gate has not passed")
            selected = list(enumerate(candidates))
        else:
            selected = list(enumerate(candidates))[: config["pilot_candidates"]]
            if any(c["split"] != "train" for _, c in selected):
                raise ValueError("pilot must use training users only")
        client = AnnotationClient(config, root)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: client.stop.set())
        pending = []
        for index, c in selected:
            path = root / "results" / f"candidate-{index:05d}.json"
            if not path.exists() or not json.loads(path.read_text())["ok"]:
                pending.append((index, c))
        save(
            root / f"{stage}.launch.json",
            dict(
                started_at=now(),
                stage=stage,
                selected=len(selected),
                pending=len(pending),
                concurrency=config["concurrency"],
            ),
        )
        done, started = 0, time.perf_counter()
        # Keep at most one active candidate per worker, refilling only completed slots.
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
            cursor, failures = 0, 0
            futures = set()
            while futures or (cursor < len(pending) and not client.stop.is_set()):
                while (
                    len(futures) < config["concurrency"]
                    and cursor < len(pending)
                    and not client.stop.is_set()
                ):
                    i, c = pending[cursor]
                    futures.add(pool.submit(annotate, client, c, i))
                    cursor += 1
                completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    result = future.result()
                    failures += not result["ok"]
                    done += 1
                    print(
                        json.dumps(
                            dict(
                                done=done,
                                pending_total=len(pending),
                                candidate=result["candidate_index"],
                                ok=result["ok"],
                                accepted=len(result["accepted"]),
                                seconds=round(result["elapsed_seconds"], 2),
                            )
                        ),
                        flush=True,
                    )
                if done % config["concurrency"] == 0 or not futures:
                    progress = collect(config)
                    progress.update(
                        elapsed_this_run=time.perf_counter() - started,
                        observed_peak_requests=client.peak,
                    )
                    save(root / "progress.json", progress)
                if failures >= 3:
                    print(
                        "Stopping after three failed candidates; inspect saved errors.", flush=True
                    )
                    client.stop.set()
        progress = collect(config)
        save(
            root / f"{stage}.summary.json",
            dict(
                progress,
                finished_at=now(),
                interrupted=client.stop.is_set(),
                elapsed_this_run=time.perf_counter() - started,
                observed_peak_requests=client.peak,
            ),
        )
        print(json.dumps(progress, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "pilot", "full", "collect"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "prepare":
        print(json.dumps(prepare(config), ensure_ascii=False, indent=2))
    elif args.stage == "collect":
        print(json.dumps(collect(config), ensure_ascii=False, indent=2))
    else:
        run(config, args.stage)


if __name__ == "__main__":
    main()
