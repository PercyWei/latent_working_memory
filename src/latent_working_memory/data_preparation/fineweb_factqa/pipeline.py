"""Generate, verify and assemble resumable batches of FineWeb factual QA."""

from __future__ import annotations

import copy
import fcntl
import json
import os
import platform
import time
import uuid
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from latent_working_memory.data_preparation.fineweb_factqa.annotation import (
    DOCUMENT_REVIEW_SCHEMA,
    GENERATE_SCHEMA,
    VERIFY_SCHEMA,
    STAGES,
    AnnotationClient,
    DocumentAnnotationError,
    locate_candidates,
)
from latent_working_memory.data_preparation.fineweb_factqa.assembly import (
    qa_quotas,
    assemble_document,
    validate_trajectory,
)
from latent_working_memory.data_preparation.fineweb_factqa.sources import prepare_selection
from latent_working_memory.data_preparation.fineweb_factqa.storage import (
    save_json as _save_json,
    load_json as _load_json,
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _root(config: dict) -> Path:
    return Path(config["artifacts_dir"])


def _prompt_texts(config: dict) -> dict[str, str]:
    max_answer_chars = config["qa"]["max_answer_chars"]
    if type(max_answer_chars) is not int or max_answer_chars < 1:
        raise ValueError("qa.max_answer_chars must be a positive integer")
    required = set(STAGES)
    if set(config["prompts"]) != required:
        raise ValueError("FineWeb QA prompts must name generate, verify and document_review")
    texts = {
        name: Path(path).read_text(encoding="utf-8") for name, path in config["prompts"].items()
    }
    if any(not value.strip() for value in texts.values()):
        raise ValueError("FineWeb QA prompts must not be empty")
    placeholder = "{{max_answer_chars}}"
    if placeholder not in texts["generate"]:
        raise ValueError("generate prompt requires {{max_answer_chars}}")
    texts["generate"] = texts["generate"].replace(placeholder, str(max_answer_chars))
    return texts


def _without_endpoint(config: dict) -> dict:
    frozen = copy.deepcopy(config)
    frozen["annotation"] = {
        key: value for key, value in frozen["annotation"].items() if key != "endpoint"
    }
    return frozen


def _batch(config: dict, create: bool = False) -> tuple[Path, dict[str, str]]:
    for key in ("max_supplement_rounds", "supplement_surplus"):
        if type(config["qa"][key]) is not int or config["qa"][key] < 0:
            raise ValueError(f"qa.{key} must be a nonnegative integer")
    if type(config["batch_index"]) is not int or config["batch_index"] < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    root = _root(config)
    config_path, prompts_path = root / "config.json", root / "prompts.json"
    prompts = _prompt_texts(config)
    if create and not config_path.exists() and not prompts_path.exists():
        root.mkdir(parents=True, exist_ok=True)
        _save_json(config_path, config)
        _save_json(prompts_path, prompts)
    if not config_path.exists() or not prompts_path.exists():
        raise FileNotFoundError("run prepare to freeze the batch configuration and prompts")
    if _without_endpoint(_load_json(config_path)) != _without_endpoint(config):
        raise ValueError("batch configuration changed; start a new batch")
    if _load_json(prompts_path) != prompts:
        raise ValueError("batch prompts changed; start a new batch")
    return root, prompts


def _selection(root: Path) -> dict:
    selection = _load_json(root / "selection.json")
    if not isinstance(selection.get("documents"), list):
        raise ValueError("selection.json does not contain documents")
    return selection


def prepare(config: dict) -> dict:
    """Freeze a text-only source selection after the local Parquet download."""
    root, _ = _batch(config, create=True)
    path = root / "selection.json"
    if path.exists():
        selection = _selection(root)
    else:
        selection = prepare_selection(config)
        _save_json(path, selection)
    return {
        "stage": "prepare",
        "selection": str(path),
        "frozen_trajectories": len(selection["documents"]),
        "source_pool_id": selection["source_pool_id"],
        "batch_index": selection["batch_index"],
        "ranges": selection["ranges"],
        "statistics": selection["statistics"],
    }


def _document_result_path(root: Path, index: int) -> Path:
    return root / "documents" / f"doc-{index:03d}.json"


def _document_failures(root: Path, documents: list[dict]) -> dict[int, dict]:
    failures = {}
    for path in sorted((root / "failed-documents").glob("doc-*.json")):
        failure = _load_json(path)
        index = failure["document_index"]
        if failure["trajectory_id"] != documents[index]["trajectory_id"]:
            raise ValueError("failed document differs from frozen selection")
        failures[index] = failure
    return failures


def _failure_counts(failures: dict[int, dict]) -> dict[str, int]:
    return {
        f"{reason}_trajectories": sum(f["reason"] == reason for f in failures.values())
        for reason in ("content_filtered", "annotation_contract_failed")
    }


def _record_document_failure(
    root: Path, index: int, document: dict, error: DocumentAnnotationError
) -> dict:
    path = root / "failed-documents" / f"doc-{index:03d}.json"
    if path.exists():
        return _load_json(path)
    failure = {
        "document_index": index,
        "trajectory_id": document["trajectory_id"],
        "reason": error.reason,
        "error": str(error),
        "phase": "annotate",
        "stage": error.stage,
        "request_id": error.request_id,
        "raw_response_path": error.raw_response_path,
        "at": _now(),
    }
    _save_json(path, failure)
    return failure


def _run_round(
    document: dict,
    result: dict,
    config: dict,
    client: AnnotationClient,
) -> None:
    """Append one generation/filtering round, keeping stable IDs for every attempt."""
    round_index = len(result["rounds"])
    before = result["assembly"]
    previous_candidate_ids = {qa["qa_id"] for qa in result["candidates"]}
    before_ids = {qa["qa_id"] for qa in before["eligible_qas"]} if before else set()
    limits = (
        {f"seg{i}": n for i, n in enumerate(qa_quotas(len(document["segments"]))[0])}
        if before is None
        else {
            item["segment_id"]: item["missing_total"] + config["qa"]["supplement_surplus"]
            for item in before["shortfalls"]
        }
    )
    context = {
        "trajectory_id": document["trajectory_id"],
        "round_index": round_index,
        "phase": "annotate",
    }
    prior_rejections = {
        item["qa_id"]: item["reason"] for item in result["review_decisions"] if not item["accepted"]
    }
    segments = []
    started = time.perf_counter()
    for segment_index, segment in enumerate(document["segments"]):
        segment_id = segment["segment_id"]
        if segment_id not in limits:
            continue
        if client.stop.is_set():
            raise InterruptedError("annotation stopped after a fatal error")
        start, end = segment["char_span"]
        segment_text = document["text"][start:end]
        rejected = []
        for previous in result["rounds"]:
            for entry in previous["segments"]:
                if entry["segment_id"] != segment_id:
                    continue
                rejected.extend(
                    {"qa": item["qa"], "reason": item["reason"]}
                    for item in entry["program_rejections"]
                )
                rejected_ids = {
                    item["qa_id"]: item["reason"]
                    for item in entry["verification_decisions"]
                    if not item["accepted"]
                }
                rejected.extend(
                    {"qa": qa, "reason": rejected_ids[qa["qa_id"]]}
                    for qa in result["candidates"]
                    if qa["qa_id"] in rejected_ids
                )
        rejected.extend(
            {"qa": qa, "reason": prior_rejections[qa["qa_id"]]}
            for qa in result["local_candidates"]
            if qa["segment_id"] == segment_id and qa["qa_id"] in prior_rejections
        )
        generated = client.call(
            "generate",
            {
                **context,
                "segment_id": segment_id,
                "segment_text": segment_text,
                "candidate_limit": limits[segment_id],
                "existing_facts": [qa["fact_statement"] for qa in before["eligible_qas"]]
                if before
                else [],
                "rejected_candidates": rejected,
            },
            GENERATE_SCHEMA,
        )
        valid, program_rejections = locate_candidates(
            segment_text,
            generated,
            document["trajectory_id"],
            segment_index,
            start,
            limits[segment_id],
            config["qa"]["max_answer_chars"],
            round_index,
        )
        decisions = []
        if valid:
            verified = client.call(
                "verify",
                {
                    **context,
                    "segment_id": segment_id,
                    "segment_text": segment_text,
                    "qas": [
                        {key: qa[key] for key in ("qa_id", "question", "answer", "evidence_quote")}
                        for qa in valid
                    ],
                },
                VERIFY_SCHEMA,
            )
            decisions = verified["decisions"]
        accepted_ids = {item["qa_id"] for item in decisions if item["accepted"]}
        result["candidates"].extend(valid)
        result["local_candidates"].extend(qa for qa in valid if qa["qa_id"] in accepted_ids)
        segments.append(
            {
                "segment_id": segment_id,
                "candidate_limit": limits[segment_id],
                "generated_count": len(generated["qas"]),
                "skip_reason": generated["skip_reason"],
                "program_valid_count": len(valid),
                "program_rejections": program_rejections,
                "verification_decisions": decisions,
            }
        )
    decisions = []
    if result["local_candidates"]:
        reviewed = client.call(
            "document_review",
            {
                **context,
                "text": document["text"],
                "segments": document["segments"],
                "qas": [
                    {
                        key: qa[key]
                        for key in (
                            "qa_id",
                            "segment_id",
                            "fact_statement",
                            "question",
                            "answer",
                            "evidence_quote",
                        )
                    }
                    for qa in result["local_candidates"]
                ],
            },
            DOCUMENT_REVIEW_SCHEMA,
        )
        decisions = reviewed["decisions"]
    # Rejected attempts stay rejected. A repaired question receives a new round ID.
    result["review_decisions"] = [
        dict(item, accepted=False, reason=prior_rejections[item["qa_id"]], fact_group_id="")
        if item["qa_id"] in prior_rejections
        else item
        for item in decisions
    ]
    result["assembly"] = assemble_document(
        document, result["local_candidates"], result["review_decisions"], config["qa"]
    )
    after = result["assembly"]
    result["rounds"].append(
        {
            "round_index": round_index,
            "phase": "annotate",
            "segments": segments,
            "model_review_decisions": decisions,
            "statistics": after["statistics"],
            "complete_quota": after["ok"],
            "shortfalls": after["shortfalls"],
            "new_eligible_facts": sum(
                qa["qa_id"] not in previous_candidate_ids for qa in after["eligible_qas"]
            ),
            "net_eligible_change": len(after["eligible_qas"]) - len(before_ids),
            "elapsed_seconds": time.perf_counter() - started,
            "requests": _request_statistics(
                client.batch_root, document["trajectory_id"], round_index
            ),
        }
    )
    result["finished"] = after["ok"] or round_index >= config["qa"]["max_supplement_rounds"]
    result["completed_at"] = _now()


def _annotate_one(
    index: int,
    document: dict,
    config: dict,
    root: Path,
    client: AnnotationClient,
) -> dict:
    path = _document_result_path(root, index)
    failed_path = root / "failed-documents" / f"doc-{index:03d}.json"
    if failed_path.exists():
        return _load_json(failed_path)
    if path.exists():
        result = _load_json(path)
        if result["trajectory_id"] != document["trajectory_id"]:
            raise ValueError(f"document {index} changed after annotation")
    else:
        result = {
            "document_index": index,
            "trajectory_id": document["trajectory_id"],
            "rounds": [],
            "candidates": [],
            "local_candidates": [],
            "review_decisions": [],
            "assembly": None,
        }
    try:
        while not result["rounds"] or (
            not result["assembly"]["ok"]
            and len(result["rounds"]) <= config["qa"]["max_supplement_rounds"]
        ):
            _run_round(document, result, config, client)
            _save_json(path, result)
        (path.parent / f"doc-{index:03d}.error.json").unlink(missing_ok=True)
        return result
    except DocumentAnnotationError as error:
        failure = _record_document_failure(root, index, document, error)
        (path.parent / f"doc-{index:03d}.error.json").unlink(missing_ok=True)
        return failure
    except Exception as error:
        client.stop.set()
        _save_json(
            path.parent / f"doc-{index:03d}.error.json",
            {"document_index": index, "at": _now(), "error": f"{type(error).__name__}: {error}"},
        )
        raise


def _document_results(root: Path, documents: list[dict], require_all: bool = False) -> list[dict]:
    results = []
    failures = _document_failures(root, documents)
    for index, document in enumerate(documents):
        path = _document_result_path(root, index)
        if not path.exists():
            if require_all and index not in failures:
                raise FileNotFoundError(f"document {index} has not been annotated")
            continue
        result = _load_json(path)
        if (
            result["document_index"] != index
            or result["trajectory_id"] != document["trajectory_id"]
        ):
            raise ValueError(f"annotated document {index} does not match frozen selection")
        if require_all and not result["finished"] and index not in failures:
            raise ValueError(f"document {index} annotation is incomplete; resume annotate first")
        results.append(result)
    return results


def _request_statistics(
    root: Path, trajectory_id: str | None = None, round_index: int | None = None
) -> dict:
    path = root / "requests.jsonl"
    by_stage: dict[str, dict] = defaultdict(
        lambda: {
            "calls": 0,
            "new_logical_requests": 0,
            "network_attempts": 0,
            "cache_hits": 0,
            "failed_calls": 0,
            "content_filtered_calls": 0,
            "annotation_contract_failed_calls": 0,
            "network_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "usage_missing": 0,
        }
    )
    if not path.exists():
        return {}
    actual_requests: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        lines = stream.read().splitlines()
        fcntl.flock(stream, fcntl.LOCK_UN)
    for line in lines:
        record = json.loads(line)
        if trajectory_id is not None and record.get("trajectory_id") != trajectory_id:
            continue
        if round_index is not None and record.get("round_index") != round_index:
            continue
        stage = by_stage[record["stage"]]
        stage["calls"] += 1
        stage["network_attempts"] += record["network_attempts"]
        stage["cache_hits"] += bool(record["cache_hit"])
        stage["failed_calls"] += not record["ok"]
        stage["content_filtered_calls"] += record.get("failure_reason") == "content_filtered"
        stage["annotation_contract_failed_calls"] += (
            record.get("failure_reason") == "annotation_contract_failed"
        )
        stage["network_seconds"] += record["network_seconds"]
        if "request_id" in record:
            key = (record["stage"], record["request_id"])
            request = actual_requests.setdefault(key, {"network_attempts": set(), "usage": {}})
            count, total = record["network_attempts"], record["attempts_total"]
            request["network_attempts"].update(
                record.get("network_attempt_numbers", range(total - count + 1, total + 1))
            )
            if "usage_by_attempt" in record:
                for attempt, usage in record["usage_by_attempt"].items():
                    if usage is not None:
                        request["usage"][int(attempt)] = usage
            elif record.get("usage") is not None:
                # Before response retries, only the last returned attempt had usage.
                # Old raw-cache replays recorded attempts_total=0.
                attempt = total or max(request["network_attempts"], default=1)
                request["usage"][attempt] = record["usage"]
    for (stage_name, _), request in actual_requests.items():
        if not request["network_attempts"]:
            continue
        stage = by_stage[stage_name]
        stage["new_logical_requests"] += 1
        missing = False
        for attempt in request["network_attempts"]:
            usage = request["usage"].get(attempt)
            if usage is None or any(
                type(usage.get(k)) is not int for k in ("input_tokens", "output_tokens")
            ):
                missing = True
            else:
                stage["input_tokens"] += usage["input_tokens"]
                stage["output_tokens"] += usage["output_tokens"]
        stage["usage_missing"] += missing
    return dict(by_stage)


def _round_summary(results: list[dict]) -> dict:
    rounds = []
    for number in sorted({r["round_index"] for result in results for r in result["rounds"]}):
        entries = [r for result in results for r in result["rounds"] if r["round_index"] == number]
        rounds.append(
            {
                "round_index": number,
                "trajectories": len(entries),
                "complete_quota_trajectories": sum(r["complete_quota"] for r in entries),
                "generated_candidates": sum(
                    s["generated_count"] for r in entries for s in r["segments"]
                ),
                "new_eligible_facts": sum(r["new_eligible_facts"] for r in entries),
                "net_eligible_change": sum(r["net_eligible_change"] for r in entries),
                "requests": {
                    key: sum(stage[key] for r in entries for stage in r["requests"].values())
                    for key in (
                        "new_logical_requests",
                        "network_attempts",
                        "cache_hits",
                        "input_tokens",
                        "output_tokens",
                        "usage_missing",
                        "network_seconds",
                    )
                },
            }
        )
    return {
        "trajectories": len(results),
        "initial_complete_trajectories": sum(r["rounds"][0]["complete_quota"] for r in results),
        "latest_complete_trajectories": sum(r["assembly"]["ok"] for r in results),
        "rounds": rounds,
    }


def _annotation_summary(root: Path, documents: list[dict]) -> dict:
    results = _document_results(root, documents)
    failures = _document_failures(root, documents)
    return {
        "frozen_trajectories": len(documents),
        "annotated_trajectories": sum(result["finished"] for result in results),
        **_failure_counts(failures),
        "incomplete_trajectories": sum(
            not result["finished"] and result["document_index"] not in failures
            for result in results
        ),
        "complete_quota_trajectories": sum(result["assembly"]["ok"] for result in results),
        "generated_candidates": sum(
            segment["generated_count"]
            for result in results
            for round_record in result["rounds"]
            for segment in round_record["segments"]
        ),
        "program_valid_candidates": sum(len(result["candidates"]) for result in results),
        "program_rejected_candidates": sum(
            len(segment["program_rejections"])
            for result in results
            for round_record in result["rounds"]
            for segment in round_record["segments"]
        ),
        "locally_verified_candidates": sum(len(result["local_candidates"]) for result in results),
        "distinct_reviewed_facts": sum(
            result["assembly"]["statistics"]["eligible_fact_count"] for result in results
        ),
        "requests": _request_statistics(root),
        "supplementation": _round_summary(results),
    }


def annotate(config: dict, limit: int | None = None) -> dict:
    """Generate, locally verify and review the frozen trajectories without replacing any."""
    root, prompts = _batch(config)
    documents = _selection(root)["documents"]
    if limit is None:
        limit = len(documents)
    if type(limit) is not int or not 0 <= limit <= len(documents):
        raise ValueError("annotate limit must be within the frozen trajectory count")
    client = AnnotationClient(config["annotation"], root, Path(config["cache_dir"]), prompts)
    pending = [(index, document) for index, document in enumerate(documents[:limit])]
    failures: list[Exception] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config["annotation"]["concurrency"]) as pool:
        cursor = 0
        futures = set()
        while futures or (cursor < len(pending) and not client.stop.is_set()):
            while (
                cursor < len(pending)
                and len(futures) < config["annotation"]["concurrency"]
                and not client.stop.is_set()
            ):
                index, document = pending[cursor]
                futures.add(pool.submit(_annotate_one, index, document, config, root, client))
                cursor += 1
            if not futures:
                break
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                try:
                    result = future.result()
                    if "reason" in result:
                        print(json.dumps(result, ensure_ascii=False), flush=True)
                        continue
                    print(
                        json.dumps(
                            {
                                "document_index": result["document_index"],
                                "complete_quota": result["assembly"]["ok"],
                                "eligible_facts": result["assembly"]["statistics"][
                                    "eligible_fact_count"
                                ],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                except Exception as error:
                    client.stop.set()
                    failures.append(error)
    summary = _annotation_summary(root, documents)
    summary.update(
        stage="annotate", limit=limit, elapsed_this_run_seconds=time.perf_counter() - started
    )
    _save_json(root / "annotate-summary.json", summary)
    if failures:
        raise RuntimeError(f"annotation stopped after a fatal error: {failures[0]}") from failures[
            0
        ]
    return summary


def _save_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stage_counts(results: list[dict]) -> dict:
    counts = {}
    max_segments = max(
        (len(r["assembly"]["statistics"]["eligible_by_segment"]) for r in results), default=0
    )
    for segment_index in range(max_segments):
        segment_id = f"seg{segment_index}"
        stage = {
            "generated": 0,
            "program_valid": 0,
            "program_rejected": 0,
            "locally_verified": 0,
            "full_review_accepted": 0,
            "distinct_facts": 0,
        }
        for result in results:
            for round_record in result["rounds"]:
                for segment in round_record["segments"]:
                    if segment["segment_id"] == segment_id:
                        stage["generated"] += segment["generated_count"]
                        stage["program_valid"] += segment["program_valid_count"]
                        stage["program_rejected"] += len(segment["program_rejections"])
            stage["locally_verified"] += sum(
                qa["segment_id"] == segment_id for qa in result["local_candidates"]
            )
            decision_by_id = {
                decision["qa_id"]: decision for decision in result["review_decisions"]
            }
            stage["full_review_accepted"] += sum(
                qa["segment_id"] == segment_id and decision_by_id[qa["qa_id"]]["accepted"]
                for qa in result["local_candidates"]
            )
            stage["distinct_facts"] += result["assembly"]["statistics"]["eligible_by_segment"].get(
                segment_id, 0
            )
        counts[segment_id] = stage
    return counts


def finalize(config: dict) -> dict:
    """Validate complete annotations and write batch data without model requests."""
    root, prompts = _batch(config)
    selection = _selection(root)
    documents = selection["documents"]
    results = _document_results(root, documents, require_all=True)
    by_index = {r["document_index"]: r for r in results}
    failures = _document_failures(root, documents)
    trajectories, final_results = [], []
    for index, document in enumerate(documents):
        final = {
            "document_index": index,
            "trajectory_id": document["trajectory_id"],
            "split": document["split"],
            "segment_count": len(document["segments"]),
        }
        if index in failures:
            final.update(complete_quota=False, failure_reason=failures[index]["reason"])
        else:
            result = by_index[index]
            assembly = assemble_document(
                document, result["local_candidates"], result["review_decisions"], config["qa"]
            )
            result["assembly"] = assembly
            final.update(
                complete_quota=assembly["ok"],
                failure_reason=None if assembly["ok"] else "quota_shortfall",
                supplement_rounds_used=len(result["rounds"]) - 1,
                statistics=assembly["statistics"],
                shortfalls=assembly["shortfalls"],
                eligible_qa_ids=[q["qa_id"] for q in assembly["eligible_qas"]],
            )
            if assembly["ok"]:
                validate_trajectory(assembly["trajectory"], config["qa"])
                trajectories.append(assembly["trajectory"])
        _save_json(root / "final-documents" / f"doc-{index:03d}.json", final)
        final_results.append(final)

    dataset_dir = Path(config["dataset_dir"])
    preparation_path = dataset_dir / "preparation.json"
    preparation_path.unlink(missing_ok=True)
    split_paths = {}
    for split in ("train", "dev", "test"):
        path = dataset_dir / f"{split}.jsonl"
        _save_jsonl(path, [row for row in trajectories if row["split"] == split])
        split_paths[split] = str(path)
    request_stats = _request_statistics(root)
    total_tokens = sum(
        stage["input_tokens"] + stage["output_tokens"] for stage in request_stats.values()
    )
    summary = {
        "frozen_trajectories": len(documents),
        "frozen_source_documents": len({d["document_id"] for d in documents}),
        "annotated_trajectories": sum(r["finished"] for r in results),
        "complete_trajectories": len(trajectories),
        "complete_source_documents": len({d["document_id"] for d in trajectories}),
        **_failure_counts(failures),
        "quota_failed_trajectories": sum(
            d["failure_reason"] == "quota_shortfall" for d in final_results
        ),
        "trajectory_success_rate": len(trajectories) / len(documents) if documents else None,
        "final_qas": sum(len(t["qas"]) for t in trajectories),
        "by_split": {
            split: {
                "frozen_trajectories": sum(d["split"] == split for d in documents),
                "frozen_source_documents": len(
                    {d["document_id"] for d in documents if d["split"] == split}
                ),
                "complete_trajectories": sum(d["split"] == split for d in trajectories),
                "complete_source_documents": len(
                    {d["document_id"] for d in trajectories if d["split"] == split}
                ),
                **_failure_counts(
                    {i: f for i, f in failures.items() if documents[i]["split"] == split}
                ),
                "quota_failed_trajectories": sum(
                    d["split"] == split and d["failure_reason"] == "quota_shortfall"
                    for d in final_results
                ),
                "qas": sum(len(d["qas"]) for d in trajectories if d["split"] == split),
            }
            for split in ("train", "dev", "test")
        },
        "by_segment_count": {
            str(n): {
                "frozen_trajectories": sum(len(d["segments"]) == n for d in documents),
                "complete_trajectories": sum(len(d["segments"]) == n for d in trajectories),
            }
            for n in sorted({len(d["segments"]) for d in documents})
        },
        "network_tokens_per_frozen_trajectory": total_tokens / len(documents)
        if documents
        else None,
        "network_tokens_per_successful_trajectory": total_tokens / len(trajectories)
        if trajectories
        else None,
    }
    preparation = {
        "created_at": _now(),
        "dataset": dataset_dir.name,
        "token_estimation_rule": "len(text) / 4",
        "source_pool_id": selection["source_pool_id"],
        "source_pool_dir": config["source_pool_dir"],
        "source_pool_config": selection["source_pool_config"],
        "batch_index": config["batch_index"],
        "batch_ranges": selection["ranges"],
        "qa": config["qa"],
        "annotation": {k: v for k, v in config["annotation"].items() if k != "endpoint"},
        "prompts": prompts,
        "selection_statistics": selection["statistics"],
        "stage_counts": _stage_counts(results),
        "supplementation": _round_summary(results),
        "request_statistics": request_stats,
        "summary": summary,
        "documents": final_results,
        "document_failures": list(failures.values()),
        "artifacts_dir": str(root),
        "software": {
            "python": platform.python_version(),
            "pyarrow": version("pyarrow"),
            "transformers": version("transformers"),
        },
    }
    _save_json(preparation_path, preparation)
    return {
        "stage": "finalize",
        "split_paths": split_paths,
        "preparation_path": str(preparation_path),
        "summary": summary,
    }
