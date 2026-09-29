"""Explicit, resumable stages for the local FineWeb factual QA pilot."""

from __future__ import annotations

import copy
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
from typing import Any

from latent_working_memory.data_preparation.fineweb_qa.annotation import (
    ANSWER_SCHEMA,
    DOCUMENT_REVIEW_SCHEMA,
    GENERATE_SCHEMA,
    VERIFY_SCHEMA,
    AnnotationClient,
    locate_candidates,
)
from latent_working_memory.data_preparation.fineweb_qa.diagnostics import (
    review_shards,
    score_answers,
    select_review_panel,
)
from latent_working_memory.data_preparation.fineweb_qa.finalization import (
    apply_resolved_reviews,
    validate_resolved_reviews,
)
from latent_working_memory.data_preparation.fineweb_qa.sources import prepare_selection


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _root(config: dict) -> Path:
    return Path(config["artifacts_dir"])


def _prompt_texts(config: dict) -> dict[str, str]:
    required = {"generate", "verify", "document_review", "answer", "subagent_review"}
    if set(config["prompts"]) != required:
        raise ValueError("FineWeb QA prompts must name the five pilot stages")
    texts = {
        name: Path(path).read_text(encoding="utf-8") for name, path in config["prompts"].items()
    }
    if any(not value.strip() for value in texts.values()):
        raise ValueError("FineWeb QA prompts must not be empty")
    return texts


def _without_endpoint(config: dict) -> dict:
    frozen = copy.deepcopy(config)
    frozen["annotation"].pop("endpoint")
    return frozen


def _batch(config: dict, create: bool = False) -> tuple[Path, dict[str, str]]:
    for key in ("max_supplement_rounds", "supplement_surplus"):
        if type(config["qa"][key]) is not int or config["qa"][key] < 0:
            raise ValueError(f"qa.{key} must be a nonnegative integer")
    root = _root(config)
    config_path, prompts_path = root / "config.json", root / "prompts.json"
    prompts = _prompt_texts(config)
    if create and not config_path.exists() and not prompts_path.exists():
        root.mkdir(parents=True, exist_ok=True)
        _save_json(config_path, config)
        _save_json(prompts_path, prompts)
    if not config_path.exists() or not prompts_path.exists():
        raise FileNotFoundError("run prepare to freeze the pilot configuration and prompts")
    if _without_endpoint(_load_json(config_path)) != _without_endpoint(config):
        raise ValueError("pilot configuration changed; start a new batch")
    if _load_json(prompts_path) != prompts:
        raise ValueError("pilot prompts changed; start a new batch")
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
        "frozen_documents": len(selection["documents"]),
        "statistics": selection["statistics"],
    }


def _check_decisions(output: dict, qa_ids: list[str], stage: str) -> list[dict]:
    decisions = output["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(qa_ids):
        raise ValueError(f"{stage} must decide on every QA once")
    ids = [decision["qa_id"] for decision in decisions]
    if len(set(ids)) != len(ids) or set(ids) != set(qa_ids):
        raise ValueError(f"{stage} QA IDs do not match the candidate set")
    for decision in decisions:
        if type(decision["accepted"]) is not bool or not isinstance(decision["reason"], str):
            raise ValueError(f"{stage} decision needs a boolean and a string reason")
        if not decision["accepted"] and not decision["reason"].strip():
            raise ValueError(f"rejected {stage} decision needs a reason")
        if (
            stage == "document_review"
            and decision["accepted"]
            and not decision["fact_group_id"].strip()
        ):
            raise ValueError("accepted document-review QA needs a fact group")
    return decisions


def _document_result_path(root: Path, index: int) -> Path:
    return root / "documents" / f"doc-{index:03d}.json"


def _run_round(
    document: dict,
    result: dict,
    config: dict,
    client: AnnotationClient,
    phase: str,
    resolved_by_id: dict[str, dict],
) -> None:
    """Append one generation/filtering round, keeping stable IDs for every attempt."""
    round_index = len(result["rounds"])
    before = result["assembly"]
    previous_candidate_ids = {qa["qa_id"] for qa in result["candidates"]}
    before_ids = {qa["qa_id"] for qa in before["eligible_qas"]} if before else set()
    limits = (
        {f"seg{i}": n for i, n in enumerate(config["qa"]["candidate_per_segment"])}
        if before is None
        else {
            item["segment_id"]: item["missing_total"] + config["qa"]["supplement_surplus"]
            for item in before["shortfalls"]
        }
    )
    context = {
        "trajectory_id": document["trajectory_id"],
        "round_index": round_index,
        "phase": phase,
    }
    prior_rejections = {
        item["qa_id"]: item["reason"] for item in result["review_decisions"] if not item["accepted"]
    }
    prior_rejections.update(
        {qa_id: item["reason"] for qa_id, item in resolved_by_id.items() if not item["accepted"]}
    )
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
            decisions = _check_decisions(verified, [qa["qa_id"] for qa in valid], "verify")
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
        decisions = _check_decisions(
            reviewed, [qa["qa_id"] for qa in result["local_candidates"]], "document_review"
        )
    # Rejected attempts stay rejected. A repaired question receives a new round ID.
    result["review_decisions"] = [
        dict(item, accepted=False, reason=prior_rejections[item["qa_id"]], fact_group_id="")
        if item["qa_id"] in prior_rejections
        else item
        for item in decisions
    ]
    result["assembly"] = apply_resolved_reviews(document, result, resolved_by_id, config["qa"])
    after = result["assembly"]
    result["rounds"].append(
        {
            "round_index": round_index,
            "phase": phase,
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
    resolved_by_id: dict[str, dict] | None = None,
) -> dict:
    phase = "annotate" if resolved_by_id is None else "finalize"
    directory = root / ("documents" if phase == "annotate" else "reviewed-documents")
    path = directory / f"doc-{index:03d}.json"
    reviews = {} if resolved_by_id is None else resolved_by_id
    if path.exists():
        result = _load_json(path)
        if result["trajectory_id"] != document["trajectory_id"]:
            raise ValueError(f"document {index} changed after annotation")
        if phase == "finalize" and result["resolved_review"] != reviews:
            raise ValueError(
                "resolved review changed after finalization started; start a new batch"
            )
    elif phase == "finalize":
        result = _load_json(_document_result_path(root, index))
        result["resolved_review"] = reviews
        result["assembly"] = apply_resolved_reviews(document, result, reviews, config["qa"])
        result["finished"] = (
            result["assembly"]["ok"]
            or len(result["rounds"]) > config["qa"]["max_supplement_rounds"]
        )
        _save_json(path, result)
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
            _run_round(document, result, config, client, phase, reviews)
            _save_json(path, result)
        (directory / f"doc-{index:03d}.error.json").unlink(missing_ok=True)
        return result
    except Exception as error:
        client.stop.set()
        _save_json(
            directory / f"doc-{index:03d}.error.json",
            {
                "document_index": index,
                "trajectory_id": document["trajectory_id"],
                "at": _now(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def _document_results(root: Path, documents: list[dict], require_all: bool = False) -> list[dict]:
    results = []
    for index, document in enumerate(documents):
        path = _document_result_path(root, index)
        if not path.exists():
            if require_all:
                raise FileNotFoundError(f"document {index} has not been annotated")
            continue
        result = _load_json(path)
        if (
            result["document_index"] != index
            or result["trajectory_id"] != document["trajectory_id"]
        ):
            raise ValueError(f"annotated document {index} does not match frozen selection")
        if require_all and not result["finished"]:
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
            "network_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "usage_missing": 0,
        }
    )
    if not path.exists():
        return {}
    actual_requests: dict[tuple[str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
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
        stage["network_seconds"] += record["network_seconds"]
        if "request_id" in record:
            key = (record["stage"], record["request_id"])
            request = actual_requests.setdefault(key, {"network_attempted": False, "usage": None})
            request["network_attempted"] |= bool(record["network_attempts"])
            if record.get("usage") is not None:
                request["usage"] = record["usage"]
    for (stage_name, _), request in actual_requests.items():
        if not request["network_attempted"]:
            continue
        stage = by_stage[stage_name]
        stage["new_logical_requests"] += 1
        usage = request["usage"]
        if usage is None or any(
            type(usage.get(key)) is not int for key in ("input_tokens", "output_tokens")
        ):
            stage["usage_missing"] += 1
        else:
            stage["input_tokens"] += usage["input_tokens"]
            stage["output_tokens"] += usage["output_tokens"]
    return dict(by_stage)


def _round_summary(results: list[dict]) -> dict:
    rounds = []
    for number in sorted({r["round_index"] for result in results for r in result["rounds"]}):
        entries = [r for result in results for r in result["rounds"] if r["round_index"] == number]
        rounds.append(
            {
                "round_index": number,
                "documents": len(entries),
                "complete_quota_documents": sum(r["complete_quota"] for r in entries),
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
        "documents": len(results),
        "initial_complete_documents": sum(r["rounds"][0]["complete_quota"] for r in results),
        "latest_complete_documents": sum(r["assembly"]["ok"] for r in results),
        "rounds": rounds,
    }


def _annotation_summary(root: Path, documents: list[dict]) -> dict:
    results = _document_results(root, documents)
    return {
        "frozen_documents": len(documents),
        "annotated_documents": sum(result["finished"] for result in results),
        "incomplete_documents": sum(not result["finished"] for result in results),
        "complete_quota_documents": sum(result["assembly"]["ok"] for result in results),
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
    """Generate, locally verify and review the frozen documents without replacing any."""
    root, prompts = _batch(config)
    documents = _selection(root)["documents"]
    if limit is None:
        limit = len(documents)
    if type(limit) is not int or not 0 <= limit <= len(documents):
        raise ValueError("annotate limit must be within the frozen document count")
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


def _answer_one(client: AnnotationClient, qa: dict, condition: str) -> tuple[str, str, str]:
    content = {"question": qa["question"]}
    if condition == "evidence":
        content["evidence"] = qa["evidence_text"]
    elif condition != "question_only":
        raise ValueError(f"unknown answer condition: {condition}")
    response = client.call("answer", content, ANSWER_SCHEMA)
    return qa["qa_id"], condition, response["answer"]


def diagnose(config: dict) -> dict:
    """Answer a fixed, dispersed panel with and without the original evidence."""
    root, prompts = _batch(config)
    documents = _selection(root)["documents"]
    results = _document_results(root, documents, require_all=True)
    panel = select_review_panel(
        documents, results, config["qa"]["review_sample"], config["qa"]["review_seed"]
    )
    panel_path = root / "diagnostic-panel.json"
    if panel_path.exists():
        if _load_json(panel_path) != panel:
            raise ValueError("diagnostic panel changed after it was frozen")
    else:
        _save_json(panel_path, panel)

    answer_path = root / "diagnostic-answers.json"
    answers: dict[str, dict[str, str]] = _load_json(answer_path) if answer_path.exists() else {}
    panel_ids = {qa["qa_id"] for qa in panel}
    if set(answers) - panel_ids or any(
        set(value) - {"evidence", "question_only"}
        or any(not isinstance(answer, str) for answer in value.values())
        for value in answers.values()
    ):
        raise ValueError("saved diagnostic answers do not match the frozen panel")
    client = AnnotationClient(config["annotation"], root, Path(config["cache_dir"]), prompts)
    pending = [
        (qa, condition)
        for qa in panel
        for condition in ("evidence", "question_only")
        if condition not in answers.get(qa["qa_id"], {})
    ]
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
                qa, condition = pending[cursor]
                futures.add(pool.submit(_answer_one, client, qa, condition))
                cursor += 1
            if not futures:
                break
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                try:
                    qa_id, condition, answer = future.result()
                    answers.setdefault(qa_id, {})[condition] = answer
                    _save_json(answer_path, answers)
                except Exception as error:
                    client.stop.set()
                    failures.append(error)
    if failures:
        raise RuntimeError(f"diagnosis stopped after a fatal error: {failures[0]}") from failures[0]
    scores = score_answers(panel, answers)
    diagnostics = {
        "completed_at": _now(),
        "panel_qa_ids": [qa["qa_id"] for qa in panel],
        "scores": scores,
        "requests": _request_statistics(root),
        "elapsed_this_run_seconds": time.perf_counter() - started,
    }
    _save_json(root / "diagnostics.json", diagnostics)
    first, second = review_shards(panel, documents, results, answers)
    _save_json(root / "review-panel-1.json", first)
    _save_json(root / "review-panel-2.json", second)
    return {
        "stage": "diagnose",
        "panel_count": len(panel),
        "evidence": scores["evidence"],
        "question_only": scores["question_only"],
        "review_panels": [str(root / "review-panel-1.json"), str(root / "review-panel-2.json")],
        "requests": _request_statistics(root),
    }


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
    for segment_index in range(8):
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
            stage["distinct_facts"] += result["assembly"]["statistics"]["eligible_by_segment"][
                segment_id
            ]
        counts[segment_id] = stage
    return counts


def finalize(config: dict) -> dict:
    """Apply the fixed semantic review, recheck quotas and write the local train data."""
    root, prompts = _batch(config)
    selection = _selection(root)
    documents = selection["documents"]
    results = _document_results(root, documents, require_all=True)
    diagnostics = _load_json(root / "diagnostics.json")
    panel = _load_json(root / "diagnostic-panel.json")
    resolved = _load_json(root / "resolved-review.json")
    resolved_by_id = validate_resolved_reviews(panel, resolved)
    if any(decision["evidence_prediction_correct"] is None for decision in resolved_by_id.values()):
        raise ValueError("every sampled evidence answer needs a semantic review decision")
    if diagnostics["panel_qa_ids"] != [qa["qa_id"] for qa in panel]:
        raise ValueError("diagnostics and resolved review refer to different panels")

    # Validate review membership against the original panel pool, before supplementation.
    original_decisions = {
        item["qa_id"]: item for result in results for item in result["review_decisions"]
    }
    for qa_id, decision in resolved_by_id.items():
        for target in [qa_id, *decision["same_fact_with"]]:
            if target not in original_decisions or not original_decisions[target]["accepted"]:
                raise ValueError("resolved review references a model-rejected or unknown QA")
    client = AnnotationClient(config["annotation"], root, Path(config["cache_dir"]), prompts)
    final_results = []
    completed_results = []
    trajectories = []
    for index, (document, result) in enumerate(zip(documents, results, strict=True)):
        local_ids = {qa["qa_id"] for qa in result["local_candidates"]}
        reviewed = _annotate_one(
            index,
            document,
            config,
            root,
            client,
            {key: value for key, value in resolved_by_id.items() if key in local_ids},
        )
        completed_results.append(reviewed)
        assembly = reviewed["assembly"]
        final = {
            "document_index": index,
            "trajectory_id": document["trajectory_id"],
            "complete_quota": assembly["ok"],
            "supplement_rounds_used": len(reviewed["rounds"]) - 1,
            "statistics": assembly["statistics"],
            "shortfalls": assembly["shortfalls"],
            "eligible_qa_ids": [qa["qa_id"] for qa in assembly["eligible_qas"]],
        }
        _save_json(root / "final-documents" / f"doc-{index:03d}.json", final)
        final_results.append(final)
        if assembly["ok"]:
            trajectories.append(assembly["trajectory"])

    dataset_dir = Path(config["dataset_dir"])
    train_path = dataset_dir / "train.jsonl"
    _save_jsonl(train_path, trajectories)
    request_stats = _request_statistics(root)
    review_decisions = resolved["decisions"]
    semantically_checked = [
        decision["evidence_prediction_correct"]
        for decision in review_decisions
        if decision["evidence_prediction_correct"] is not None
    ]
    total_network_tokens = sum(
        stage["input_tokens"] + stage["output_tokens"] for stage in request_stats.values()
    )
    summary = {
        "frozen_documents": len(documents),
        "annotated_documents": len(results),
        "complete_quota_documents_before_review": sum(
            result["assembly"]["ok"] for result in results
        ),
        "complete_quota_documents_after_review": len(trajectories),
        "trajectory_success_rate": len(trajectories) / len(documents) if documents else None,
        "final_qas": sum(len(trajectory["qas"]) for trajectory in trajectories),
        "review_panel_qas": len(panel),
        "review_rejected_qas": sum(not decision["accepted"] for decision in review_decisions),
        "review_same_fact_links": sum(
            len(decision["same_fact_with"]) for decision in review_decisions
        ),
        "evidence_prediction_semantic_checked": len(semantically_checked),
        "evidence_prediction_semantic_correct": sum(semantically_checked),
        "network_tokens_per_frozen_document": total_network_tokens / len(documents)
        if documents
        else None,
        "network_tokens_per_successful_trajectory": total_network_tokens / len(trajectories)
        if trajectories
        else None,
    }
    preparation = {
        "created_at": _now(),
        "dataset": Path(config["dataset_dir"]).name,
        "token_estimation_rule": "len(text) / 4",
        "source": config["source"],
        "window": config["window"],
        "qa": config["qa"],
        "annotation": {
            key: value for key, value in config["annotation"].items() if key != "endpoint"
        },
        "prompts": prompts,
        "selection_statistics": selection["statistics"],
        "stage_counts": _stage_counts(completed_results),
        "supplementation": _round_summary(completed_results),
        "diagnostics": {
            "count": diagnostics["scores"]["count"],
            "evidence": diagnostics["scores"]["evidence"],
            "question_only": diagnostics["scores"]["question_only"],
        },
        "request_statistics": request_stats,
        "summary": summary,
        "documents": final_results,
        "artifacts_dir": str(root),
        "software": {
            "python": platform.python_version(),
            "pyarrow": version("pyarrow"),
            "transformers": version("transformers"),
        },
    }
    preparation_path = dataset_dir / "preparation.json"
    _save_json(preparation_path, preparation)
    return {
        "stage": "finalize",
        "train_path": str(train_path),
        "preparation_path": str(preparation_path),
        "summary": summary,
    }
