"""Explicit stages for the fixed FineWeb QA pilot."""

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import string
from statistics import median
from zoneinfo import ZoneInfo

from latent_working_memory.data_preparation.fineweb_qa.annotation import (
    ANSWER_SCHEMA,
    AnnotationClient,
    annotate_segment,
    review_document,
)
from latent_working_memory.data_preparation.fineweb_qa.assembly import (
    assemble_trajectory,
    validate_trajectory,
)
from latent_working_memory.data_preparation.fineweb_qa.sources import prepare_sources


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare(config: dict) -> dict:
    directory = Path(config["output"]["artifacts_dir"])
    path = directory / "selection.json"
    if (directory / "config.json").exists():
        load_frozen_batch(config)
    else:
        save_json(directory / "prompts.json", load_prompts(config))
        save_json(directory / "config.json", config)
    if path.exists():
        result = json.loads(path.read_text())
        print(f"Reusing frozen selection: {len(result['trajectories'])} documents", flush=True)
        return result
    result = prepare_sources(config)
    lengths = [len(trajectory["text"]) for trajectory in result["trajectories"]]
    segment_lengths = [
        segment["char_end"] - segment["char_start"]
        for trajectory in result["trajectories"]
        for segment in trajectory["segments"]
    ]
    for name, values in (("trajectory_chars", lengths), ("segment_chars", segment_lengths)):
        result["statistics"][name] = (
            {"min": min(values), "median": median(values), "max": max(values)} if values else None
        )
    result["statistics"]["estimated_tokens"] = (
        {"min": min(lengths) / 4, "median": median(lengths) / 4, "max": max(lengths) / 4}
        if lengths
        else None
    )
    result["source_config"] = config["source"]
    result["text_config"] = config["text"]
    save_json(path, result)
    print(json.dumps(result["statistics"], ensure_ascii=False), flush=True)
    return result


def load_prompts(config: dict) -> dict:
    return {name: Path(path).read_text() for name, path in config["annotation"]["prompts"].items()}


def load_frozen_batch(config: dict) -> tuple[dict, dict]:
    """Only the service address may change within a prepared batch."""
    directory = Path(config["output"]["artifacts_dir"])
    frozen_config = json.loads((directory / "config.json").read_text())
    frozen_prompts = json.loads((directory / "prompts.json").read_text())

    def identity(settings: dict) -> dict:
        return {
            **settings,
            "annotation": {
                key: value for key, value in settings["annotation"].items() if key != "endpoint"
            },
        }

    if identity(config) != identity(frozen_config):
        raise ValueError(
            "Frozen batch settings differ; only annotation.endpoint may change. "
            "Use a new artifacts directory for changed settings."
        )
    if load_prompts(config) != frozen_prompts:
        raise ValueError(
            "Frozen batch prompts differ; use a new artifacts directory for changed prompts"
        )
    return frozen_config, frozen_prompts


def run_bounded(
    function, work: list, task_ids: list[str], client, config: dict, stage: str
) -> list:
    """Stop submitting on the first failure and let only issued requests finish."""
    concurrency = config["annotation"]["concurrency"]
    pool = ThreadPoolExecutor(max_workers=concurrency)
    pending = {}
    results = [None] * len(work)
    submitted = 0

    def run_one(item):
        try:
            if client.stop_event.is_set():
                raise RuntimeError("Batch stopped after another task failed")
            return function(item)
        except BaseException:
            client.stop_event.set()
            raise

    def fill_slots():
        nonlocal submitted
        while (
            submitted < len(work) and len(pending) < concurrency and not client.stop_event.is_set()
        ):
            future = pool.submit(run_one, work[submitted])
            pending[future] = submitted
            submitted += 1

    try:
        fill_slots()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Inspect all completed tasks before allowing any replacement submissions.
            for future in sorted(done, key=pending.__getitem__):
                index = pending[future]
                results[index] = future.result()
                del pending[future]
            fill_slots()
    except BaseException as error:
        client.stop_event.set()
        cancelled = []
        running = []
        for future, index in pending.items():
            if future.cancel():
                cancelled.append(task_ids[index])
            elif not future.done():
                running.append(task_ids[index])
        # An HTTP request already issued cannot be withdrawn. The shared stop event
        # prevents subsequent requests (including a segment's verification request).
        pool.shutdown(wait=True, cancel_futures=True)
        save_json(
            Path(config["output"]["artifacts_dir"]) / f"failure-{stage}.json",
            {
                "stage": stage,
                "error": f"{type(error).__name__}: {error}",
                "concurrency": concurrency,
                "submitted_tasks": submitted,
                "unsubmitted_tasks": len(work) - submitted,
                "cancelled_tasks": cancelled,
                "running_tasks_at_stop": running,
                "issued_requests_withdrawn": False,
                "running_tasks_drained": True,
                "logical_calls": client.calls,
            },
        )
        raise
    else:
        pool.shutdown(wait=True)
    return results


def annotate(config: dict, limit: int) -> dict:
    directory = Path(config["output"]["artifacts_dir"])
    _, prompts = load_frozen_batch(config)
    selection = json.loads((directory / "selection.json").read_text())
    trajectories = selection["trajectories"][:limit]
    client = AnnotationClient(config)
    work = [
        (trajectory, segment) for trajectory in trajectories for segment in trajectory["segments"]
    ]

    def annotate_one(item: tuple[dict, dict]) -> dict:
        trajectory, segment = item
        result = annotate_segment(client, trajectory, segment, prompts)
        print(
            f"Segment {trajectory['trajectory_id']} / {segment['segment_id']}: {result['counts']}",
            flush=True,
        )
        return result

    task_ids = [f"{t['trajectory_id']}/segment-{s['segment_id']}" for t, s in work]
    segment_results = run_bounded(annotate_one, work, task_ids, client, config, "annotate")
    results = []
    for trajectory in trajectories:
        segments = [r for r in segment_results if r["trajectory_id"] == trajectory["trajectory_id"]]
        accepted = [qa for result in segments for qa in result["accepted"]]
        review = review_document(client, trajectory, accepted, prompts)
        assembly = assemble_trajectory(trajectory, accepted, review, config)
        document = {
            "trajectory_id": trajectory["trajectory_id"],
            "segments": segments,
            "review": review,
            "assembly": assembly,
        }
        save_json(directory / "documents" / f"{trajectory['trajectory_id']}.json", document)
        results.append(document)
        print(
            f"Document {trajectory['trajectory_id']}: complete={assembly['complete']}", flush=True
        )
    summary = {
        "documents": len(results),
        "complete": sum(r["assembly"]["complete"] for r in results),
        "logical_calls": client.calls,
    }
    save_json(directory / f"annotation-{limit}.json", summary)
    return summary


def normalize_answer(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_metrics(prediction: str, answer: str) -> dict:
    prediction, answer = normalize_answer(prediction), normalize_answer(answer)
    p, a = prediction.split(), answer.split()
    overlap = sum((Counter(p) & Counter(a)).values())
    f1 = 2 * overlap / (len(p) + len(a)) if p and a else float(p == a)
    return {"em": float(prediction == answer), "f1": f1}


def select_panel(selection: dict, documents: dict, config: dict) -> list[dict]:
    """Round-robin documents and segment positions before taking additional questions."""
    seed = config["source"]["seed"]
    buckets = []
    trajectories = sorted(
        selection["trajectories"],
        key=lambda t: sha256(f"{seed}:panel:{t['trajectory_id']}".encode()).hexdigest(),
    )
    segment_count = config["text"]["segments"]
    for position in range(segment_count):
        for trajectory in trajectories:
            offset = (
                int(sha256(trajectory["trajectory_id"].encode()).hexdigest(), 16) % segment_count
            )
            segment_id = (offset + 3 * position) % segment_count + 1
            document = documents[trajectory["trajectory_id"]]
            candidates = [
                qa for qa in document["assembly"]["accepted_qas"] if qa["segment_id"] == segment_id
            ]
            candidates.sort(key=lambda q: sha256(f"{seed}:panel:{q['qa_id']}".encode()).hexdigest())
            buckets.append(
                [{"trajectory_id": trajectory["trajectory_id"], "qa": qa} for qa in candidates]
            )
    panel = []
    while any(buckets) and len(panel) < config["review"]["question_limit"]:
        for bucket in buckets:
            if bucket:
                panel.append(bucket.pop(0))
                if len(panel) == config["review"]["question_limit"]:
                    break
    return panel


def diagnose(config: dict) -> dict:
    directory = Path(config["output"]["artifacts_dir"])
    _, prompts = load_frozen_batch(config)
    selection = json.loads((directory / "selection.json").read_text())
    documents = {
        t["trajectory_id"]: json.loads(
            (directory / "documents" / f"{t['trajectory_id']}.json").read_text()
        )
        for t in selection["trajectories"]
    }
    panel = select_panel(selection, documents, config)
    client = AnnotationClient(config)

    def answer_one(item: tuple[dict, str]) -> tuple[str, str, dict]:
        entry, condition = item
        qa = entry["qa"]
        content = {"question": qa["question"]}
        if condition == "evidence":
            content["evidence"] = qa["evidence_quote"]
        prediction = client.call("answer", prompts["answer"], content, ANSWER_SCHEMA)["answer"]
        return (
            qa["qa_id"],
            condition,
            {"prediction": prediction, **answer_metrics(prediction, qa["answer"])},
        )

    work = [(entry, condition) for entry in panel for condition in config["review"]["conditions"]]
    task_ids = [f"{entry['qa']['qa_id']}/{condition}" for entry, condition in work]
    answers = run_bounded(answer_one, work, task_ids, client, config, "diagnose")
    indexed = {entry["qa"]["qa_id"]: entry for entry in panel}
    for qa_id, condition, result in answers:
        indexed[qa_id][condition] = result
    output = {"panel": panel, "logical_calls": client.calls}
    save_json(directory / "diagnostics.json", output)
    for agent_index in range(config["review"]["subagents"]):
        assigned = panel[agent_index :: config["review"]["subagents"]]
        save_json(directory / f"review-panel-{agent_index + 1}.json", assigned)
    return output


def combine_review(automated: dict, manual: list[dict], all_ids: set[str]) -> dict:
    """Apply resolved model-review decisions and union discovered duplicate facts."""
    decisions = {row["qa_id"]: dict(row) for row in automated["decisions"]}
    parent = {qa_id: qa_id for qa_id in all_ids}

    def find(qa_id: str) -> str:
        while parent[qa_id] != qa_id:
            parent[qa_id] = parent[parent[qa_id]]
            qa_id = parent[qa_id]
        return qa_id

    groups = list(automated["same_fact_groups"])
    for row in manual:
        if row["qa_id"] not in all_ids:
            raise ValueError(f"Unknown reviewed QA: {row['qa_id']}")
        if not row["accepted"]:
            decisions[row["qa_id"]] = {
                "qa_id": row["qa_id"],
                "accepted": False,
                "reason": row["reason"],
            }
        if row["same_fact_with"]:
            groups.append([row["qa_id"], *row["same_fact_with"]])
    for group in groups:
        for qa_id in group:
            if qa_id not in parent:
                raise ValueError(f"Unknown duplicate QA: {qa_id}")
            parent[find(qa_id)] = find(group[0])
    merged = {}
    for qa_id in sorted(all_ids):
        merged.setdefault(find(qa_id), []).append(qa_id)
    return {
        "decisions": list(decisions.values()),
        "same_fact_groups": [g for g in merged.values() if len(g) > 1],
    }


def finalize(config: dict) -> dict:
    directory = Path(config["output"]["artifacts_dir"])
    frozen_config, prompts = load_frozen_batch(config)
    output_directory = Path(config["output"]["dataset_dir"])
    output_directory.mkdir(parents=True, exist_ok=True)
    selection = json.loads((directory / "selection.json").read_text())
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    # The root agent resolves flagged items before writing this exact review contract.
    resolved = json.loads((directory / "resolved-review.json").read_text())
    panel_ids = {entry["qa"]["qa_id"] for entry in diagnostics["panel"]}
    reviewed_ids = [row["qa_id"] for row in resolved]
    if len(set(reviewed_ids)) != len(reviewed_ids) or set(reviewed_ids) != panel_ids:
        raise ValueError("Resolved review must cover each panel question exactly once")
    records, details = [], []
    for trajectory in selection["trajectories"]:
        doc = json.loads(
            (directory / "documents" / f"{trajectory['trajectory_id']}.json").read_text()
        )
        accepted = [qa for segment in doc["segments"] for qa in segment["accepted"]]
        qa_ids = {qa["qa_id"] for qa in accepted}
        rows = [row for row in resolved if row["qa_id"] in qa_ids]
        review = combine_review(doc["review"], rows, qa_ids)
        result = assemble_trajectory(trajectory, accepted, review, config)
        save_json(directory / "final-documents" / f"{trajectory['trajectory_id']}.json", result)
        if result["complete"]:
            validate_trajectory(result["trajectory"], config)
            records.append(result["trajectory"])
        details.append(
            {
                "trajectory_id": trajectory["trajectory_id"],
                "complete_before_review": doc["assembly"]["complete"],
                "complete_after_review": result["complete"],
                "per_segment": result["per_segment"],
            }
        )
    metrics = {}
    for condition in config["review"]["conditions"]:
        metrics[condition] = {
            metric: sum(entry[condition][metric] for entry in diagnostics["panel"]) / len(panel_ids)
            if panel_ids
            else None
            for metric in ("em", "f1")
        }
    summary = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S UTC+08:00"),
        "config": frozen_config,
        "prompts": prompts,
        "source_provenance": selection["source_provenance"],
        "source_statistics": selection["statistics"],
        "frozen_documents": len(selection["trajectories"]),
        "complete_documents": len(records),
        "final_qa_count": sum(len(record["qas"]) for record in records),
        "review_panel_count": len(panel_ids),
        "review_rejected_count": sum(not row["accepted"] for row in resolved),
        "answer_metrics": metrics,
        "documents": details,
        "review_type": "model_review_with_root_agent_adjudication",
    }
    request_stats = {}
    for line in (directory / "requests.jsonl").read_text().splitlines():
        call = json.loads(line)
        stats = request_stats.setdefault(
            call["stage"],
            {
                "logical_calls": 0,
                "cache_hits": 0,
                "attempts": 0,
                "failed_attempts": 0,
                "attempt_elapsed_seconds": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "attempts_without_usage": 0,
            },
        )
        stats["logical_calls"] += 1
        stats["cache_hits"] += call["cache_hit"]
        for attempt in call["attempts"]:
            stats["attempts"] += 1
            stats["failed_attempts"] += not attempt["ok"]
            stats["attempt_elapsed_seconds"] += attempt["elapsed_seconds"]
            usage = attempt.get("usage")
            if usage is None:
                stats["attempts_without_usage"] += 1
            else:
                stats["input_tokens"] += usage["input_tokens"]
                stats["output_tokens"] += usage["output_tokens"]
    summary["requests_by_stage"] = request_stats
    temporary = output_directory / "train.jsonl.tmp"
    with temporary.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output_directory / "train.jsonl")
    save_json(output_directory / "preparation.json", summary)
    return summary
