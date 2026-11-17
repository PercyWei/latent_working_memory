"""In-Place TTT 的评估协议、全局样本记录与原生 RULER 汇总。"""

from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from evaluation_runtime import write_json
from opencompass.datasets.ruler.ruler_cwe import RulerCweEvaluator
from opencompass.datasets.ruler.ruler_fwe import RulerFweEvaluator
from opencompass.datasets.ruler.ruler_niah import RulerNiahEvaluator
from opencompass.datasets.ruler.ruler_qa import RulerQaEvaluator
from opencompass.datasets.ruler.ruler_vt import RulerVtEvaluator
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

from in_place_ttt.evaluation.provenance import UPSTREAM_REVISION, capture_source


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RULER_DATA = PROJECT_ROOT / "data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b"
LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
GENERATION_BUDGET = 64
SAMPLES_PER_TASK_LENGTH = 100
TASKS = {
    "niah_single_1": RulerNiahEvaluator,
    "niah_single_2": RulerNiahEvaluator,
    "niah_single_3": RulerNiahEvaluator,
    "niah_multikey_1": RulerNiahEvaluator,
    "niah_multikey_2": RulerNiahEvaluator,
    "niah_multikey_3": RulerNiahEvaluator,
    "niah_multivalue": RulerNiahEvaluator,
    "niah_multiquery": RulerNiahEvaluator,
    "vt": RulerVtEvaluator,
    "cwe": RulerCweEvaluator,
    "fwe": RulerFweEvaluator,
    "qa_1": RulerQaEvaluator,
    "qa_2": RulerQaEvaluator,
}
MODEL_KWARGS = {
    "device_map": "cuda:0",
    "torch_dtype": "torch.bfloat16",
    "local_files_only": True,
}
TOKENIZER_KWARGS = {"local_files_only": True}
GENERATION_KWARGS = {
    "do_sample": False,
    "top_k": 20,
    "top_p": 0.95,
    "use_cache": True,
}


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S Asia/Shanghai")


def build_protocol(checkpoint, lengths, inference_mode, mlp_chunk_size):
    checkpoint = Path(checkpoint).expanduser().resolve()
    lengths = list(lengths)
    if not lengths or len(lengths) != len(set(lengths)) or any(length not in LENGTHS for length in lengths):
        raise ValueError(f"--lengths must select distinct values from {LENGTHS}")
    if inference_mode not in ("original", "optimized"):
        raise ValueError(f"Unknown inference mode: {inference_mode}")
    if mlp_chunk_size < 0 or (inference_mode == "original" and mlp_chunk_size != 0):
        raise ValueError("--mlp-chunk-size must be nonnegative and original requires 0")
    return {
        "benchmark": "ruler",
        "upstream_revision": UPSTREAM_REVISION,
        "checkpoint": str(checkpoint),
        "checkpoint_config": json.loads((checkpoint / "config.json").read_text()),
        "inference_mode": inference_mode,
        "mlp_chunk_size": mlp_chunk_size,
        "compact_tails": inference_mode == "optimized",
        "ruler_data": str(RULER_DATA),
        "lengths": lengths,
        "tasks": list(TASKS),
        "samples_per_task": SAMPLES_PER_TASK_LENGTH,
        "tokens_to_generate": GENERATION_BUDGET,
        "prompt": "input + answer_prefix; base completion; no chat template",
        "model_wrapper": "in_place_ttt.evaluation.runtime.EvaluationModel",
        "generation_path": (
            "OpenCompass HuggingFaceBaseModel.generate"
            if inference_mode == "original"
            else "in_place_ttt.evaluation.runtime.generate_batch; single tokenization, left padding and native generation parameters"
        ),
        "model_kwargs": MODEL_KWARGS.copy(),
        "tokenizer_kwargs": TOKENIZER_KWARGS.copy(),
        "generation_kwargs": GENERATION_KWARGS.copy(),
        "postprocessor": "opencompass.utils.text_postprocessors.extract_non_reasoning_content",
        "max_seq_len": max(lengths),
    }


def read_rows(path, count):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != count:
        raise ValueError(f"{path}: expected {count} rows, got {len(rows)}")
    for index, row in enumerate(rows):
        if not isinstance(row["input"], str) or not isinstance(row["answer_prefix"], str):
            raise ValueError(f"{path}:{index + 1}: input and answer_prefix must be strings")
        if not isinstance(row["outputs"], list) or not row["outputs"]:
            raise ValueError(f"{path}:{index + 1}: outputs must be a nonempty list")
        if any(not isinstance(value, str) or not value for value in row["outputs"]):
            raise ValueError(f"{path}:{index + 1}: outputs must contain nonempty strings")
    return rows


def load_samples(protocol):
    expected = {}
    for length in protocol["lengths"]:
        for task in protocol["tasks"]:
            source = Path(protocol["ruler_data"]) / str(length) / task / "test.jsonl"
            for index, row in enumerate(read_rows(source, protocol["samples_per_task"])):
                sample_id = f"ruler/{length}/{task}/{index}"
                expected[sample_id] = length, task, index, str(source), row
    return expected


def validate_sample_ids(ids, expected):
    if (
        not isinstance(ids, list)
        or any(not isinstance(sample_id, str) for sample_id in ids)
        or len(ids) != len(set(ids))
        or not set(ids) <= expected.keys()
    ):
        raise ValueError("Assigned sample IDs must be a unique JSON list within the evaluation protocol")
    return {sample_id: expected[sample_id] for sample_id in ids}


def validate_prediction(record, expected):
    sample_id = record["id"]
    if sample_id not in expected:
        raise ValueError(f"Prediction is outside the assigned samples: {sample_id}")
    length, task, index, source, row = expected[sample_id]
    if (
        record["length"] != length
        or record["task"] != task
        or record["row"] != index
        or record["source"] != source
        or record["source_index"] != row["index"]
        or record["gold"] != row["outputs"]
    ):
        raise ValueError(f"Saved prediction does not match current data: {sample_id}")
    token_ids = record["token_ids"]
    if (
        not isinstance(token_ids, list)
        or any(type(value) is not int or value < 0 for value in token_ids)
        or record["generated_tokens"] != len(token_ids)
        or len(token_ids) > GENERATION_BUDGET
    ):
        raise ValueError(f"Saved generated token IDs or token count are invalid: {sample_id}")
    prediction = extract_non_reasoning_content(record["raw_prediction"])
    if prediction != record["prediction"]:
        raise ValueError(f"Saved prediction postprocessing does not match: {sample_id}")
    return sample_id


def collect_records(directory, protocol, expected):
    records, sources = {}, []
    global_runtime_path = directory / "runtime.json"
    global_runtime = json.loads(global_runtime_path.read_text()) if global_runtime_path.is_file() else None
    for path in sorted(directory.glob("attempts/attempt-*/shards/shard-*")):
        predictions = path / "predictions.jsonl"
        if not predictions.is_file():
            continue
        plan_path, execution_path, runtime_path = path / "plan.json", path / "execution.json", path / "runtime.json"
        plan = json.loads(plan_path.read_text())
        if plan["protocol"] != protocol:
            raise ValueError(f"Worker evaluation protocol differs: {plan_path}")
        assigned = validate_sample_ids(plan["sample_ids"], expected)
        execution = json.loads(execution_path.read_text())
        runtime = json.loads(runtime_path.read_text())
        seen = []
        ignored_tail = False
        with predictions.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.endswith(b"\n"):
                    ignored_tail = True
                    break
                record = json.loads(line)
                sample_id = validate_prediction(record, assigned)
                if sample_id in records:
                    raise ValueError(f"Duplicate committed prediction ID: {sample_id} ({path}:{line_number})")
                if record["execution_attempt"] != plan["execution_attempt"]:
                    raise ValueError(f"Prediction execution attempt differs: {sample_id} ({path})")
                records[sample_id] = record
                seen.append(sample_id)
        if seen:
            if global_runtime is None:
                raise FileNotFoundError(f"Committed predictions require the shared runtime: {global_runtime_path}")
            if runtime != global_runtime or execution["runtime"] != runtime:
                raise ValueError(f"Worker model, backend, dtype or dependency settings differ: {path}")
        sources.append({
            "directory": str(path),
            "plan_path": str(plan_path),
            "execution_path": str(execution_path),
            "runtime_path": str(runtime_path),
            "source": plan["source"],
            "physical_gpu": plan["physical_gpu"],
            "execution": execution,
            "sample_ids": seen,
            "samples": len(seen),
            "ignored_incomplete_tail": ignored_tail,
        })
    return records, sources


def build_summary(protocol, expected, records, output):
    if records.keys() != expected.keys():
        raise ValueError("Predictions do not exactly cover the planned samples")
    summary = {
        "benchmark": "ruler",
        "checkpoint": protocol["checkpoint"],
        "inference_mode": protocol["inference_mode"],
        "mlp_chunk_size": protocol["mlp_chunk_size"],
        "completed": True,
        "score_unit": "percent",
        "samples": len(records),
        "lengths": {},
    }
    group_scores = []
    for length in protocol["lengths"]:
        length_results = {}
        result_dir = output / "results" / str(length)
        result_dir.mkdir(parents=True, exist_ok=True)
        for task in protocol["tasks"]:
            ids = [sample_id for sample_id, sample in expected.items() if sample[:2] == (length, task)]
            native_score = TASKS[task]().score(
                predictions=[extract_non_reasoning_content(records[sample_id]["raw_prediction"]) for sample_id in ids],
                gold=[expected[sample_id][4]["outputs"] for sample_id in ids],
            )
            length_results[task] = native_score
            group_scores.append(native_score["score"])
            write_json(result_dir / f"{task}.json", native_score)
        summary["lengths"][str(length)] = {
            "tasks": length_results,
            "score": sum(result["score"] for result in length_results.values()) / len(protocol["tasks"]),
        }
    summary["task_length_macro_average"] = sum(group_scores) / len(group_scores)
    return summary


def merge_records(directory, protocol, expected, records, sources, output):
    if records.keys() != expected.keys():
        raise ValueError("Predictions do not exactly cover the planned samples")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    write_json(manifest_path, {"parts": sources})
    plan = {
        "created_at": timestamp(),
        "protocol": protocol,
        "source": capture_source(__file__, {}),
        "manifest_path": str(manifest_path),
        "samples": len(expected),
    }
    write_json(output / "plan.json", plan)
    write_json(output / "runtime.json", json.loads((directory / "runtime.json").read_text()))
    execution = {"started_at": timestamp(), "status": "running"}
    write_json(output / "execution.json", execution)
    started = time.perf_counter()
    try:
        summary = build_summary(protocol, expected, records, output)
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for sample_id in expected:
                stream.write(json.dumps(records[sample_id], ensure_ascii=False, allow_nan=False) + "\n")
        summary.update(
            plan_path=str(output / "plan.json"),
            manifest_path=str(manifest_path),
            worker_seconds_total=sum(item["execution"]["seconds"] for item in sources),
            peak_worker_memory_allocated_bytes=max(
                (item["execution"]["peak_memory_allocated_bytes"] for item in sources), default=0
            ),
        )
        write_json(output / "summary.json", summary)
        execution["status"] = "completed"
        return summary
    except BaseException as error:
        execution.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        execution.update(finished_at=timestamp(), merge_wall_seconds=time.perf_counter() - started)
        write_json(output / "execution.json", execution)
