"""Run original OpenCompass generation and scoring over preconstructed RULER rows."""

import argparse
from datetime import datetime
import importlib.metadata
import logging
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo
import time

import inference_model  # noqa: F401 -- registers upstream TTT models before loading.
from opencompass.datasets.ruler.ruler_cwe import RulerCweEvaluator
from opencompass.datasets.ruler.ruler_fwe import RulerFweEvaluator
from opencompass.datasets.ruler.ruler_niah import RulerNiahEvaluator
from opencompass.datasets.ruler.ruler_qa import RulerQaEvaluator
from opencompass.datasets.ruler.ruler_vt import RulerVtEvaluator
from opencompass.models import HuggingFaceBaseModel
from opencompass.utils.text_postprocessors import extract_non_reasoning_content


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


def read_rows(path, count):
    with path.open() as stream:
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


def evaluate(plan):
    data, output = Path(plan["ruler_data"]), Path(plan["output_dir"])
    budget = plan["tokens_to_generate"]
    wrapper = HuggingFaceBaseModel(
        path=plan["checkpoint"],
        model_kwargs={
            "device_map": "cuda:0",
            "torch_dtype": "torch.bfloat16",
            "local_files_only": True,
        },
        tokenizer_kwargs={"local_files_only": True},
        max_seq_len=plan["max_seq_len"],
        generation_kwargs={"do_sample": False, "top_k": 20, "top_p": 0.95, "use_cache": True},
    )
    if not type(wrapper.model).__module__.startswith("inference_model."):
        raise TypeError(
            "The checkpoint was not loaded through the upstream TTT model registration."
        )
    config = wrapper.model.config
    runtime = {
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "opencompass"]
        },
        "model_class": f"{type(wrapper.model).__module__}.{type(wrapper.model).__name__}",
        "ttt_mode": config.ttt_mode,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_scaling": config.rope_scaling,
        "max_seq_len": wrapper.max_seq_len,
        "attention_implementation": config._attn_implementation,
        "postprocessor": "opencompass.utils.text_postprocessors.extract_non_reasoning_content",
    }
    (output / "runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n")
    logging.info("Runtime: %s", json.dumps(runtime, ensure_ascii=False))
    summary = {"checkpoint": plan["checkpoint"], "lengths": {}, "samples": 0}
    with (output / "predictions.jsonl").open("w") as predictions_file:
        for length in plan["lengths"]:
            length_results = {}
            for task, evaluator_class in TASKS.items():
                source = data / str(length) / task / "test.jsonl"
                rows = read_rows(source, plan["samples_per_task"])
                predictions, gold = [], []
                evaluator = evaluator_class()
                for index, row in enumerate(rows):
                    prompt = row["input"] + row["answer_prefix"]
                    input_tokens = len(
                        wrapper.tokenizer.encode(prompt, add_special_tokens=True, truncation=False)
                    )
                    if input_tokens + budget > length:
                        raise ValueError(
                            f"{source}:{index + 1}: {input_tokens} input + {budget} output "
                            f"exceeds the {length}-token budget; truncation is forbidden"
                        )
                    start = time.perf_counter()
                    # Every generate call creates a new upstream cache, including TTT state.
                    raw = wrapper.generate(inputs=[prompt], max_out_len=budget)[0]
                    prediction = extract_non_reasoning_content(raw)
                    record = {
                        "length": length,
                        "task": task,
                        "row": index,
                        "source": str(source),
                        "input_tokens": input_tokens,
                        "raw_prediction": raw,
                        "prediction": prediction,
                        "gold": row["outputs"],
                        "native_score": evaluator.score([prediction], [row["outputs"]]),
                        "seconds": time.perf_counter() - start,
                    }
                    predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    predictions_file.flush()
                    predictions.append(prediction)
                    gold.append(row["outputs"])
                    logging.info("%s/%s: %s/%s", length, task, index + 1, len(rows))
                native_score = evaluator.score(predictions=predictions, gold=gold)
                length_results[task] = native_score
                result_dir = output / "results" / str(length)
                result_dir.mkdir(parents=True, exist_ok=True)
                (result_dir / f"{task}.json").write_text(
                    json.dumps(native_score, ensure_ascii=False, indent=2) + "\n"
                )
                summary["samples"] += len(rows)
            average = sum(result["score"] for result in length_results.values()) / len(TASKS)
            summary["lengths"][str(length)] = {
                "tasks": length_results,
                "score": average,
            }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ruler-data",
        type=Path,
        default=root / "data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768, 65536, 131072, 262144],
    )
    args = parser.parse_args()
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if physical_gpu not in {"0", "1"}:
        parser.error("CUDA_VISIBLE_DEVICES must select a single physical GPU: 0 or 1.")
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    if any(length <= 64 for length in args.lengths) or len(set(args.lengths)) != len(args.lengths):
        parser.error("--lengths must contain distinct integers greater than 64.")
    checkpoint = args.checkpoint.expanduser().resolve()
    data = args.ruler_data.expanduser().resolve()
    if not checkpoint.is_dir():
        parser.error(f"Checkpoint directory does not exist: {checkpoint}")
    for length in args.lengths:
        for task in TASKS:
            path = data / str(length) / task / "test.jsonl"
            if not path.is_file():
                parser.error(f"RULER data file does not exist: {path}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "checkpoint": str(checkpoint),
        "ruler_data": str(data),
        "lengths": args.lengths,
        "samples_per_task": 100,
        "tokens_to_generate": 64,
        "prompt": "input + answer_prefix; base completion; no chat template",
        "max_seq_len": max(args.lengths),
        "cuda_visible_devices": physical_gpu,
        "device": "cuda:0",
        "output_dir": str(output),
    }
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(output / "evaluation.log"), logging.StreamHandler()],
    )
    try:
        evaluate(plan)
    except Exception as error:
        logging.exception("Evaluation failed")
        (output / "execution.json").write_text(
            json.dumps({"exit_code": 1, "error": str(error)}, ensure_ascii=False) + "\n"
        )
        raise
    (output / "execution.json").write_text(json.dumps({"exit_code": 0}) + "\n")


if __name__ == "__main__":
    main()
