"""Evaluate E²-TTT checkpoints with the pinned harness and local benchmark files."""

import argparse
from datetime import datetime
import json
import logging
import os
import random
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import datasets
import e2_ttt  # noqa: F401 -- registers the original checkpoint model types
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable, set_torch_seed
import numpy as np
import torch
import transformers

from local_tasks import LocalRuler, LocalSQuAD, LocalSWDE


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RULER_TASKS = ("niah_single_1", "niah_single_2")


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=handle_non_serializable) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmarks",
        choices=["ruler", "squad_completion", "swde"],
        nargs="+",
        default=["ruler", "squad_completion", "swde"],
    )
    parser.add_argument(
        "--ruler-data",
        type=Path,
        default=PROJECT_ROOT / "data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b",
    )
    parser.add_argument(
        "--squad-data", type=Path, default=PROJECT_ROOT / "data/benchmarks/based-squad"
    )
    parser.add_argument(
        "--swde-data", type=Path, default=PROJECT_ROOT / "data/benchmarks/based-swde"
    )
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384]
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        parser.error("Use one physical GPU: CUDA_VISIBLE_DEVICES=0 or 1.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This evaluation requires one process.")
    if len(args.benchmarks) != len(set(args.benchmarks)):
        parser.error("--benchmarks must not contain duplicates.")
    if len(args.lengths) != len(set(args.lengths)) or any(n <= 128 for n in args.lengths):
        parser.error("--lengths must contain distinct integers greater than 128.")
    for name in ("checkpoint", "output_dir", "ruler_data", "squad_data", "swde_data"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not (args.checkpoint / "config.json").is_file():
        parser.error(f"No Hugging Face config.json under {args.checkpoint}")
    if not torch.cuda.is_available():
        parser.error("The method environment must provide a CUDA-enabled PyTorch runtime.")
    return args


def execute(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    native_dir = args.output_dir / "native_harness"
    native_dir.mkdir()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "evaluation.log")],
        force=True,
    )
    parquet = {
        "squad_completion": args.squad_data / "data/validation-00000-of-00001.parquet",
        "swde": args.swde_data / "data/validation-00000-of-00001.parquet",
    }
    sources = {name: str(parquet[name]) for name in args.benchmarks if name != "ruler"}
    if "ruler" in args.benchmarks:
        sources["ruler"] = str(args.ruler_data)
    plan = {
        "created_at": timestamp(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_config": json.loads((args.checkpoint / "config.json").read_text()),
        "benchmarks": args.benchmarks,
        "data": sources,
        "ruler_lengths": args.lengths if "ruler" in args.benchmarks else [],
        "ruler_tasks": list(RULER_TASKS) if "ruler" in args.benchmarks else [],
        "ruler_samples_per_task_length": 500,
        "ruler_generation_tokens": 128,
        "recall_total_token_limit": 2048,
        "recall_input_token_limit": 2000,
        "recall_generation_tokens": 48,
        "recall_truncation": "left",
        "ruler_truncation": "forbidden",
        "prompt_format": "base; RULER input + answer_prefix once",
        "batch_size": 1,
        "num_fewshot": 0,
        "dtype": "bfloat16",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "processes": 1,
        "bootstrap_iters": 0,
        "seeds": {"random": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "e2_ttt": e2_ttt.__version__,
        },
    }
    write_json(args.output_dir / "plan.json", plan)
    summary = {"checkpoint": str(args.checkpoint), "completed": False, "metrics": {}}
    write_json(args.output_dir / "summary.json", summary)
    started = time.monotonic()
    try:
        lm = HFLM(
            pretrained=str(args.checkpoint),
            tokenizer=str(args.checkpoint),
            dtype="bfloat16",
            device="cuda:0",
            batch_size=1,
            max_length=2048,
            truncation=False,
            trust_remote_code=False,
        )
        if lm.world_size != 1:
            raise ValueError("The harness initialized more than one evaluation process.")
        with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
            for benchmark in ("squad_completion", "swde"):
                if benchmark not in args.benchmarks:
                    continue
                task_class = LocalSQuAD if benchmark == "squad_completion" else LocalSWDE
                task = task_class(parquet[benchmark])
                result = evaluate_tasks(lm, [task])
                save_result(result, benchmark, native_dir, predictions)
                metric = result["results"][benchmark]["contains,none"]
                summary["metrics"][benchmark] = {
                    "samples": len(task.eval_docs),
                    "accuracy_percent": 100 * metric,
                }
                write_json(args.output_dir / "summary.json", summary)
            if "ruler" in args.benchmarks:
                # Reuse the same loaded model; only the harness sequence budget changes.
                ruler_lm = HFLM(
                    pretrained=lm.model,
                    tokenizer=lm.tokenizer,
                    device="cuda:0",
                    batch_size=1,
                    max_length=max(args.lengths),
                    truncation=False,
                    add_bos_token=lm.add_bos_token,
                )
                summary["metrics"]["ruler"] = {}
                for length in args.lengths:
                    tasks = [
                        LocalRuler(
                            args.ruler_data / str(length) / name / "test.jsonl",
                            name,
                            length,
                            ruler_lm,
                        )
                        for name in RULER_TASKS
                    ]
                    result = evaluate_tasks(ruler_lm, tasks)
                    save_result(result, f"ruler_{length}", native_dir, predictions)
                    summary["metrics"]["ruler"][str(length)] = {
                        name: {
                            "samples": 500,
                            "accuracy_percent": 100
                            * result["results"][f"{name}_{length}"][f"{length},none"],
                        }
                        for name in RULER_TASKS
                    }
                    write_json(args.output_dir / "summary.json", summary)
        summary["completed"] = True
        write_json(args.output_dir / "summary.json", summary)
        write_json(
            args.output_dir / "execution.json",
            {"finished_at": timestamp(), "exit_code": 0, "elapsed_seconds": time.monotonic() - started},
        )
    except Exception as exc:
        write_json(
            args.output_dir / "execution.json",
            {
                "finished_at": timestamp(),
                "exit_code": 1,
                "elapsed_seconds": time.monotonic() - started,
                "error": repr(exc),
            },
        )
        raise


def evaluate_tasks(lm, tasks):
    # Match the native simple_evaluate defaults without TaskManager's YAML lookup.
    random.seed(0)
    np.random.seed(1234)
    set_torch_seed(1234)
    for task in tasks:
        task.set_fewshot_seed(1234)
    return evaluator.evaluate(
        lm=lm,
        task_dict={task.config.task: task for task in tasks},
        log_samples=True,
        apply_chat_template=False,
        cache_requests=False,
        bootstrap_iters=0,
    )


def save_result(result, name, native_dir, predictions):
    samples = result.pop("samples")
    write_json(native_dir / f"{name}.json", result)
    for task, rows in samples.items():
        for row in rows:
            predictions.write(
                json.dumps(
                    {"task": task, **row},
                    ensure_ascii=False,
                    default=handle_non_serializable,
                )
                + "\n"
            )
    predictions.flush()


if __name__ == "__main__":
    execute(parse_args())
