"""LoCoMo 四条件诊断：固定分层抽样、已有结果复用和逐题恢复。"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import torch
from transformers import set_seed

from latent_working_memory.benchmarks.data import load_samples
from latent_working_memory.benchmarks.protocol import (
    comparison_config,
    data_identity,
    encode_prompt,
    protocol,
)
from latent_working_memory.benchmarks.scoring import score_sample, summarize_scores
from latent_working_memory.runtime import initialize_device
from latent_working_memory.v4.checkpoint import read_checkpoint
from latent_working_memory.v4.evaluation_runtime import load_evaluation_inputs
from latent_working_memory.v4.exploration.memory_failure.constants import CONDITIONS, DESCRIPTIONS
from latent_working_memory.v4.exploration.memory_failure.model import DiagnosticMemoryLM
from latent_working_memory.v4.exploration.memory_failure.report import build_report
from latent_working_memory.v4.io import write_json


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


@contextmanager
def locked(path, blocking=False):
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(f"another process owns {path}") from error
        yield


def select_samples(samples, samples_per_conversation, seed):
    """每个对话等量；类别按最大余数法分配，不读取预测结果。"""
    if samples_per_conversation < 1 or seed < 0:
        raise ValueError("samples_per_conversation must be positive; selection seed nonnegative")
    conversations = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        conversations[sample.metadata["conversation_id"]][sample.metadata["category"]].append(sample)
    selected_ids, statistics = set(), []
    for conversation_id, categories in sorted(conversations.items()):
        available = sum(len(rows) for rows in categories.values())
        if samples_per_conversation > available:
            raise ValueError(f"{conversation_id} has only {available} questions")
        numerators = {
            category: samples_per_conversation * len(rows) for category, rows in categories.items()
        }
        quotas = {category: numerator // available for category, numerator in numerators.items()}
        remainder = samples_per_conversation - sum(quotas.values())
        priority = sorted(categories, key=lambda key: (-(numerators[key] % available), key))
        for category in priority[:remainder]:
            quotas[category] += 1
        for category, rows in categories.items():
            ordered = sorted(
                rows,
                key=lambda sample: hashlib.sha256(f"{seed}:{sample.id}".encode()).digest(),
            )
            selected_ids.update(sample.id for sample in ordered[:quotas[category]])
        statistics.append({
            "conversation_id": conversation_id,
            "available": available,
            "selected": samples_per_conversation,
            "categories": [
                {"category": key, "available": len(categories[key]), "selected": quotas[key]}
                for key in sorted(categories)
            ],
        })
    selected = [sample for sample in samples if sample.id in selected_ids]
    return selected, {
        "seed": seed,
        "samples_per_conversation": samples_per_conversation,
        "sample_ids": [sample.id for sample in selected],
        "conversations": statistics,
    }


def prepare_plan(args, checkpoint, config, tokenizer, backbone_config, samples):
    selected, selection = select_samples(
        samples, args.samples_per_conversation, args.selection_seed,
    )
    training = checkpoint["run"]["training"]
    plan = {
        "created_at": now(),
        "checkpoint": {
            "path": str(args.checkpoint),
            "step": checkpoint["cursor"]["step"],
            "training_run": training["output_dir"],
        },
        "model": asdict(config),
        "model_window": backbone_config.max_position_embeddings,
        "eos_token_id": tokenizer.eos_token_id,
        "seed": training["seed"],
        "data": data_identity(samples, args.benchmark_data, "locomo"),
        "protocol": protocol("locomo", (), "four diagnostic conditions; see condition descriptions"),
        "selection": selection,
    }
    if tokenizer.eos_token_id is None:
        raise ValueError("generation requires tokenizer.eos_token_id")
    with locked(args.output_dir / ".plan.lock", blocking=True):
        path = args.output_dir / "plan.json"
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if {k: v for k, v in saved.items() if k != "created_at"} != {
                k: v for k, v in plan.items() if k != "created_at"
            }:
                raise ValueError("diagnostic plan differs; use a new output directory")
            plan = saved
        else:
            write_json(path, plan)
    encoded = {
        sample.id: encode_prompt(sample, tokenizer, plan["model_window"], "locomo")
        for sample in selected
    }
    return plan, selected, encoded


def import_records(directory, condition, plan, samples, encoded, tokenizer):
    """只接收公共评估器的完整 canonical 结果；导入后重算所选子集。"""
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    expected = {**plan, "benchmark": "locomo"}
    if comparison_config(summary) != comparison_config(expected):
        raise ValueError(f"baseline/model/data/generation protocol differs: {directory}")
    if condition == "full-context":
        if summary["method"] != "full-context":
            raise ValueError(f"{directory} is not a full-context result")
    else:
        if summary["model"] != plan["model"] or summary["seed"] != plan["seed"]:
            raise ValueError(f"TTT model or seed differs: {directory}")
        if summary["checkpoint"] != {
            "path": plan["checkpoint"]["path"], "step": plan["checkpoint"]["step"],
        } or summary["training_run"] != plan["checkpoint"]["training_run"]:
            raise ValueError(f"TTT checkpoint differs: {directory}")
    rows = [
        json.loads(line)
        for line in (directory / "details.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or len(rows) != summary["samples"]:
        raise ValueError(f"duplicate IDs or incomplete details: {directory}")
    if len(rows) != plan["data"]["samples"]:
        raise ValueError(f"import requires the complete LoCoMo evaluation: {directory}")
    selected = []
    for sample in samples:
        row = by_id[sample.id]
        for name, expected_value in (
            ("metadata", sample.metadata),
            ("references", list(sample.references)),
            ("prompt_tokens", len(encoded[sample.id])),
            ("generation_budget", sample.max_new_tokens),
        ):
            if row[name] != expected_value:
                raise ValueError(f"{directory}: {sample.id} differs at {name}")
        generated = row["generated_token_ids"]
        if not generated or len(generated) > sample.max_new_tokens:
            raise ValueError(f"{sample.id}: invalid generated length")
        if row["generated_tokens"] != len(generated):
            raise ValueError(f"{sample.id}: generated token count differs")
        if tokenizer.decode(generated, skip_special_tokens=True) != row["prediction"]:
            raise ValueError(f"{sample.id}: prediction does not match current tokenizer")
        reason = "eos" if generated[-1] == tokenizer.eos_token_id else "max_new_tokens"
        if row["finish_reason"] != reason or (
            reason == "max_new_tokens" and len(generated) != sample.max_new_tokens
        ):
            raise ValueError(f"{sample.id}: inconsistent generation finish reason")
        score = score_sample("locomo", sample, row["prediction"])
        if not math.isclose(score, row["score"], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{sample.id}: stored score differs from the current scorer")
        selected.append({**row, "score": score})
    return selected


def existing_records(path, sample_ids):
    """恢复已刷新的完整行；中断留下的最后一条不完整行重新生成。"""
    if not path.is_file():
        return []
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        content = content[:content.rfind(b"\n") + 1]
        path.write_bytes(content)
    rows = [json.loads(line) for line in content.splitlines()]
    if [row["id"] for row in rows] != sample_ids[:len(rows)]:
        raise ValueError(f"partial results are not the selected sample prefix: {path}")
    return rows


def compute_records(model, condition, samples, encoded, tokenizer, device, path, rows, status_path):
    model.condition = condition
    with path.open("a", encoding="utf-8") as stream:
        for sample in samples[len(rows):]:
            ids = torch.tensor(encoded[sample.id], dtype=torch.long, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.no_grad():
                output = model.generate_condition(ids, condition, tokenizer, sample.max_new_tokens)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            generated = output[len(ids):].tolist()
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            row = {
                "id": sample.id,
                "metadata": sample.metadata,
                "references": list(sample.references),
                "prediction": prediction,
                "score": score_sample("locomo", sample, prediction),
                "prompt_tokens": len(ids),
                "generated_tokens": len(generated),
                "generated_token_ids": generated,
                "generation_budget": sample.max_new_tokens,
                "finish_reason": (
                    "eos" if generated[-1] == tokenizer.eos_token_id else "max_new_tokens"
                ),
                "generation_seconds": elapsed,
                "boundary_events": model.boundary_events,
                "memory_update_events": model.memory_update_events,
            }
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            write_json(status_path, {
                "state": "running", "condition": condition, "completed": len(rows),
                "samples": len(samples), "updated_at": now(),
            })
            print(
                f"{condition} {len(rows)}/{len(samples)} | "
                f"f1={row['score']:.4f} | {elapsed:.2f}s",
                flush=True,
            )
    return rows


def evaluate(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("diagnostics use one process per GPU; do not launch with torchrun")
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("conditions must be unique")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.benchmark_data = args.benchmark_data.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = read_checkpoint(args.checkpoint)
    config, tokenizer, backbone_config = load_evaluation_inputs(checkpoint)
    samples = load_samples("locomo", args.benchmark_data)
    plan, selected, encoded = prepare_plan(
        args, checkpoint, config, tokenizer, backbone_config, samples,
    )
    imports = {"full-context": args.full_context_results, "ttt": args.ttt_results}
    model, device = None, None
    for condition in CONDITIONS:
        if condition not in args.conditions:
            continue
        directory = args.output_dir / condition
        directory.mkdir(exist_ok=True)
        status_path = directory / "status.json"
        with locked(directory / ".evaluation.lock"):
            source_dir = imports.get(condition)
            if source_dir is not None:
                source_dir = source_dir.expanduser().resolve()
            source = {
                "kind": "imported" if source_dir is not None else "computed",
                "directory": str(source_dir) if source_dir is not None else None,
            }
            condition_config = {"condition": condition, "source": source}
            config_path = directory / "eval.json"
            if config_path.is_file():
                if json.loads(config_path.read_text(encoding="utf-8")) != condition_config:
                    raise ValueError(f"result source differs at {config_path}; keep the original args")
            else:
                write_json(config_path, condition_config)
            if (directory / "summary.json").is_file():
                print(f"{condition}: reusing completed diagnostic results", flush=True)
                continue
            path = directory / "details.jsonl"
            try:
                write_json(status_path, {
                    "state": "running", "condition": condition, "updated_at": now(),
                    "samples": len(selected),
                })
                if source_dir is not None:
                    if path.exists():
                        rows = existing_records(path, plan["selection"]["sample_ids"])
                        if len(rows) != len(selected):
                            raise ValueError(f"incomplete atomic import: {path}")
                    else:
                        rows = import_records(
                            source_dir, condition, plan, selected, encoded, tokenizer,
                        )
                        temporary = directory / "details.jsonl.tmp"
                        with temporary.open("w", encoding="utf-8") as stream:
                            for row in rows:
                                stream.write(
                                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                )
                        temporary.replace(path)
                else:
                    rows = existing_records(path, plan["selection"]["sample_ids"])
                    if len(rows) < len(selected):
                        if model is None:
                            set_seed(plan["seed"])
                            device = initialize_device(args.device)
                            model = DiagnosticMemoryLM(config).to(device).eval()
                            model.load_trainable_state_dict(checkpoint["memory"])
                            model.requires_grad_(False)
                        rows = compute_records(
                            model, condition, selected, encoded, tokenizer, device, path,
                            rows, status_path,
                        )
                summary = {
                    "condition": condition,
                    "description": DESCRIPTIONS[condition],
                    "samples": len(rows),
                    "metrics": summarize_scores("locomo", rows),
                    "source": source,
                }
                write_json(directory / "summary.json", summary)
                write_json(status_path, {
                    "state": "completed", "condition": condition, "completed": len(rows),
                    "samples": len(selected), "updated_at": now(),
                })
                print(f"{condition}: F1={summary['metrics']['score']:.4%}", flush=True)
            except BaseException as error:
                write_json(status_path, {
                    "state": "failed", "condition": condition, "updated_at": now(),
                    "error": f"{type(error).__name__}: {error}",
                })
                raise
    return args.output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--benchmark-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--samples-per-conversation", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--full-context-results", type=Path, help="完整基线的 locomo 结果目录")
    parser.add_argument("--ttt-results", type=Path, help="同 checkpoint 的完整 locomo 结果目录")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output_dir = evaluate(args)
    build_report(output_dir)
    print(f"report: {output_dir / 'report' / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
