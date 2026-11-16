"""在训练时的同一批 dev 上比较确定性策略与固定块数的质量和记忆成本。"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import torch
from transformers import AutoTokenizer, set_seed

from latent_working_memory.runtime import initialize_device
from ..cli import positive_int
from ..config import config_from_dict
from ..data import load_documents
from ..io import write_json
from ..modeling.compressor import MemoryModel
from ..training.checkpoint import checkpoint_file, read_checkpoint
from .evaluate import _check_complete_details
from .inference import EvaluationInput, EvaluationQuestion, compute_batch
from .records import completed_evaluation, evaluation_lock, prepare_evaluation
from .shards import commit_results, load_progress, write_work
from .test import make_record, sample_batches, summarize as summarize_qa


CONDITIONS = ("policy", "fixed-1", "fixed-2", "fixed-4", "fixed-8")


def fixed_actions(context_tokens, unit_tokens, blocks):
    """在处理单元网格上均匀放置边界；2、4、8 块的已有边界保持嵌套。"""
    units = (context_tokens + unit_tokens - 1) // unit_tokens
    if blocks > units:
        raise ValueError(f"cannot create {blocks} nonempty blocks from {units} units")
    boundaries = {index * units // blocks for index in range(1, blocks)}
    return tuple(index in boundaries for index in range(1, units))


def summarize(records):
    return {
        **summarize_qa(records),
        "memory_cost": sum(row["memory_cost"] for row in records) / len(records),
        "reward": sum(row["reward"] for row in records) / len(records),
    }


def _prepare(args, state, config, documents, statistics):
    output = Path(args.output_dir).expanduser().resolve()
    samples = tuple(documents.values())
    if not samples:
        raise ValueError("checkpoint dev selection is empty")
    if statistics["identity"] != state["binding"]["dev"]["identity"]:
        raise ValueError("dev documents or tokenized QA differ from the training checkpoint")
    unit_tokens = config.model.structure.unit_tokens
    for sample in samples:
        fixed_actions(len(sample.input_ids), unit_tokens, 8)
    plan = {
        "checkpoint": str(checkpoint_file(args.checkpoint)),
        "checkpoint_step": state["cursor"]["step"],
        "config": state["config"],
        "data": statistics["identity"],
        "data_file": statistics["file"],
        "documents": len(samples),
        "questions": sum(len(sample.qas) for sample in samples),
        "document_ids": [sample.id for sample in samples],
        "slot_cost": config.training.grpo.slot_cost,
        "memory_slots": config.model.structure.memory_slots,
        "unit_tokens": unit_tokens,
        "conditions": list(CONDITIONS),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": config.seed,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4], text=True,
        ).strip(),
        "protocol": {
            "selection": "exact checkpoint dev loader and tokenized data identity",
            "policy": "commit at probability >= 0.5; forced final pending flush",
            "fixed": "B blocks; boundaries floor(j * ceil(context_tokens / unit_tokens) / B), j=1..B-1; forced final pending flush",
            "input": "training document encoding; questions remain uncompressed; all QA share one document memory",
            "generation": "greedy; fresh generation config; identical per-question token budget",
            "nll": "QA-token mean, then QA mean per document, then document mean; assistant ending included",
            "scoring": "single-reference QA EM/token-overlap F1; normalized case/articles/punctuation",
            "memory_cost": "slot_cost * memory_slots * blocks",
            "reward": "-document QA NLL - memory_cost",
        },
    }
    plan_path = output / "plan.json"
    if args.resume:
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        changed = [key for key, value in plan.items() if saved[key] != value]
        if changed:
            raise ValueError(f"comparison plan differs: {', '.join(changed)}")
    else:
        if plan_path.exists() or any((output / name).exists() for name in CONDITIONS):
            raise FileExistsError(f"comparison output already exists; use --resume: {output}")
        write_json(plan_path, {
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            **plan,
        })
    return output, samples, plan


def _run(args):
    state = read_checkpoint(args.checkpoint)
    config = config_from_dict(state["config"])
    if config.stage != "post":
        raise ValueError("boundary comparison requires a post-training checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    documents, statistics = load_documents(config, tokenizer, "dev")
    output, samples, plan = _prepare(args, state, config, documents, statistics)
    pending = []
    for condition in CONDITIONS:
        directory = output / condition
        directory.mkdir(parents=True, exist_ok=True)
        request = {
            "eval_id": output.name, "task": "test", "condition": condition,
            "checkpoint": plan["checkpoint"], "checkpoint_step": plan["checkpoint_step"],
            "training_run": {"directory": config.training.run.output_dir},
            "model": {"model_name_or_path": config.model_name_or_path,
                      "structure": asdict(config.model.structure), "lora": asdict(config.model.lora),
                      "compute": asdict(config.compute)},
            "seed": config.seed, "data": plan["data"], "slot_cost": plan["slot_cost"],
            "protocol": {**plan["protocol"], "condition": condition,
                         "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size},
        }
        prepare_evaluation(directory, request)
        summary = _check_complete_details(
            directory, completed_evaluation(directory, request), samples,
        )
        if summary is not None:
            print(f"{condition}: complete ({len(samples)} documents)", flush=True)
            continue
        entries = load_progress(directory, request, samples, repair=True)
        pending.append((condition, directory, request, entries))
    if not pending:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run v3.1.1 boundary comparison")
    device = initialize_device("cuda")
    set_seed(config.seed)
    model = MemoryModel(config.model_name_or_path, asdict(config.model), asdict(config.compute), device)
    model.load_trainable_state_dict(state["model"])
    model.eval()
    del state, tokenizer
    execution = {"eval_batch_size_per_gpu": args.batch_size}
    for condition, directory, request, entries in pending:
        write_work(directory, samples, entries, 1)
        selected = [sample for sample in samples if sample.id not in entries]
        progress = directory / "progress"
        progress.mkdir(parents=True, exist_ok=True)
        with (progress / "worker_0.jsonl").open("a", encoding="utf-8") as stream:
            def commit(entry):
                row = entry["record"]
                row["condition"] = condition
                row["memory_cost"] = plan["slot_cost"] * row["memory_slots"]
                row["reward"] = -sum(qa["nll"] for qa in row["qas"]) / len(row["qas"]) - row["memory_cost"]
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                entries[row["id"]] = entry
                print(f"{condition} {len(entries)}/{len(samples)} {row['id']} "
                      f"blocks={row['blocks']} seconds={row['seconds']:.1f}", flush=True)

            for batch in sample_batches(selected, execution):
                actions = None if condition == "policy" else [
                    fixed_actions(len(sample.input_ids), plan["unit_tokens"], int(condition.split("-")[1]))
                    for sample in batch
                ]
                with torch.no_grad(), model.forward_context():
                    compute_batch(
                        batch, model, config, commit,
                        lambda sample: EvaluationInput(sample.input_ids, tuple(
                            EvaluationQuestion(qa.prompt_ids, args.max_new_tokens, qa.target_ids)
                            for qa in sample.qas
                        )),
                        make_record, actions=actions,
                    )
        summary = commit_results(directory, request, samples, entries, sys.modules[__name__])
        summary["resources"]["wall_timing_scope"] = (
            "current condition from work allocation to merge; shared model loading excluded; "
            "cached samples are not rerun"
        )
        write_json(directory / "summary.json", summary)
        print(json.dumps({"condition": condition, **summary["metrics"]}, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--max-new-tokens", type=positive_int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    with evaluation_lock(Path(args.output_dir).expanduser().resolve()):
        _run(args)


if __name__ == "__main__":
    main()
