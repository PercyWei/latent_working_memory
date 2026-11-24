"""在现有 checkpoint 的 train/dev 正文上记录边界概率，不计算 QA 或更新参数。"""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import torch
from transformers import AutoTokenizer, set_seed

from latent_working_memory.runtime import initialize_device
from ..cli import positive_int
from ..config import config_from_dict
from ..data import load_documents, tokenization_identity
from ..io import write_json
from ..modeling.compressor import MemoryModel
from ..rollout import execute_batch
from ..training.checkpoint import checkpoint_file, read_checkpoint
from .records import evaluation_lock


def _now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S Asia/Shanghai")


def _inputs(checkpoint_path, batch_size):
    checkpoint_path = checkpoint_file(checkpoint_path)
    state = read_checkpoint(checkpoint_path)
    config = config_from_dict(state["config"])
    if not config.uses_boundary_policy or config.training.objective.type != "qa":
        raise ValueError("boundary diagnostics require a QA boundary-policy checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenization_identity(tokenizer) != state["binding"]["tokenization"]:
        raise ValueError("diagnostic tokenizer differs from the checkpoint binding")
    documents, statistics = {}, {}
    for split in ("train", "dev"):
        selected, metadata = load_documents(config, tokenizer, split)
        if metadata["identity"] != state["binding"][split]["identity"]:
            raise ValueError(f"{split} data selection differs from the checkpoint binding")
        documents[split] = list(selected.values())
        statistics[split] = metadata
    step = state["cursor"]["step"]
    transition = config.training.objective.policy_transition
    temperature = (1.0 if transition is None
                   else transition.temperature(step, config.training.loop.max_steps))
    protocols = {
        "train": {"mode": "sampled", "temperature": temperature,
                  "paths_per_document": config.training.objective.group_size,
                  "recent_size": config.recent_size(training=True)},
        "dev": {"mode": "greedy", "temperature": 1.0,
                "paths_per_document": 1, "recent_size": config.recent_size()},
    }
    plan = {
        "checkpoint": str(checkpoint_path), "checkpoint_step": step,
        "run_id": checkpoint_path.parents[3].name,
        "training_directory": str(checkpoint_path.parents[2]),
        "model_name_or_path": config.model_name_or_path,
        "model_structure": asdict(config.model.structure),
        "compute": asdict(config.compute),
        "data": {split: statistics[split]["identity"] for split in documents},
        "data_selection": {split: {
            key: statistics[split][key] for key in (
                "file", "minimum_body_tokens", "maximum_body_tokens", "max_samples",
                "selection_seed",
            )
        } for split in documents},
        "split_protocols": protocols,
        "pending_stride": config.model.structure.pending_stride,
        "pending_size": config.model.structure.pending_size,
        "batch_size": batch_size, "seed": config.seed,
        "sampling_seed": "blake2b(seed:boundary-diagnostic:split:document_id:path_id), 8 bytes",
        "trajectory_scope": "New checkpoint diagnostic trajectories; not historical training rollouts.",
        "event_scope": "Only autonomous decision positions; excludes forced wait and EOF.",
        "pending_definition": "Pending before the arriving unit, which is the text committed by this action.",
        "probability_definition": "sigmoid(raw_logit / temperature), retaining the rollout computation dtype",
        "temperature_reference": "Checkpoint optimizer step; dev always uses T=1 and greedy actions.",
        "qa_scoring": False,
    }
    return state, config, documents, plan


def _uniforms(document, path_id, split, plan):
    seed_text = f"{plan['seed']}:boundary-diagnostic:{split}:{document.id}:{path_id}"
    seed = int.from_bytes(hashlib.blake2b(seed_text.encode(), digest_size=8).digest(), "big")
    generator = torch.Generator().manual_seed(seed)
    prefix = max(0, len(document.input_ids) - plan["split_protocols"][split]["recent_size"])
    count = max(0, (prefix - 1) // plan["pending_stride"])
    return torch.rand(count, generator=generator).tolist()


def _document_record(document, paths, split, plan):
    protocol = plan["split_protocols"][split]
    temperature = protocol["temperature"]
    stride = plan["pending_stride"]
    trajectories, decisions = [], []
    for path_id, path in enumerate(paths):
        prefix_length = len(document.input_ids) - len(path.recent_ids)
        unit_lengths = [min(stride, prefix_length - offset)
                        for offset in range(0, prefix_length, stride)]
        pending = unit_lengths[0] if unit_lengths else 0
        read_tokens = pending
        probabilities = (path.raw_logits / temperature).sigmoid().float().cpu().tolist()
        probabilities_t1 = path.raw_logits.sigmoid().float().cpu().tolist()
        logits = path.raw_logits.float().cpu().tolist()
        path_decisions = 0
        for index, (unit, action, allowed, logit, probability, probability_t1) in enumerate(zip(
            unit_lengths[1:], path.actions, path.decision_mask, logits, probabilities,
            probabilities_t1, strict=True,
        )):
            read_tokens += unit
            if allowed:
                decisions.append({
                    "run_id": plan["run_id"], "checkpoint_step": plan["checkpoint_step"],
                    "split": split, "document_id": document.id, "path_id": path_id,
                    "decision_index": index, "context_tokens": len(document.input_ids),
                    "read_tokens": read_tokens, "pending_tokens_before": pending,
                    "raw_logit": logit, "temperature": temperature,
                    "commit_probability": probability,
                    "commit_probability_t1": probability_t1, "action": action,
                })
                path_decisions += 1
            pending = unit if action else pending + unit
        trajectories.append({
            "path_id": path_id, "decisions": path_decisions, "commits": sum(path.actions),
            "logit_dtype": str(path.raw_logits.dtype),
            "memory_blocks": len(path.blocks), "block_lengths": list(path.block_lengths),
            "pending_tokens": len(path.pending_ids), "recent_tokens": len(path.recent_ids),
        })
    return {
        "document_id": document.id, "context_tokens": len(document.input_ids),
        "split": split, "trajectories": trajectories, "decisions": decisions,
    }


def _finalize_split(output_dir, split, documents, plan):
    folder = output_dir / split
    destination = folder / "decisions.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    summary = {"documents": len(documents), "paths": 0, "decisions": 0, "commits": 0,
               "probability_sum": 0.0, "probability_t1_sum": 0.0,
               "probability_ge_half": 0, "probability_t1_ge_half": 0}
    expected_paths = plan["split_protocols"][split]["paths_per_document"]
    with temporary.open("w", encoding="utf-8") as stream:
        for document in documents:
            record = json.loads((folder / "documents" / f"{document.id}.json").read_text())
            if (record["document_id"] != document.id or record["split"] != split
                or len(record["trajectories"]) != expected_paths):
                raise ValueError("completed document differs from the diagnostic request")
            summary["paths"] += len(record["trajectories"])
            for trajectory in record["trajectories"]:
                events = [row for row in record["decisions"]
                          if row["path_id"] == trajectory["path_id"]]
                if (len(events) != trajectory["decisions"]
                    or sum(row["action"] for row in events) != trajectory["commits"]):
                    raise ValueError("document decision records do not match its trajectories")
            for row in record["decisions"]:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                summary["decisions"] += 1
                summary["commits"] += row["action"]
                summary["probability_sum"] += row["commit_probability"]
                summary["probability_t1_sum"] += row["commit_probability_t1"]
                summary["probability_ge_half"] += row["commit_probability"] >= 0.5
                summary["probability_t1_ge_half"] += row["commit_probability_t1"] >= 0.5
    temporary.replace(destination)
    decisions = summary["decisions"]
    summary["mean_commit_probability"] = summary.pop("probability_sum") / decisions if decisions else None
    summary["mean_commit_probability_t1"] = summary.pop("probability_t1_sum") / decisions if decisions else None
    summary["commit_rate"] = summary["commits"] / decisions if decisions else None
    return summary


def diagnose(checkpoint, output_dir, batch_size=1):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("boundary diagnostic workers use one GPU without torchrun")
    output_dir = Path(output_dir).expanduser().resolve()
    with evaluation_lock(output_dir):
        state, config, documents, plan = _inputs(checkpoint, batch_size)
        plan_path = output_dir / "plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("existing diagnostic plan differs from this request")
        else:
            write_json(plan_path, plan)
        completed = {}
        for split, selected in documents.items():
            folder = output_dir / split / "documents"
            folder.mkdir(parents=True, exist_ok=True)
            completed[split] = {document.id for document in selected
                                if (folder / f"{document.id}.json").exists()}
        progress = {
            "status": "running", "started_at": _now(), "updated_at": _now(),
            "checkpoint": plan["checkpoint"], "checkpoint_step": plan["checkpoint_step"],
            "run_id": plan["run_id"], "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "batch_size": batch_size,
            "completed_documents": {split: len(ids) for split, ids in completed.items()},
            "total_documents": {split: len(rows) for split, rows in documents.items()},
        }
        write_json(output_dir / "progress.json", progress)
        started = time.monotonic()
        try:
            if any(len(completed[split]) < len(selected) for split, selected in documents.items()):
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is required for boundary diagnostics")
                device = initialize_device("cuda")
                set_seed(config.seed)
                model = MemoryModel(
                    config.model_name_or_path, asdict(config.model), asdict(config.compute), device,
                    with_value_head=config.training.objective.strategy == "ppo",
                )
                model.load_trainable_state_dict(state["model"])
                model.requires_grad_(False)
                model.eval()
                del state
                for split, selected in documents.items():
                    protocol = plan["split_protocols"][split]
                    remaining = [document for document in selected
                                 if document.id not in completed[split]]
                    for start in range(0, len(remaining), batch_size):
                        batch = remaining[start:start + batch_size]
                        progress.update(split=split, current_document=batch[0].id, updated_at=_now())
                        write_json(output_dir / "progress.json", progress)
                        paths = [[] for _ in batch]
                        for path_id in range(protocol["paths_per_document"]):
                            uniforms = ([_uniforms(document, path_id, split, plan) for document in batch]
                                        if protocol["mode"] == "sampled" else None)
                            with torch.no_grad(), model.forward_context():
                                results = execute_batch(
                                    model, [document.input_ids for document in batch],
                                    plan["pending_stride"], uniforms=uniforms,
                                    temperature=protocol["temperature"],
                                    pending_size=plan["pending_size"],
                                    recent_size=protocol["recent_size"], collect_logits=True,
                                )
                            for index, result in enumerate(results):
                                paths[index].append(result)
                        for document, document_paths in zip(batch, paths, strict=True):
                            record = _document_record(document, document_paths, split, plan)
                            write_json(output_dir / split / "documents" / f"{document.id}.json", record)
                            completed[split].add(document.id)
                            print(f"{plan['run_id']} step={plan['checkpoint_step']} {split} "
                                  f"{len(completed[split])}/{len(selected)} "
                                  f"document={document.id} decisions={len(record['decisions'])}", flush=True)
                        del paths, results
                        progress["completed_documents"][split] = len(completed[split])
                        progress["updated_at"] = _now()
                        write_json(output_dir / "progress.json", progress)
                del model
            split_summaries = {
                split: _finalize_split(output_dir, split, selected, plan)
                for split, selected in documents.items()
            }
            summary = {
                "status": "complete", "checkpoint": plan["checkpoint"],
                "checkpoint_step": plan["checkpoint_step"], "run_id": plan["run_id"],
                "started_at": progress["started_at"], "finished_at": _now(),
                "seconds_this_invocation": time.monotonic() - started,
                "split_summaries": split_summaries,
            }
            write_json(output_dir / "summary.json", summary)
            progress.update(status="complete", updated_at=_now())
            write_json(output_dir / "progress.json", progress)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            return summary
        except Exception as error:
            progress.update(status="failed", error=f"{type(error).__name__}: {error}",
                            updated_at=_now())
            write_json(output_dir / "progress.json", progress)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    args = parser.parse_args(argv)
    diagnose(args.checkpoint, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
