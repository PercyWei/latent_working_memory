"""在最终记忆上统一评估 FactQA；门控题不计入质量指标。"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from latent_working_memory.data_preparation.personamem.audit import scores
from latent_working_memory.v1.tracking import swanlab_training_run
from latent_working_memory.v3.config import (
    DYNAMIC_METHODS,
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainingConfig,
    load_evaluation,
)
from latent_working_memory.v3.data import load_factqa
from latent_working_memory.v3.engine import initialize_device
from latent_working_memory.v3.objective import TokenMemoryTask
from latent_working_memory.v3.qa_selection import filter_qa_budget, select_step_qas
from latent_working_memory.v3.runtime import (
    dataset_identity,
    load_codec,
    read_checkpoint,
    select_examples,
)
from latent_working_memory.v3.tracking import (
    evaluation_media,
    experiment_directory,
    tracking_method,
)
from latent_working_memory.tracking_credentials import swanlab_api_key


def _quality(rows, metrics):
    def average(selected):
        return {
            "questions": len(selected),
            **{
                name: sum(row[name] for row in selected) / len(selected) if selected else None
                for name in metrics
            },
        }

    return {
        "all": average(rows),
        "by_introduction": {
            str(step): average([row for row in rows if row["introduced_at"] == step])
            for step in sorted({row["introduced_at"] for row in rows})
        },
    }


def _nll_retention(rows):
    def trajectory_average(values, metric):
        trajectories = {}
        for value, identity in values:
            trajectories.setdefault(identity, []).append(value)
        return {
            "questions": len(values),
            "trajectories": len(trajectories),
            metric: sum(sum(group) / len(group) for group in trajectories.values())
            / len(trajectories),
        }

    cells, ages = {}, {}
    for row in rows:
        first = row["nll_by_step"][0]["nll"]
        for observation in row["nll_by_step"]:
            key = (row["introduced_at"], observation["read_step"])
            cells.setdefault(key, []).append((observation["nll"], row["trajectory_id"]))
            ages.setdefault(observation["age"], []).append(
                (observation["nll"] - first, row["trajectory_id"])
            )
    return {
        "cells": [
            {
                "introduced_at": introduced,
                "read_step": read,
                **trajectory_average(values, "nll"),
            }
            for (introduced, read), values in sorted(cells.items())
        ],
        "by_age": {
            str(age): trajectory_average(values, "delta_nll")
            for age, values in sorted(ages.items())
        },
    }


def _qa_protocol(task, trajectories):
    task_digest, gate_digest = hashlib.blake2b(digest_size=32), hashlib.blake2b(digest_size=32)
    gate_reads = 0
    for trajectory in sorted(trajectories, key=lambda row: row.trajectory_id):
        task_record = asdict(trajectory)
        task_record.pop("gate_qa_ids")
        task_record["qas"] = {
            qid: task_record["qas"][qid] for ids in trajectory.task_qa_ids for qid in ids
        }
        task_digest.update(json.dumps(task_record, ensure_ascii=False, sort_keys=True).encode())
        for step in range(len(trajectory.segments)):
            selection = select_step_qas(trajectory, step, task.cfg, epoch=0, seed=task.qa_seed)
            record = {
                "trajectory_id": trajectory.trajectory_id,
                "step": step,
                "qas": [asdict(trajectory.qas[qid]) for qid in selection.gate_qa_ids],
            }
            gate_reads += len(selection.gate_qa_ids)
            gate_digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True).encode())
    return task_digest.hexdigest(), {
        "seed": task.qa_seed,
        "gate_qa_count": task.cfg.gate_qa_count,
        "gate_qa_reads": gate_reads,
        "gate_signature": gate_digest.hexdigest(),
    }


def _synchronize(task):
    device = task.codec.memory_embeddings.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _check_qa_sources(trajectories, pretraining_sources):
    documents = {trajectory.document_id for trajectory in trajectories}
    clusters = {trajectory.dedup_cluster for trajectory in trajectories}
    if documents.intersection(pretraining_sources["document_ids"]) or clusters.intersection(
        pretraining_sources["dedup_clusters"]
    ):
        raise ValueError("evaluation QA sources overlap pretraining document IDs or dedup clusters")


def _training_run_directory(config, run):
    if config.training.swanlab_project is None:
        raise ValueError("--log-to-swanlab requires a checkpoint configured for SwanLab")
    stage_directory = Path(config.training.output_dir)
    directory = experiment_directory(config.training)
    if not (directory / "swanlab.json").is_file():
        raise ValueError("--log-to-swanlab requires the original training SwanLab identity")
    if json.loads((stage_directory / "run.json").read_text(encoding="utf-8")) != run:
        raise ValueError("checkpoint run differs from the original training directory")
    if config.training.experiment_dir is not None:
        experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
        stage = experiment["stages"][config.objective.stage]
        if (
            experiment["experiment_id"] != config.training.experiment_id
            or experiment["method"] != tracking_method(config, experiment)
            or any(stage[name] != run[name] for name in ("config", "step_offset", "pretraining"))
        ):
            raise ValueError("checkpoint stage differs from the saved method experiment")
    identity = json.loads((directory / "swanlab.json").read_text(encoding="utf-8"))
    expected = {
        "project": config.training.swanlab_project,
        "group": config.training.group,
        "job_type": "train",
        "mode": "online",
    }
    if any(identity[name] != value for name, value in expected.items()):
        raise ValueError("SwanLab identity differs from the original online training run")
    return directory


def _append_evaluation(summary, training_dir, output_dir, step):
    values = evaluation_media(summary)
    # 既有 helper 保留云端训练 config/name，检查训练已结束并使用 resume="must"。
    with swanlab_training_run(training_dir, api_key=swanlab_api_key()) as tracking:
        tracking.log(values, step=step)
    identity = json.loads((training_dir / "swanlab.json").read_text(encoding="utf-8"))
    _save_json(
        Path(output_dir) / "swanlab-publication.json",
        {
            "training_dir": str(training_dir.resolve()),
            "run_id": identity["id"],
            "project": identity["project"],
            "group": identity["group"],
            "split": summary["split"],
            "step": step,
            "panels": list(values),
            "quality": summary["quality"],
            "capacity": summary["capacity"],
            "selection": summary["metadata"].get("selection"),
        },
    )


@torch.no_grad()
def evaluate(task, trajectories, output_dir, split, max_new_tokens, metadata=None):
    """质量按题平均，容量按轨迹平均，成本累计实际调用与耗时。"""
    if split not in {"dev", "test"}:
        raise ValueError("QA evaluation requires the dev or test split")
    if task.cfg.method == "dynamic":
        raise ValueError("shared dynamic pretraining has no FactQA memory policy")
    if not trajectories or any(trajectory.split != split for trajectory in trajectories):
        raise ValueError("evaluation requires nonempty trajectories from the requested split")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    dynamic = task.cfg.method in DYNAMIC_METHODS
    generate_answers = split == "test"
    quality_metrics = ("nll", "em", "f1") if generate_answers else ("nll",)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "trajectories.jsonl.tmp"
    all_questions, capacities, costs = [], [], []
    signature, qa_protocol = _qa_protocol(task, trajectories)
    was_training = task.training
    task.eval()
    try:
        precision = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if task.codec.memory_embeddings.device.type == "cuda"
            else nullcontext()
        )
        with precision, temporary.open("w", encoding="utf-8") as stream:
            for trajectory in trajectories:
                qa_ids = [qid for ids in trajectory.task_qa_ids for qid in ids]
                if not qa_ids:
                    raise ValueError(
                        f"trajectory {trajectory.trajectory_id} has no evaluation questions"
                    )
                segment_positions = {
                    segment.segment_id: index for index, segment in enumerate(trajectory.segments)
                }
                nll_by_qa = {qid: [] for qid in qa_ids} if dynamic else None
                final_nll = {}
                events, build_seconds = [], 0.0
                prefix_nll_seconds, prefix_nll_reads = 0.0, 0
                states = (
                    iter(task._states(trajectory, epoch=0, force_policy=True)) if dynamic else None
                )
                read_steps = (
                    range(len(trajectory.segments))
                    if dynamic
                    else (len(trajectory.segments) - 1,)
                )
                for step in read_steps:
                    _synchronize(task)
                    started = time.perf_counter()
                    if dynamic:
                        blocks, event = next(states)
                        events.append(event)
                    else:
                        blocks, events = task.build_memory(
                            trajectory, epoch=0, force_policy=True
                        )
                    _synchronize(task)
                    build_seconds += time.perf_counter() - started
                    current_ids = [qid for ids in trajectory.task_qa_ids[: step + 1] for qid in ids]
                    if not current_ids:
                        continue
                    started = time.perf_counter()
                    losses = task.qa_losses(blocks, trajectory, current_ids).cpu().tolist()
                    _synchronize(task)
                    elapsed = time.perf_counter() - started
                    if step == len(trajectory.segments) - 1:
                        nll_seconds = elapsed
                        final_nll = dict(zip(current_ids, losses, strict=True))
                    else:
                        prefix_nll_seconds += elapsed
                        prefix_nll_reads += len(current_ids)
                    if dynamic:
                        for qid, loss in zip(current_ids, losses, strict=True):
                            nll_by_qa[qid].append(
                                {
                                    "read_step": step + 1,
                                    "age": step - segment_positions[trajectory.qas[qid].segment_id],
                                    "nll": loss,
                                }
                            )
                memory = torch.cat(blocks)
                questions = []
                _synchronize(task)
                started = time.perf_counter()
                for qa_id in qa_ids:
                    qa = trajectory.qas[qa_id]
                    question = {
                        "qa_id": qa_id,
                        "segment_id": qa.segment_id,
                        "introduced_at": segment_positions[qa.segment_id] + 1,
                        "question": qa.question,
                        "answer": qa.answer,
                        "answer_tokens": len(qa.answer_ids),
                        "nll": final_nll[qa_id],
                    }
                    if dynamic:
                        question.update(
                            nll_by_step=nll_by_qa[qa_id],
                            age=len(trajectory.segments) - 1 - segment_positions[qa.segment_id],
                        )
                    if generate_answers:
                        generated = task.codec.generate(
                            memory,
                            task.prompt_ids(qa.question),
                            max_new_tokens,
                            eos_token_id=task.tokenizer.eos_token_id,
                            pad_token_id=task.tokenizer.pad_token_id,
                        )
                        generated_ids = generated.cpu().tolist()
                        prediction = task.tokenizer.decode(generated_ids, skip_special_tokens=True)
                        question.update(
                            prediction=prediction,
                            generated_ids=generated_ids,
                            **scores(prediction, qa.answer),
                        )
                    questions.append(question)
                _synchronize(task)
                generation_seconds = time.perf_counter() - started if generate_answers else 0.0
                capacity = {
                    "final_slots": len(memory),
                    "mean_slots": sum(event["slots"] for event in events) / len(events),
                    "appends": sum(event["action"] == "append" for event in events),
                    "overwrites": sum(event["action"] == "overwrite" for event in events),
                }
                cost = {
                    name: sum(event[name] for event in events)
                    for name in ("write_calls", "write_seconds", "gate_qa_reads", "gate_seconds")
                }
                cost.update(
                    build_seconds=build_seconds,
                    nll_seconds=nll_seconds,
                    generation_seconds=generation_seconds,
                    nll_qa_reads=len(questions),
                    generation_qa_reads=len(questions) if generate_answers else 0,
                    generated_tokens=(
                        sum(len(qa["generated_ids"]) for qa in questions)
                        if generate_answers
                        else 0
                    ),
                )
                if dynamic:
                    cost.update(
                        prefix_nll_seconds=prefix_nll_seconds,
                        prefix_nll_qa_reads=prefix_nll_reads,
                    )
                record = {
                    "trajectory_id": trajectory.trajectory_id,
                    "document_id": trajectory.document_id,
                    "segments": len(trajectory.segments),
                    "source_tokens": len(trajectory.full_input_ids),
                    "capacity": capacity,
                    "costs": cost,
                    "events": events,
                    "quality": _quality(questions, quality_metrics),
                    "questions": questions,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                capacities.append(capacity)
                costs.append(cost)
                all_questions.extend(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        **{
                            name: qa[name]
                            for name in ("introduced_at", *quality_metrics)
                        },
                        **({"nll_by_step": qa["nll_by_step"]} if dynamic else {}),
                    }
                    for qa in questions
                )
    finally:
        task.train(was_training)
    temporary.replace(directory / "trajectories.jsonl")
    summary = {
        "method": task.cfg.method,
        "stage": task.cfg.stage,
        "offline_oracle": task.cfg.method == "information_loss",
        "split": split,
        "dataset_signature": signature,
        "trajectories": len(trajectories),
        "max_new_tokens": max_new_tokens if generate_answers else None,
        "quality": _quality(all_questions, quality_metrics),
        **({"nll_retention": _nll_retention(all_questions)} if dynamic else {}),
        "qa_sampling": qa_protocol,
        "capacity": {
            name: sum(row[name] for row in capacities) / len(capacities) for name in capacities[0]
        },
        "costs": {name: sum(row[name] for row in costs) for name in costs[0]},
        "aggregation": {
            "quality": "question_mean",
            "capacity": "trajectory_mean",
            "mean_slots": "mean over observed memory states; single compression has one state",
            "costs": "sum",
            "introduction": "one-based source segment containing the evidence",
            **(
                {
                    "retention": "trajectory_mean",
                    "age": "number of source segments after the evidence segment",
                    "prefix_costs": "NLL reads before the final memory state; final reads remain separate",
                }
                if dynamic
                else {}
            ),
        },
        "objective": asdict(task.cfg),
        "metadata": {} if metadata is None else metadata,
    }
    _save_json(directory / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--log-to-swanlab", action="store_true", help="追加到原训练 run，保留训练配置"
    )
    args = parser.parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
        dist.is_initialized() and dist.get_world_size() != 1
    ):
        raise ValueError(
            "final QA evaluation supports a single process; do not launch with torchrun"
        )
    evaluation_config = load_evaluation(args.config)
    device = initialize_device(args.device)
    checkpoint = read_checkpoint(args.checkpoint)
    raw = checkpoint["run"]["config"]
    config = ExperimentConfig(
        ModelConfig(**raw["model"]),
        ObjectiveConfig(**raw["objective"]),
        TrainingConfig(**raw["training"]),
    )
    training_dir = (
        _training_run_directory(config, checkpoint["run"]) if args.log_to_swanlab else None
    )
    revision = checkpoint["run"]["resolved_model_revision"]
    model_config = (
        replace(config.model, revision=revision) if revision is not None else config.model
    )
    codec, tokenizer = load_codec(model_config, device, config.objective)
    codec.load_trainable_state_dict(checkpoint["trainable"])
    objective = replace(config.objective, gate_qa_count=evaluation_config.gate_qa_count)
    task = TokenMemoryTask(codec, tokenizer, objective, qa_seed=evaluation_config.seed)
    trajectories = load_factqa(evaluation_config.dataset_dir, tokenizer)[evaluation_config.split]
    _check_qa_sources(trajectories, checkpoint["run"]["pretraining_sources"])
    source_data = dataset_identity(trajectories)
    total_trajectories = len(trajectories)
    filtered, budget_statistics = filter_qa_budget(
        {evaluation_config.split: trajectories}, objective, include_train=False
    )
    trajectories = filtered[evaluation_config.split]
    retained_trajectories = len(trajectories)
    trajectories = select_examples(
        trajectories,
        evaluation_config.max_trajectories,
        evaluation_config.seed,
        evaluation_config.split,
        pretraining=False,
    )
    summary = evaluate(
        task,
        trajectories,
        args.output_dir,
        evaluation_config.split,
        evaluation_config.max_new_tokens,
        metadata={
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "dataset_dir": str(Path(evaluation_config.dataset_dir).resolve()),
            "config": config.to_dict(),
            "evaluation_config": asdict(evaluation_config),
            "source_data": source_data,
            "resolved_model_revision": revision,
            "device": args.device,
            "autocast_dtype": "bfloat16" if device.type == "cuda" else None,
            "pretraining": checkpoint["run"]["pretraining"],
            "global_step": checkpoint["run"]["step_offset"] + checkpoint["cursor"]["step"],
            "selection": {
                "total": total_trajectories,
                "selected": len(trajectories),
                "retained": retained_trajectories,
                "qa_budget": budget_statistics[evaluation_config.split],
                "limit": evaluation_config.max_trajectories,
                "seed": evaluation_config.seed,
            },
        },
    )
    if training_dir is not None:
        _append_evaluation(
            summary,
            training_dir,
            args.output_dir,
            checkpoint["run"]["step_offset"] + checkpoint["cursor"]["step"],
        )
    quality = summary["quality"]["all"]
    quality_text = " ".join(
        f"{name}={quality[name]:.4f}" for name in quality if name != "questions"
    )
    print(
        f"[evaluation] method={summary['method']} split={evaluation_config.split} "
        f"trajectories={summary['trajectories']} {quality_text} "
        f"summary={Path(args.output_dir) / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
