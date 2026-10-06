"""在最终记忆上统一评估 FactQA；门控题不计入质量指标。"""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from latent_working_memory.data_preparation.personamem.audit import scores
from latent_working_memory.v1.tracking import swanlab_training_run
from latent_working_memory.v3.config import (
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainingConfig,
)
from latent_working_memory.v3.data import load_factqa
from latent_working_memory.v3.engine import initialize_device
from latent_working_memory.v3.objective import TokenMemoryTask
from latent_working_memory.v3.runtime import (
    dataset_identity,
    load_codec,
    read_checkpoint,
    select_examples,
)
from latent_working_memory.v3.tracking import evaluation_media


def _quality(rows):
    def average(selected):
        return {
            "questions": len(selected),
            **{
                name: sum(row[name] for row in selected) / len(selected) if selected else None
                for name in ("nll", "em", "f1")
            },
        }

    return {
        "all": average(rows),
        "new": average([row for row in rows if row["age"] == 0]),
        "old": average([row for row in rows if row["age"] > 0]),
        "by_age": {
            str(age): average([row for row in rows if row["age"] == age])
            for age in sorted({row["age"] for row in rows})
        },
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
    directory = Path(config.training.output_dir)
    if not (directory / "swanlab.json").is_file():
        raise ValueError("--log-to-swanlab requires the original training SwanLab identity")
    if json.loads((directory / "run.json").read_text(encoding="utf-8")) != run:
        raise ValueError("checkpoint run differs from the original training directory")
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
    rows, questions = [], 0
    with (Path(output_dir) / "trajectories.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows.append(row)
            questions += len(row["questions"])
            if questions >= 8:
                break
    values = evaluation_media(summary, rows)
    # 既有 helper 保留云端训练 config/name，检查训练已结束并使用 resume="must"。
    with swanlab_training_run(training_dir) as tracking:
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
    if not trajectories or any(trajectory.split != split for trajectory in trajectories):
        raise ValueError("evaluation requires nonempty trajectories from the requested split")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "trajectories.jsonl.tmp"
    all_questions, capacities, costs = [], [], []
    signature = dataset_identity(trajectories)["fingerprint"]
    was_training = task.training
    task.eval()
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for trajectory in trajectories:
                qa_ids = [qa.qa_id for qa in trajectory.qas.values() if qa.role == "evaluation"]
                if not qa_ids:
                    raise ValueError(
                        f"trajectory {trajectory.trajectory_id} has no evaluation questions"
                    )
                segment_positions = {
                    segment.segment_id: index for index, segment in enumerate(trajectory.segments)
                }
                _synchronize(task)
                started = time.perf_counter()
                blocks, events = task.build_memory(trajectory, epoch=0, force_policy=True)
                _synchronize(task)
                build_seconds = time.perf_counter() - started
                memory = torch.cat(blocks)
                started = time.perf_counter()
                losses = task.qa_losses(blocks, trajectory, qa_ids).cpu().tolist()
                _synchronize(task)
                nll_seconds = time.perf_counter() - started
                questions = []
                started = time.perf_counter()
                for qa_id, nll in zip(qa_ids, losses, strict=True):
                    qa = trajectory.qas[qa_id]
                    generated = task.codec.generate(
                        memory,
                        task.prompt_ids(qa.question),
                        max_new_tokens,
                        eos_token_id=task.tokenizer.eos_token_id,
                        pad_token_id=task.tokenizer.pad_token_id,
                    )
                    generated_ids = generated.cpu().tolist()
                    prediction = task.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    questions.append(
                        {
                            "qa_id": qa_id,
                            "segment_id": qa.segment_id,
                            "age": len(trajectory.segments) - 1 - segment_positions[qa.segment_id],
                            "question": qa.question,
                            "answer": qa.answer,
                            "prediction": prediction,
                            "generated_ids": generated_ids,
                            "answer_tokens": len(qa.answer_ids),
                            "nll": nll,
                            **scores(prediction, qa.answer),
                        }
                    )
                _synchronize(task)
                generation_seconds = time.perf_counter() - started
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
                    generation_qa_reads=len(questions),
                    generated_tokens=sum(len(qa["generated_ids"]) for qa in questions),
                )
                record = {
                    "trajectory_id": trajectory.trajectory_id,
                    "document_id": trajectory.document_id,
                    "segments": len(trajectory.segments),
                    "source_tokens": len(trajectory.full_input_ids),
                    "capacity": capacity,
                    "costs": cost,
                    "events": events,
                    "quality": _quality(questions),
                    "questions": questions,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                capacities.append(capacity)
                costs.append(cost)
                all_questions.extend(
                    {name: qa[name] for name in ("age", "nll", "em", "f1")} for qa in questions
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
        "max_new_tokens": max_new_tokens,
        "quality": _quality(all_questions),
        "capacity": {
            name: sum(row[name] for row in capacities) / len(capacities) for name in capacities[0]
        },
        "costs": {name: sum(row[name] for row in costs) for name in costs[0]},
        "aggregation": {
            "quality": "question_mean",
            "capacity": "trajectory_mean",
            "mean_slots": "mean over observed memory states; single compression has one state",
            "costs": "sum",
            "age": "number of source segments after the evidence segment",
        },
        "objective": asdict(task.cfg),
        "metadata": {} if metadata is None else metadata,
    }
    _save_json(directory / "summary.json", summary)
    return summary


def _positive_integer(value):
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return integer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=_positive_integer, default=64)
    parser.add_argument(
        "--max-trajectories",
        type=_positive_integer,
        help="使用训练 seed 确定性选择完整评估轨迹；省略时评估完整 split",
    )
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
    codec, tokenizer = load_codec(model_config, device)
    codec.load_trainable_state_dict(checkpoint["trainable"])
    task = TokenMemoryTask(codec, tokenizer, config.objective)
    trajectories = load_factqa(args.dataset_dir, tokenizer)[args.split]
    _check_qa_sources(trajectories, checkpoint["run"]["pretraining_sources"])
    if config.objective.stage in {"qa", "warmup", "policy"}:
        if dataset_identity(trajectories) != checkpoint["run"]["source_data"][args.split]:
            raise ValueError("evaluation FactQA split differs from the checkpoint's original split")
    total_trajectories = len(trajectories)
    trajectories = select_examples(
        trajectories, args.max_trajectories, config.training.seed, args.split, pretraining=False
    )
    summary = evaluate(
        task,
        trajectories,
        args.output_dir,
        args.split,
        args.max_new_tokens,
        metadata={
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "dataset_dir": str(Path(args.dataset_dir).resolve()),
            "config": config.to_dict(),
            "resolved_model_revision": revision,
            "device": args.device,
            "selection": {
                "total": total_trajectories,
                "selected": len(trajectories),
                "limit": args.max_trajectories,
                "seed": config.training.seed,
            },
        },
    )
    if training_dir is not None:
        _append_evaluation(summary, training_dir, args.output_dir, checkpoint["cursor"]["step"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
