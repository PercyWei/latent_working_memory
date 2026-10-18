"""在最终记忆上统一评估 FactQA；门控题不计入质量指标。"""

import argparse
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from transformers import set_seed

from latent_working_memory.data_preparation.personamem.audit import scores
from latent_working_memory.v3.config import (
    DYNAMIC_METHODS,
    EvaluationConfig,
    ExperimentConfig,
    load_common_config,
)
from latent_working_memory.v3.data import load_factqa
from latent_working_memory.v3.engine import initialize_device
from latent_working_memory.v3.evaluation_records import EvaluationHistory, evaluation_log_path
from latent_working_memory.v3.objective import TokenMemoryTask
from latent_working_memory.v3.qa_selection import filter_qa_budget, select_step_qas
from latent_working_memory.v3.runtime import (
    dataset_identity,
    load_codec,
    read_checkpoint,
    select_examples,
)
from latent_working_memory.v3.tracking import (
    configure_evaluation_layout,
    evaluation_media,
    evaluation_panel_types,
    evaluation_training_run,
    evaluation_tracking_run,
    experiment_directory,
    load_experiment_stages,
    tracking_method,
    verify_evaluation_publication,
)
from latent_working_memory.v3.training_status import read_training_status
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
            selection = select_step_qas(trajectory, step, task.cfg, task.seed, epoch=0)
            record = {
                "trajectory_id": trajectory.trajectory_id,
                "step": step,
                "qas": [asdict(trajectory.qas[qid]) for qid in selection.gate_qa_ids],
            }
            gate_reads += len(selection.gate_qa_ids)
            gate_digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True).encode())
    return task_digest.hexdigest(), {
        "seed": task.seed,
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
        experiment = read_training_status(directory)["experiment"]
        stage = load_experiment_stages(directory, experiment)[config.objective.stage]
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


def _append_evaluation(summary, training_dir, output_dir, step, history):
    """发布当前结果，并将此前已发布的缓存结果同步到当前展示布局。"""
    failures = _append_evaluations([(summary, output_dir, step, history)], training_dir)
    if failures:
        raise failures[0][1]


class _PublicationLog:
    """父进程的发布摘要同时写入终端和对应 checkpoint 的评估日志。"""

    def __init__(self, console, log):
        self.console, self.log = console, log

    @property
    def encoding(self):
        return self.console.encoding

    def isatty(self):
        return False

    def write(self, text):
        self.log.write(text)
        return self.console.write(text)

    def flush(self):
        self.log.flush()
        self.console.flush()


def _append_evaluations(evaluations, training_dir, log_publications=False):
    """将选中结果与已发布历史合成一份报告，由父进程串行发布。"""
    history = evaluations[0][3]
    failures = []
    with history.publication_lock():
        summaries = {
            saved["metadata"]["checkpoint"]: (
                saved,
                directory,
                saved["metadata"]["global_step"],
            )
            for directory, saved in history.published_summaries()
        }
        for summary, output_dir, step, _ in evaluations:
            summaries[summary["metadata"]["checkpoint"]] = (
                summary, output_dir, step
            )
        ordered = sorted(summaries.values(), key=lambda item: item[2])
        try:
            history.validate_summaries([summary for summary, _, _ in ordered])
            with ExitStack() as stack:
                if log_publications:
                    for _, directory, _ in ordered:
                        log_path = evaluation_log_path(directory)
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log = stack.enter_context(
                            log_path.open("a", encoding="utf-8")
                        )
                        stack.enter_context(redirect_stdout(_PublicationLog(sys.stdout, log)))
                        stack.enter_context(redirect_stderr(_PublicationLog(sys.stderr, log)))
                _publish_evaluations(ordered, training_dir, history)
        except Exception as error:
            for saved, _, _ in ordered:
                checkpoint = Path(saved["metadata"]["checkpoint"])
                failures.append((checkpoint, error))
            print(
                f"[evaluation] SwanLab report failed checkpoints={len(ordered)} "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
    return failures


def validate_evaluation_publication(checkpoints):
    """在批量计算前检查所有 checkpoint 的原训练身份与云端完成状态。"""
    directories = set()
    for path in checkpoints:
        checkpoint = read_checkpoint(path)
        raw = checkpoint["run"]["config"]
        config = ExperimentConfig.from_dict(raw)
        directories.add(_training_run_directory(config, checkpoint["run"]))
        del checkpoint
    if len(directories) != 1:
        raise ValueError("batch publication requires checkpoints from one original training run")
    training_dir = directories.pop()
    evaluation_training_run(training_dir, swanlab_api_key())
    return training_dir


def publish_cached_evaluations(checkpoints, output_dirs, evaluation_config, training_dir):
    """读取已完成结果统一发布；不加载模型、不初始化 CUDA、不重新计算 QA。"""
    evaluations = []
    failures = []
    for checkpoint, output in zip(checkpoints, output_dirs, strict=True):
        try:
            history = EvaluationHistory(output, asdict(evaluation_config))
            summary = history.cached_summary(checkpoint)
            if summary is None:
                raise ValueError(f"SwanLab publication requires completed local results: {output}")
            evaluations.append((summary, Path(output), summary["metadata"]["global_step"], history))
        except Exception as error:
            failures.append((Path(checkpoint), error))
            print(
                f"[evaluation] SwanLab cache rejected checkpoint={checkpoint} "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
    if evaluations:
        failures.extend(_append_evaluations(evaluations, training_dir, log_publications=True))
    return failures


def _publish_evaluations(evaluations, training_dir, history):
    """发布完整 checkpoint 合图，匹配的缓存报告只核对云端布局。"""
    identity = json.loads((training_dir / "swanlab.json").read_text(encoding="utf-8"))
    summaries = [summary for summary, _, _ in evaluations]
    media_types = evaluation_panel_types(summaries)
    signature = hashlib.sha256(
        json.dumps(summaries, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    publication = history.report_publication()
    if publication is not None and (
        publication["run_id"] != identity["id"]
        or publication["project"] != identity["project"]
        or publication["group"] != identity["group"]
    ):
        raise ValueError("evaluation report publication differs from the original training run")
    already_published = (
        publication is not None
        and publication["signature"] == signature
        and publication["panels"] == list(media_types)
    )
    print(
        f"[evaluation] SwanLab report {'layout check' if already_published else 'publication'} started "
        f"method={summaries[0]['method']} checkpoints={len(summaries)} "
        f"global_steps={[summary['metadata']['global_step'] for summary in summaries]}",
        flush=True,
    )
    api_key = swanlab_api_key()
    try:
        if already_published:
            evaluation_training_run(training_dir, api_key)
            publication_step = publication["publication_step"]
        else:
            media = evaluation_media(summaries)
            with evaluation_tracking_run(training_dir, api_key) as tracking:
                # 发布编号只接续评估媒体历史，训练 step 不参与累计报告编号。
                resumed_step = max(
                    (
                        metric.min_step
                        for key, metric in tracking._ctx.core._metrics._metrics.items()
                        if key in media
                    ),
                    default=0,
                )
                publication_step = max(
                    0, resumed_step,
                    publication["publication_step"] if publication is not None else -1,
                ) + 1
                tracking.log(media, step=publication_step)
            verify_evaluation_publication(
                training_dir, media, publication_step, api_key
            )
        view = configure_evaluation_layout(training_dir, media_types, api_key)
    except BaseException as error:
        print(
            f"[evaluation] SwanLab report failed; cached results retained "
            f"path={history.path} "
            f"error={type(error).__name__}: {error}",
            flush=True,
        )
        raise
    publication = {
        "training_dir": str(training_dir.resolve()),
        "run_id": identity["id"],
        "project": identity["project"],
        "group": identity["group"],
        "signature": signature,
        "summaries": [
            str(Path(output_dir).resolve() / "summary.json")
            for _, output_dir, _ in evaluations
        ],
        "publication_step": publication_step,
        "view": view,
        "panels": list(media_types),
    }
    history.record_report_publication(publication)
    print(
        f"[evaluation] SwanLab report {'layout ready' if already_published else 'publication finished'} "
        f"run_id={identity['id']} checkpoints={len(summaries)} publication_step={publication_step} "
        f"view={view['url']}",
        flush=True,
    )
    return view


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
    temporary = directory / "details.jsonl.tmp"
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
            evaluation_started = time.perf_counter()
            for sample_index, trajectory in enumerate(trajectories, start=1):
                sample_started = time.perf_counter()
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
                if split == "test":
                    finished = time.perf_counter()
                    progress_elapsed = finished - evaluation_started
                    print(
                        f"[evaluation] {task.cfg.method}/{task.cfg.stage} "
                        f"| samples={sample_index}/{len(trajectories)} "
                        f"| qa={len(questions)} slots={capacity['final_slots']} "
                        f"| sample={finished - sample_started:.2f}s "
                        f"| elapsed={progress_elapsed:.2f}s "
                        f"| avg={progress_elapsed / sample_index:.2f}s/sample",
                        flush=True,
                    )
    finally:
        task.train(was_training)
    temporary.replace(directory / "details.jsonl")
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
    if split == "test" and "evaluation_config" in summary["metadata"]:
        EvaluationHistory(directory, summary["metadata"]["evaluation_config"]).save_summary(summary)
    else:
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
    output = Path(args.output_dir).resolve()
    common_path = output.parents[4] / "plan" / output.parents[2].name / "config.json"
    seed = load_common_config(common_path)["seed"]
    snapshot = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if snapshot["seed"] != seed:
        raise ValueError("evaluation snapshot seed differs from the common seed")
    evaluation_config = EvaluationConfig(
        **{name: value for name, value in snapshot.items() if name != "seed"}, seed=seed
    )
    checkpoint_path = Path(args.checkpoint).resolve()
    print(f"[evaluation] loading checkpoint={checkpoint_path}", flush=True)
    checkpoint = read_checkpoint(checkpoint_path)
    raw = checkpoint["run"]["config"]
    config = ExperimentConfig.from_dict(raw)
    training_dir = (
        _training_run_directory(config, checkpoint["run"]) if args.log_to_swanlab else None
    )
    stage_step = checkpoint["cursor"]["step"]
    global_step = checkpoint["run"]["step_offset"] + stage_step
    print(
        f"[evaluation] checkpoint loaded method={config.objective.method} "
        f"stage={config.objective.stage} stage_step={stage_step} global_step={global_step}",
        flush=True,
    )
    history = EvaluationHistory(Path(args.output_dir), asdict(evaluation_config))
    summary = history.cached_summary(checkpoint_path)
    if summary is None:
        print(f"[evaluation] loading model device={args.device}", flush=True)
        device = initialize_device(args.device)
        set_seed(evaluation_config.seed)
        codec, tokenizer = load_codec(config.model, device, config.objective)
        codec.load_trainable_state_dict(checkpoint["trainable"])
        task = TokenMemoryTask(codec, tokenizer, config.objective, evaluation_config.seed)
        print("[evaluation] model loaded", flush=True)
        print(
            f"[evaluation] loading test data={evaluation_config.dataset_dir}", flush=True
        )
        trajectories = load_factqa(evaluation_config.dataset_dir, tokenizer)["test"]
        _check_qa_sources(trajectories, checkpoint["run"]["pretraining_sources"])
        source_data = dataset_identity(trajectories)
        total_trajectories = len(trajectories)
        filtered, budget_statistics = filter_qa_budget(
            {"test": trajectories}, config.objective, include_train=False
        )
        trajectories = filtered["test"]
        retained_trajectories = len(trajectories)
        trajectories = select_examples(
            trajectories,
            evaluation_config.max_test_samples,
            evaluation_config.seed,
            "test",
            pretraining=False,
        )
        print(
            f"[evaluation] data ready trajectories={len(trajectories)} "
            f"retained={retained_trajectories} total={total_trajectories}",
            flush=True,
        )
        summary = evaluate(
            task,
            trajectories,
            args.output_dir,
            "test",
            evaluation_config.max_new_tokens,
            metadata={
                "checkpoint": str(checkpoint_path),
                "dataset_dir": str(Path(evaluation_config.dataset_dir).resolve()),
                "config": config.to_dict(),
                "evaluation_config": asdict(evaluation_config),
                "source_data": source_data,
                "device": args.device,
                "autocast_dtype": "bfloat16" if device.type == "cuda" else None,
                "pretraining": checkpoint["run"]["pretraining"],
                "global_step": global_step,
                "stage_step": stage_step,
                "selection": {
                    "total": total_trajectories,
                    "selected": len(trajectories),
                    "retained": retained_trajectories,
                    "qa_budget": budget_statistics["test"],
                    "limit": evaluation_config.max_test_samples,
                    "seed": evaluation_config.seed,
                },
            },
        )
    else:
        print(
            f"[evaluation] reusing cached summary "
            f"trajectories={summary['trajectories']} path={Path(args.output_dir) / 'summary.json'}",
            flush=True,
        )
    if training_dir is not None:
        _append_evaluation(
            summary,
            training_dir,
            args.output_dir,
            global_step,
            history,
        )
    quality = summary["quality"]["all"]
    quality_text = " ".join(
        f"{name}={quality[name]:.4f}" for name in quality if name != "questions"
    )
    print(
        f"[evaluation] method={summary['method']} split=test "
        f"trajectories={summary['trajectories']} {quality_text} "
        f"summary={Path(args.output_dir) / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
