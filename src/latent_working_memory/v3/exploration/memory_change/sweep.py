"""冻结同一 checkpoint，完整重跑阈值策略与固定动作基线，在 dev 上比较质量和容量。"""

import argparse
from contextlib import nullcontext
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import torch

from latent_working_memory.v3.config import parse_optional_positive_integer
from latent_working_memory.v3.evaluate import evaluate
from latent_working_memory.v3.exploration.memory_change.common import load_dev_task, save_json
from latent_working_memory.v3.objective import TokenMemoryTask


class FixedActionTask(TokenMemoryTask):
    """固定动作基线只执行所选写入，不额外构造门控候选。"""

    def __init__(self, task, append):
        super().__init__(task.codec, task.tokenizer, task.cfg, task.seed)
        self.append = append

    def build_memory(self, trajectory, epoch=0, force_policy=True):
        blocks, events = [], []
        for step, segment in enumerate(trajectory.segments):
            event = self._event(step, segment.segment_id)
            if step == 0:
                blocks = self._write_batch([segment.input_ids], [[]], [event], actions=["initial"])
            else:
                candidate = self._write_updates_batch(
                    [segment.input_ids], [blocks], [self.append], [event]
                )[0]
                blocks = blocks + [candidate] if self.append else blocks[:-1] + [candidate]
                event["action"] = "append" if self.append else "overwrite"
            event["slots"] = sum(map(len, blocks))
            events.append(event)
        return blocks, events


def _synchronize(task):
    if task.device.type == "cuda":
        torch.cuda.synchronize(task.device)


def _nll_quality(rows):
    def average(selected):
        return {
            "questions": len(selected),
            "nll": sum(row["nll"] for row in selected) / len(selected) if selected else None,
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


@torch.no_grad()
def evaluate_nll(task, trajectories, directory, metadata):
    """与正式 evaluate 使用相同的最终 task QA、按题平均和容量口径。"""
    all_questions, capacities, costs = [], [], []
    task.eval()
    precision = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if task.device.type == "cuda"
        else nullcontext()
    )
    temporary = directory / "trajectories.jsonl.tmp"
    with precision, temporary.open("w", encoding="utf-8") as stream:
        for trajectory in trajectories:
            qa_ids = [qa.qa_id for qa in trajectory.qas.values() if qa.role == "task"]
            if not qa_ids:
                raise ValueError(
                    f"trajectory {trajectory.trajectory_id} has no task questions"
                )
            positions = {
                segment.segment_id: index for index, segment in enumerate(trajectory.segments)
            }
            _synchronize(task)
            started = time.perf_counter()
            blocks, events = task.build_memory(trajectory, epoch=0, force_policy=True)
            _synchronize(task)
            build_seconds = time.perf_counter() - started
            started = time.perf_counter()
            losses = task.qa_losses(blocks, trajectory, qa_ids).cpu().tolist()
            _synchronize(task)
            nll_seconds = time.perf_counter() - started
            questions = [
                {
                    "qa_id": qa_id,
                    "segment_id": trajectory.qas[qa_id].segment_id,
                    "age": len(trajectory.segments)
                    - 1
                    - positions[trajectory.qas[qa_id].segment_id],
                    "answer_tokens": len(trajectory.qas[qa_id].answer_ids),
                    "nll": nll,
                }
                for qa_id, nll in zip(qa_ids, losses, strict=True)
            ]
            capacity = {
                "final_slots": sum(map(len, blocks)),
                "mean_slots": sum(event["slots"] for event in events) / len(events),
                "appends": sum(event["action"] == "append" for event in events),
                "overwrites": sum(event["action"] == "overwrite" for event in events),
            }
            cost = {
                name: sum(event[name] for event in events)
                for name in ("write_calls", "write_seconds", "gate_qa_reads", "gate_seconds")
            }
            cost.update(
                build_seconds=build_seconds, nll_seconds=nll_seconds, nll_qa_reads=len(qa_ids)
            )
            stream.write(
                json.dumps(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "document_id": trajectory.document_id,
                        "segments": len(trajectory.segments),
                        "source_tokens": len(trajectory.full_input_ids),
                        "capacity": capacity,
                        "costs": cost,
                        "events": events,
                        "quality": _nll_quality(questions),
                        "questions": questions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
            capacities.append(capacity)
            costs.append(cost)
            all_questions.extend(questions)
    temporary.replace(directory / "trajectories.jsonl")
    return {
        "method": task.cfg.method,
        "stage": task.cfg.stage,
        "split": "dev",
        "dataset_signature": metadata["dataset_signature"],
        "trajectories": len(trajectories),
        "quality": _nll_quality(all_questions),
        "capacity": {
            name: sum(row[name] for row in capacities) / len(capacities) for name in capacities[0]
        },
        "costs": {name: sum(row[name] for row in costs) for name in costs[0]},
        "aggregation": {
            "quality": "question_mean",
            "capacity": "trajectory_mean",
            "mean_slots": "mean over all observed memory states, including initial state",
            "costs": "sum",
            "age": "number of source segments after the evidence segment",
        },
        "objective": asdict(task.cfg),
        "metadata": metadata,
    }


def _positive_integer(value):
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return integer


def _threshold(value):
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0:
        raise argparse.ArgumentTypeError("threshold must be finite and nonnegative")
    return threshold


def _comparison_row(condition, threshold, summary, generate):
    capacity, quality = summary["capacity"], summary["quality"]
    updates = capacity["appends"] + capacity["overwrites"]
    row = {
        "condition": condition,
        "threshold_i": threshold,
        "append_ratio": capacity["appends"] / updates if updates else None,
        "update_points": round(updates * summary["trajectories"]),
        "final_slots": capacity["final_slots"],
        "mean_slots": capacity["mean_slots"],
        **{f"{group}_nll": quality[group]["nll"] for group in ("all", "old", "new")},
        **{
            name: summary["costs"][name]
            for name in ("write_calls", "write_seconds", "build_seconds", "nll_seconds")
        },
    }
    if generate:
        row.update(
            {
                f"{group}_{metric}": quality[group][metric]
                for group in ("all", "old", "new")
                for metric in ("em", "f1")
            }
        )
        row["generation_seconds"] = summary["costs"]["generation_seconds"]
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-dev-samples",
        type=parse_optional_positive_integer,
        help="开发集样本数上限：正整数或 null（不限），默认不限",
    )
    thresholds_group = parser.add_mutually_exclusive_group(required=True)
    thresholds_group.add_argument("--thresholds-file", type=Path)
    thresholds_group.add_argument("--thresholds", nargs="+", type=_threshold)
    parser.add_argument("--generate", action="store_true", help="同时计算 greedy EM/F1")
    parser.add_argument("--max-new-tokens", type=_positive_integer, default=64)
    args = parser.parse_args(argv)
    directory = Path(args.output_dir)
    if directory.exists():
        raise ValueError(f"output directory already exists: {directory}")
    task, trajectories, metadata = load_dev_task(
        args.checkpoint, args.dataset_dir, args.device, args.max_dev_samples
    )
    if args.thresholds_file is not None:
        candidates = json.loads(args.thresholds_file.read_text(encoding="utf-8"))
        if Path(candidates["checkpoint"]).resolve() != Path(metadata["checkpoint"]).resolve():
            raise ValueError("thresholds file was produced from a different checkpoint")
        if candidates["source_dataset_signature"] != metadata["source_dataset_signature"]:
            raise ValueError("thresholds file was produced from a different dev dataset")
        thresholds = [_threshold(value) for value in candidates["thresholds"]]
        if not thresholds:
            raise ValueError("thresholds file contains no thresholds")
    else:
        thresholds = args.thresholds
    thresholds = list(dict.fromkeys(thresholds))
    directory.mkdir(parents=True, exist_ok=False)
    conditions = [("always_append", None, True), ("always_overwrite", None, False)]
    conditions.extend((f"threshold-{threshold}", threshold, None) for threshold in thresholds)
    original_config = task.cfg
    comparison = []
    for condition, threshold, append in conditions:
        if append is None:
            task.cfg = replace(original_config, threshold_i=threshold)
            current_task = task
        else:
            current_task = FixedActionTask(task, append)
        current_task.eval()
        condition_metadata = {
            **metadata,
            "condition": condition,
            "forced_action": None if append is None else "append" if append else "overwrite",
            "threshold_i": threshold,
        }
        condition_directory = directory / condition
        condition_directory.mkdir()
        if args.generate:
            summary = evaluate(
                current_task,
                trajectories,
                condition_directory,
                "dev",
                args.max_new_tokens,
                metadata=condition_metadata,
            )
        else:
            summary = evaluate_nll(
                current_task, trajectories, condition_directory, condition_metadata
            )
        row = _comparison_row(condition, threshold, summary, args.generate)
        summary["capacity"]["append_ratio"] = row["append_ratio"]
        summary["aggregation"]["append_ratio"] = "sum(appends) / sum(appends + overwrites)"
        save_json(condition_directory / "summary.json", summary)
        comparison.append(row)
        save_json(
            directory / "summary.json",
            {
                "metadata": metadata,
                "split": "dev",
                "generate": args.generate,
                "max_new_tokens": args.max_new_tokens if args.generate else None,
                "trajectories": len(trajectories),
                "conditions": comparison,
            },
        )
        with (directory / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(comparison)
        ratio = "n/a" if row["append_ratio"] is None else f"{row['append_ratio']:.3f}"
        print(
            f"{condition}: append_ratio={ratio}, final_slots={row['final_slots']:.1f}, "
            f"nll={row['all_nll']:.4f}, write_calls={row['write_calls']}",
            flush=True,
        )
    print(f"比较表：{directory / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
