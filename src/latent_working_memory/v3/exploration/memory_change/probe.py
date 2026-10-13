"""在固定随机动作轨迹上，测量 I 与 append/overwrite 的反事实 QA 损失。"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from statistics import mean

import torch

from latent_working_memory.v3.config import parse_optional_positive_integer
from latent_working_memory.v3.exploration.memory_change.common import load_dev_task, save_json
from latent_working_memory.v3.objective import memory_change_score
from latent_working_memory.v3.segmentation import example_rng


def _distribution(values):
    if not values:
        return None
    quantiles = torch.quantile(
        torch.tensor(values, dtype=torch.float64),
        torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=torch.float64),
    ).tolist()
    return {
        "count": len(values),
        "mean": mean(values),
        **dict(zip(("min", "p10", "p25", "p50", "p75", "p90", "max"), quantiles, strict=True)),
    }


def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        for index in order[start:stop]:
            ranks[index] = (start + stop - 1) / 2
        start = stop
    return ranks


def _spearman(xs, ys):
    if len(xs) < 2 or len(set(xs)) == 1 or len(set(ys)) == 1:
        return None
    return float(torch.corrcoef(torch.tensor([_ranks(xs), _ranks(ys)], dtype=torch.float64))[0, 1])


def _group_summary(rows, threshold):
    changes = [row["I"] for row in rows]
    return {
        "updates": len(rows),
        "I": _distribution(changes),
        "fraction_I_ge_original_threshold": mean(value >= threshold for value in changes),
        **{
            f"delta_{group}": _distribution([row["losses"][group]["delta"] for row in rows])
            for group in ("old", "new", "all")
        },
        "damage_old": _distribution([row["losses"]["old"]["damage"] for row in rows]),
        "fraction_append_better": {
            group: mean(row["losses"][group]["delta"] > 0 for row in rows)
            for group in ("old", "new", "all")
        },
        "spearman_I_delta": {
            group: _spearman(changes, [row["losses"][group]["delta"] for row in rows])
            for group in ("old", "new", "all")
        },
    }


def _score_bins(rows, threshold):
    quantiles = _distribution([row["I"] for row in rows])
    boundaries = sorted({quantiles[name] for name in ("p25", "p50", "p75", "max")})
    bins = []
    lower = None
    for upper in boundaries:
        selected = [
            row for row in rows if (lower is None or row["I"] > lower) and row["I"] <= upper
        ]
        if selected:
            bins.append(
                {
                    "lower_exclusive": lower,
                    "upper_inclusive": upper,
                    **_group_summary(selected, threshold),
                }
            )
        lower = upper
    return bins


def _loss_summary(questions):
    result = {
        "questions": len(questions),
        "Lrw": mean(row["Lrw"] for row in questions),
        "Lapp": mean(row["Lapp"] for row in questions),
        "delta": mean(row["Lrw"] - row["Lapp"] for row in questions),
    }
    if all("L0" in row for row in questions):
        result["L0"] = mean(row["L0"] for row in questions)
        result["damage"] = mean(row["Lrw"] - row["L0"] for row in questions)
    return result


@torch.no_grad()
def probe(task, trajectories, output_dir, metadata, append_probability):
    if not 0 <= append_probability <= 1:
        raise ValueError("append_probability 必须在 [0, 1] 内")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    task.eval()
    precision = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if task.device.type == "cuda"
        else nullcontext()
    )
    with precision, (directory / "updates.jsonl").open("w", encoding="utf-8") as stream:
        for index, trajectory in enumerate(trajectories):
            rng = example_rng(task.cfg.seed, 0, trajectory.trajectory_id)
            initial = task._event(0, trajectory.segments[0].segment_id)
            blocks = task._write_batch(
                [trajectory.segments[0].input_ids], [[]], [initial], actions=["initial"]
            )
            positions = {
                segment.segment_id: step for step, segment in enumerate(trajectory.segments)
            }
            for step, segment in enumerate(trajectory.segments[1:], start=1):
                event = task._event(step, segment.segment_id)
                # 两个候选共享完全相同的旧状态，只有抽中的动作进入下一更新点。
                rewritten, appended = task._write_updates_batch(
                    [segment.input_ids, segment.input_ids],
                    [blocks, blocks],
                    [False, True],
                    [event, event],
                )
                change = float(memory_change_score(blocks[-1], rewritten, task.cfg.rms_epsilon))
                overwrite_state = blocks[:-1] + [rewritten]
                append_state = blocks + [appended]
                # 仅使用已读到的 task 题，gate 题和未来段的问题不参与质量诊断。
                old_ids = [
                    qa.qa_id
                    for qa in trajectory.qas.values()
                    if qa.role == "task" and positions[qa.segment_id] < step
                ]
                new_ids = [
                    qa.qa_id
                    for qa in trajectory.qas.values()
                    if qa.role == "task" and positions[qa.segment_id] == step
                ]
                qa_ids = old_ids + new_ids
                l0 = task.qa_losses(blocks, trajectory, old_ids).cpu().tolist()
                lrw = task.qa_losses(overwrite_state, trajectory, qa_ids).cpu().tolist()
                lapp = task.qa_losses(append_state, trajectory, qa_ids).cpu().tolist()
                questions = []
                for qindex, (qid, rw, app) in enumerate(zip(qa_ids, lrw, lapp, strict=True)):
                    old = qindex < len(old_ids)
                    questions.append(
                        {
                            "qa_id": qid,
                            "age": step - positions[trajectory.qas[qid].segment_id],
                            "group": "old" if old else "new",
                            "Lrw": rw,
                            "Lapp": app,
                            **({"L0": l0[qindex]} if old else {}),
                        }
                    )
                append = rng.random() < append_probability
                row = {
                    "trajectory_id": trajectory.trajectory_id,
                    "step": step,
                    "segment_id": segment.segment_id,
                    "tail_slots": len(blocks[-1]),
                    "slots_before": sum(map(len, blocks)),
                    "I": change,
                    "action": "append" if append else "overwrite",
                    "losses": {
                        "old": _loss_summary(questions[: len(old_ids)]),
                        "new": _loss_summary(questions[len(old_ids) :]),
                        "all": _loss_summary(questions),
                    },
                    "questions": questions,
                }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                # 汇总仅保留更新点统计，逐题值完整保存在 JSONL。
                rows.append({key: value for key, value in row.items() if key != "questions"})
                blocks = append_state if append else overwrite_state
            print(
                f"[probe] {index + 1}/{len(trajectories)} trajectories, {len(rows)} updates",
                flush=True,
            )
    if not rows:
        raise ValueError("开发集没有可诊断的记忆更新点")
    threshold = task.cfg.threshold_i
    overall = _group_summary(rows, threshold)
    thresholds = {threshold, *(overall["I"][name] for name in ("p25", "p50", "p75"))}
    by_tail = {}
    for size in sorted({row["tail_slots"] for row in rows}):
        selected = [row for row in rows if row["tail_slots"] == size]
        by_tail[str(size)] = {
            **_group_summary(selected, threshold),
            "I_bins": _score_bins(selected, threshold),
        }
        # 两种末块大小可能处于不同的分数区间，两者分位点都纳入扫描。
        thresholds.update(by_tail[str(size)]["I"][name] for name in ("p25", "p50", "p75"))
    summary = {
        "metadata": metadata,
        "rollout": {
            "policy": "random",
            "append_probability": append_probability,
            "seed": task.cfg.seed,
            "epoch": 0,
        },
        "original_threshold": threshold,
        "overall": overall,
        "by_tail_slots": by_tail,
        "by_step": {
            str(step): _group_summary([row for row in rows if row["step"] == step], threshold)
            for step in sorted({row["step"] for row in rows})
        },
        "I_bins": _score_bins(rows, threshold),
        "candidate_thresholds": sorted(thresholds),
        "definitions": {
            "I": "正式策略的逐 slot RMS 归一化记忆变化；不改写持久记忆",
            "delta": "Lrw - Lapp；正值表示 append 的当前 QA NLL 更低",
            "damage_old": "旧题 Lrw - L0；正值表示覆盖后旧题 NLL 上升",
            "all": "旧题和新题合并，按问题数平均；不将 old/new 等权平均",
            "aggregation": "分布与相关性以更新点为单位；更新点内 NLL 按题平均",
            "scope": "冻结写入器的随机轨迹反事实；不等同于阈值策略的完整 rollout",
        },
    }
    save_json(directory / "summary.json", summary)
    save_json(
        directory / "thresholds.json",
        {
            "checkpoint": metadata["checkpoint"],
            "source_dataset_signature": metadata["source_dataset_signature"],
            "thresholds": sorted(thresholds),
        },
    )
    return summary


def main():
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
    parser.add_argument(
        "--append-probability", type=float, default=0.5, help="诊断轨迹的随机 append 概率，默认 0.5"
    )
    args = parser.parse_args()
    if not 0 <= args.append_probability <= 1:
        parser.error("--append-probability 必须在 [0, 1] 内")
    task, trajectories, metadata = load_dev_task(
        args.checkpoint, args.dataset_dir, args.device, args.max_dev_samples
    )
    summary = probe(task, trajectories, args.output_dir, metadata, args.append_probability)
    for size, group in summary["by_tail_slots"].items():
        print(
            f"tail={size}: updates={group['updates']}, I_p50={group['I']['p50']:.4f}, "
            f"Δold={group['delta_old']['mean']:.4f}, Δnew={group['delta_new']['mean']:.4f}, "
            f"Spearman(I, Δall)={group['spearman_I_delta']['all']}"
        )
    print(f"报告: {Path(args.output_dir) / 'summary.json'}")
    print(f"候选阈值: {summary['candidate_thresholds']}")


if __name__ == "__main__":
    main()
