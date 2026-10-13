"""从完整 FactQA 题池确定每次更新的任务题和历史门控题。"""

from dataclasses import dataclass
import hashlib
import random


@dataclass(frozen=True)
class StepQAs:
    new_qa_ids: tuple[str, ...]
    old_qa_ids: tuple[str, ...]
    gate_qa_ids: tuple[str, ...]


def _sample(pool, count, seed, epoch, trajectory_id, step, kind):
    key = f"{seed}:qa:{epoch}:{trajectory_id}:{step}:{kind}"
    rng = random.Random(int.from_bytes(hashlib.blake2b(key.encode(), digest_size=16).digest()))
    return tuple(sorted(rng.sample(pool, count)))


def select_step_qas(trajectory, step, config, epoch=0, seed=None):
    """训练按 epoch 重采样；dev/test 固定题池和 gate，不消耗动作随机流。"""
    seed = config.seed if seed is None else seed
    epoch = epoch if trajectory.split == "train" else 0
    current = trajectory.task_qa_ids[step]
    history = tuple(qid for pool in trajectory.task_qa_ids[:step] for qid in pool)
    gates = tuple(qid for pool in trajectory.gate_qa_ids[:step] for qid in pool)
    if trajectory.split == "train":
        new_ids = _sample(
            current, config.task_new_qa_count, seed, epoch, trajectory.trajectory_id, step, "new"
        )
        old_ids = (
            _sample(
                history,
                config.task_qa_count - config.task_new_qa_count,
                seed,
                epoch,
                trajectory.trajectory_id,
                step,
                "old",
            )
            if step
            else ()
        )
    else:
        new_ids, old_ids = current, history
    gate_ids = (
        _sample(gates, config.gate_qa_count, seed, epoch, trajectory.trajectory_id, step, "gate")
        if step
        else ()
    )
    return StepQAs(new_ids, old_ids, gate_ids)


def filter_qa_budget(splits, config, include_train=True):
    """严格检查最小历史池；每个 split 保留不足原始规模的 80% 时拒绝使用。"""
    filtered, statistics = {}, {}
    new_count = config.task_new_qa_count
    old_count = config.task_qa_count - new_count
    for split, rows in splits.items():
        kept = []
        rejected = {"task_new": 0, "task_old": 0, "gate": 0}
        for row in rows:
            failures = []
            if include_train and split == "train":
                if any(len(pool) < new_count for pool in row.task_qa_ids):
                    failures.append("task_new")
                if len(row.task_qa_ids[0]) < old_count:
                    failures.append("task_old")
            if len(row.gate_qa_ids[0]) < config.gate_qa_count:
                failures.append("gate")
            if failures:
                for name in failures:
                    rejected[name] += 1
            else:
                kept.append(row)
        if 5 * len(kept) < 4 * len(rows):
            raise ValueError(
                f"FactQA {split} retains {len(kept)}/{len(rows)} trajectories below 80% "
                f"for task_new={new_count}, task_old={old_count}, "
                f"gate={config.gate_qa_count}; insufficient QA pools: {rejected}"
            )
        filtered[split] = rows if len(kept) == len(rows) else tuple(kept)
        statistics[split] = {
            "read": len(rows),
            "kept": len(kept),
            "filtered_insufficient_qas": len(rows) - len(kept),
            "insufficient_pools": rejected,
        }
    return filtered, statistics
