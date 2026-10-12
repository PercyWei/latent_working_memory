"""读取已完成的 warmup/policy 日志，列出可用于冻结诊断的 warmup checkpoint。"""

import argparse
import json
from pathlib import Path
from statistics import mean


METRICS = (
    "loss",
    "qa_new_nll",
    "qa_old_nll",
    "grad_norm",
    "append_ratio",
    "gate_I",
    "slots_final",
)


def _points(records, section):
    points = []
    for record in records:
        values = {}
        for name in METRICS:
            key = f"{section}/{name}"
            if key not in record or record[key] is None:
                continue
            if name in {"qa_new_nll", "qa_old_nll"}:
                age = name.removeprefix("qa_").removesuffix("_nll")
                if record[f"{section}/qa_{age}_count"] == 0:
                    continue
            values[name] = record[key]
        if f"{section}/appends" in record:
            appends = record[f"{section}/appends"]
            decisions = appends + record[f"{section}/overwrites"]
            if decisions > 0:
                values["append_ratio"] = appends / decisions
        if values:
            points.append(
                {
                    "step": record["step"],
                    "global_step": record["global_step"],
                    **values,
                }
            )
    return points


def _statistics(points):
    result = {}
    for name in METRICS:
        recorded = [point for point in points if name in point]
        if not recorded:
            continue
        values = [point[name] for point in recorded]
        result[name] = {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
            "max_step": recorded[values.index(max(values))]["step"],
        }
    return result


def inspect_run(run_dir, output_dir):
    run_dir = Path(run_dir).resolve()
    stages = {}
    for stage in ("warmup", "policy"):
        directory = run_dir / stage
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        objective = config["objective"]
        if objective["method"] != "memory_change" or objective["stage"] != stage:
            raise ValueError(f"{directory} 必须是 memory_change 的 {stage} 阶段目录")
        records = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        train = _points(records, "train")
        dev = _points(records, "dev")
        gradient_clip = config["training"]["gradient_clip"]
        stages[stage] = {
            "directory": str(directory),
            "config": config,
            "recorded_steps": len(records),
            "train": _statistics(train),
            "dev": _statistics(dev),
            "train_first": train[0] if train else None,
            "train_last": train[-1] if train else None,
            "dev_points": dev,
            "gradient_clip": gradient_clip,
            "steps_above_gradient_clip": sum(
                point["grad_norm"] > gradient_clip for point in train if "grad_norm" in point
            ),
            "train_all_append_steps": [
                point["step"] for point in train if point.get("append_ratio") == 1.0
            ],
        }

    dev_by_step = {point["step"]: point for point in stages["warmup"]["dev_points"]}
    checkpoints = []
    for state in sorted(
        (run_dir / "warmup" / "checkpoints").glob("global_step_*/state.pt"),
        key=lambda path: int(path.parent.name.removeprefix("global_step_")),
    ):
        step = int(state.parent.name.removeprefix("global_step_"))
        checkpoints.append({"path": str(state.parent), "step": step, "dev": dev_by_step.get(step)})
    summary = {
        "run_dir": str(run_dir),
        "stages": stages,
        "warmup_checkpoints": checkpoints,
        "notes": [
            "统计均值是已记录 optimizer step 的均值，不是逐问题或逐决策的重新加权结果。",
            "grad_norm 是裁剪前的梯度范数；超过 gradient_clip 不等于训练失效。",
            "gate_I 是日志中的聚合均值；warmup 未记录 I，不能由本报告推断其分布。",
            "checkpoint 只关联同一步已经记录的 dev 值；未评估的步数保留为空。",
        ],
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _number(value):
    return "未记录" if value is None else f"{value:.5g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="含 warmup/policy 的方法目录")
    parser.add_argument("--output-dir", required=True, type=Path, help="新建的检查报告目录")
    args = parser.parse_args()
    summary = inspect_run(args.run_dir, args.output_dir)
    for name, stage in summary["stages"].items():
        print(f"{name}: {stage['recorded_steps']} 个 optimizer step")
        for section in ("train", "dev"):
            metrics = stage[section]
            pieces = [
                f"{key}={_number(metrics[key]['first'])} → {_number(metrics[key]['last'])}"
                for key in ("loss", "qa_new_nll", "qa_old_nll", "append_ratio", "gate_I")
                if key in metrics
            ]
            print(f"  {section} 首次 → 最后: " + "; ".join(pieces))
        norm = stage["train"].get("grad_norm")
        if norm is not None:
            print(
                f"  grad_norm max={_number(norm['max'])} (step {norm['max_step']}), "
                f"超过裁剪值 {stage['gradient_clip']}: "
                f"{stage['steps_above_gradient_clip']}/{norm['count']} 步"
            )
    print("可用 warmup checkpoint（按 step 排列）：")
    for checkpoint in summary["warmup_checkpoints"]:
        dev = checkpoint["dev"]
        values = (
            "同一步未记录 dev"
            if dev is None
            else ", ".join(
                f"{key}={_number(dev.get(key))}" for key in ("loss", "qa_new_nll", "qa_old_nll")
            )
        )
        print(f"  {checkpoint['path']} | {values}")
    if not summary["warmup_checkpoints"]:
        print("  无 state.pt 已完成保存的 checkpoint")
    print(f"详细报告: {args.output_dir.resolve() / 'summary.json'}")


if __name__ == "__main__":
    main()
