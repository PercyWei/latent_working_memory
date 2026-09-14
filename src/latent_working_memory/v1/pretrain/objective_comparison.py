"""运行 AE-only、联合训练及 AE warm-up 对比实验。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from latent_working_memory.v1.experiment_execution import ExperimentExecution, now


def run_series(spec, output):
    output.mkdir(parents=True, exist_ok=True)
    plan = output / "plan"
    plan.mkdir(exist_ok=True)
    selection = json.loads(Path(spec["data_selection"]).read_text())
    evaluation_sources = list(selection["sources"])
    (plan / "data-selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (plan / "series.json").write_text(json.dumps(spec, indent=2) + "\n")
    execution = ExperimentExecution(plan, spec["gpus"])
    python = str(Path(sys.executable).absolute())
    groups = ["--swanlab-project", spec["project"], "--swanlab-group", spec["group"],
              "--swanlab-tag", "study:pretrain-objective-comparison"]
    stage = execution.stage
    reports, evaluation_outputs = [], []
    try:
        training_jobs = []
        for run in spec["runs"]:
            directory = output / "train" / run["name"]
            argv = [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                    "-m", "latent_working_memory.v1.pretrain.train", "--phase", "pretrain",
                    "--config", run["config"], "--data-selection", str(plan / "data-selection.json"),
                    "--data-run", "mixed",
                    "--output-dir", str(directory), "--max-steps", str(spec["max_steps"]),
                    "--save-every", str(spec["save_every"]), "--swanlab-mode", "online", *groups]
            if "fork_from_run" in run:
                argv += ["--fork-from", str(output / "train" / run["fork_from_run"] /
                                          f"checkpoints/pretrain-step-{run['fork_step']:06d}.pt")]
            training_jobs.append((run["label"] + "-train", argv, spec["gpus"]))
        # Each training job occupies both GPUs; the parent finishes before warm-up forks.
        for job in training_jobs:
            stage([job])
        for run in spec["runs"]:
            directory = output / "train" / run["name"]
            checkpoint = directory / f"checkpoints/pretrain-step-{spec['max_steps']:06d}.pt"
            eval_output = output / "eval" / run["evaluation_name"]
            jobs = []
            for source, gpu in zip(evaluation_sources, spec["gpus"], strict=True):
                argv = [python, "-m", "latent_working_memory.v1.pretrain.evaluate", "--checkpoint", str(checkpoint),
                        "--data-selection", str(plan / "data-selection.json"), "--evaluation-source", source,
                        "--output-dir", str(eval_output / source),
                        "--split", "test", "--examples", str(spec["evaluation"]["examples"]),
                        "--generation-examples", str(spec["evaluation"]["generation_examples"]),
                        "--prefix-tokens", *map(str, spec["evaluation"]["prefix_tokens"]),
                        "--swanlab-mode", "disabled"]
                jobs.append((run["label"] + "-test-" + source, argv, [gpu]))
                reports.append({"training_source": run["label"], "evaluation_source": source,
                                "report": str((eval_output / source /
                                               f"test-step-{spec['max_steps']:06d}.json").resolve())})
            stage(jobs)
            evaluation_outputs += ["--training-run", run["label"], str(directory)]
        (plan / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
        argv = [python, "-m", "latent_working_memory.v1.pretrain.publish_reports", "--reports", str(plan / "reports.json"),
                "--output-dir", str(output / "compare" / spec["comparison_name"]),
                "--swanlab-mode", "online", *groups, *evaluation_outputs]
        stage([("publish-comparison", argv, spec["gpus"])])
        result = summarize(spec, output, reports)
        (plan / "results.md").write_text(result)
        note = Path(spec["note"])
        text = note.read_text()
        text = "\n".join("最后修订时间：" + now() if line.startswith("最后修订时间：") else line
                         for line in text.splitlines())
        note.write_text(text + "\n\n" + result)
        execution.finish()
    except BaseException as error:
        execution.finish(error)
        raise


def summarize(spec, output, reports):
    lines = [f"完成时间：{now()}", "", "| 训练组 | 测试来源 | AE NLL | LM NLL | AE BLEU-4 | 正确前缀 | 整段匹配 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for entry in reports:
        result = json.loads(Path(entry["report"]).read_text())
        ae = result["groups"]["all/ae/memory"]
        lm = result["groups"]["all/continuation/memory"]
        lines.append(f"| {entry['training_source']} | {entry['evaluation_source']} | {ae['nll']:.4f} | "
                     f"{lm['nll']:.4f} | {ae['bleu_4']:.3f} | {ae['correct_prefix_ratio']:.3%} | {ae['exact_match']:.3%} |")
    lines += ["", f"以上为 step {spec['max_steps']} 的独立 test，NLL 不含 EOS。结果来自单模型 seed；"
              "warm-up 与直接联合的目标暴露量不同。继承的 checkpoint 见 plan/series.json。", "",
              "| 训练组 | 训练 | 评估 |", "|---|---|---|"]
    for run in spec["runs"]:
        train = json.loads((output / "train" / run["name"] / "swanlab.json").read_text())
        evaluation = train
        lines.append(f"| {run['label']} | [训练]({train['url']}) | [评估]({evaluation['url']}) |")
    comparison = json.loads((output / "compare" / spec["comparison_name"] / "swanlab.json").read_text())
    lines += ["", f"跨组比较：[SwanLab]({comparison['url']})。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("use a new experiment directory")
    spec = json.loads(args.spec.read_text())
    if spec["gpus"] != [4, 5]:
        raise ValueError("this experiment is authorized only on physical GPUs 4 and 5")
    run_series(spec, args.output_dir)


if __name__ == "__main__":
    main()
