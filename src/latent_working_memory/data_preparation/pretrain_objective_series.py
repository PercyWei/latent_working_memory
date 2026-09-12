"""依次运行短文本目标对比，双卡训练、双来源并行测试，并保存比较报告。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S UTC+08:00")


def run_series(spec, output):
    output.mkdir(parents=True, exist_ok=True)
    plan = output / "plan"
    plan.mkdir(exist_ok=True)
    selection = json.loads((Path(spec["data_dir"]) / "selection.json").read_text())
    evaluation_dirs = selection["evaluation_dirs"]
    (plan / "evaluation-dirs.json").write_text(json.dumps(evaluation_dirs, indent=2) + "\n")
    (plan / "series.json").write_text(json.dumps(spec, indent=2) + "\n")
    status = {"started": now(), "pid": os.getpid(), "status": "running", "jobs": {},
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "gpus": spec["gpus"]}
    commands = []
    python = str(Path(sys.executable).absolute())
    groups = ["--swanlab-project", spec["project"], "--swanlab-group", spec["group"],
              "--swanlab-tag", "study:pretrain-objective-comparison"]

    def save_status():
        status["updated"] = now()
        temporary = plan / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(plan / "status.json")

    def execute(name, argv, devices):
        env = os.environ | {"LWM_ALLOWED_PHYSICAL_GPUS": "4,5", "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices)),
                            "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
        with (plan / f"{name}.log").open("a") as handle:
            handle.write(f"\n{now()}\n")
            handle.flush()
            result = subprocess.run(argv, env=env, stdout=handle, stderr=subprocess.STDOUT)
        return result.returncode

    def stage(jobs):
        for name, argv, devices in jobs:
            status["jobs"][name] = {"status": "running", "started": now()}
            commands.append({"name": name, "argv": argv, "CUDA_VISIBLE_DEVICES": devices})
        (plan / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        save_status()
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [(name, executor.submit(execute, name, argv, devices)) for name, argv, devices in jobs]
            failures = []
            for name, future in futures:
                code = future.result()
                status["jobs"][name].update(status="complete" if code == 0 else "failed",
                                             exit_code=code, finished=now())
                save_status()
                if code:
                    failures.append(name)
        if failures:
            raise RuntimeError(f"failed jobs: {failures}; see plan logs")

    save_status()
    reports, evaluation_outputs = [], []
    try:
        for run in spec["runs"]:
            directory = output / "train" / run["name"]
            argv = [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                    "-m", "latent_working_memory.v1.train", "--phase", "pretrain",
                    "--config", run["config"], "--data-dir", str(Path(spec["data_dir"]) / "mixed"),
                    "--evaluation-dirs", str(plan / "evaluation-dirs.json"),
                    "--output-dir", str(directory), "--max-steps", str(spec["max_steps"]),
                    "--save-every", "1000", "--swanlab-mode", "online", *groups]
            if "fork_from_run" in run:
                argv += ["--fork-from", str(output / "train" / run["fork_from_run"] /
                                          "checkpoints/pretrain-step-005000.pt")]
            stage([(run["label"] + "-train", argv, spec["gpus"])])
            checkpoint = directory / f"checkpoints/pretrain-step-{spec['max_steps']:06d}.pt"
            eval_output = output / "eval" / run["evaluation_name"]
            jobs = []
            for (source, data_dir), gpu in zip(evaluation_dirs.items(), spec["gpus"], strict=True):
                argv = [python, "-m", "latent_working_memory.v1.evaluate", "--checkpoint", str(checkpoint),
                        "--data-dir", data_dir, "--output-dir", str(eval_output / source),
                        "--split", "test", "--examples", "240", "--generation-examples", "60",
                        "--prefix-tokens", "1", "8", "32",
                        "--swanlab-mode", "disabled"]
                jobs.append((run["label"] + "-test-" + source, argv, [gpu]))
                reports.append({"training_source": run["label"], "evaluation_source": source,
                                "report": str((eval_output / source /
                                               f"test-step-{spec['max_steps']:06d}.json").resolve())})
            stage(jobs)
            evaluation_outputs += ["--evaluation-output", run["label"], str(eval_output)]
        (plan / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
        argv = [python, "-m", "latent_working_memory.v1.publish_reports", "--reports", str(plan / "reports.json"),
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
        status.update(status="complete", finished=now())
        save_status()
    except BaseException as error:
        status.update(status="failed", error=str(error), finished=now())
        save_status()
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
    lines += ["", "以上为固定 step 20,000 的独立 test，NLL 不含 EOS。结果来自单模型 seed；"
              "warm-up 与直接联合的目标暴露量不同。C 前 5,000 步继承 A，完整轨迹关联见 plan/series.json。", "",
              "| 训练组 | 训练 | 评估 |", "|---|---|---|"]
    for run in spec["runs"]:
        train = json.loads((output / "train" / run["name"] / "swanlab.json").read_text())
        evaluation = json.loads((output / "eval" / run["evaluation_name"] / "swanlab.json").read_text())
        lines.append(f"| {run['label']} | [训练]({train['url']}) | [评估]({evaluation['url']}) |")
    comparison = json.loads((output / "compare" / spec["comparison_name"] / "swanlab.json").read_text())
    lines += ["", f"跨组比较：[SwanLab]({comparison['url']})。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir / "plan/status.json").exists():
        raise FileExistsError("series already has execution state; inspect it before restarting")
    spec = json.loads(args.spec.read_text())
    if spec["gpus"] != [4, 5]:
        raise ValueError("this experiment is authorized only on physical GPUs 4 and 5")
    run_series(spec, args.output_dir)


if __name__ == "__main__":
    main()
