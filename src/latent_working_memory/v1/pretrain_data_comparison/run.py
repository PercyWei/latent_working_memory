"""运行 semantic、random、mixed 预训练数据类型对比。"""
import argparse
import json
from pathlib import Path
import sys

from latent_working_memory.v1.experiment_execution import ExperimentExecution


def run_series(spec, output):
    plan = output / "plan"
    plan.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(spec["data_selection"]).read_text())
    (plan / "data-selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (plan / "series.json").write_text(json.dumps(spec, indent=2) + "\n")
    execution = ExperimentExecution(plan, spec["gpus"])
    python = str(Path(sys.executable).absolute())
    groups = ["--swanlab-project", spec["project"], "--swanlab-group", spec["group"],
              "--swanlab-tag", "study:boundary-comparison"]
    reports, training_runs = [], []
    try:
        for run in spec["runs"]:
            directory = output / "train" / run["name"]
            execution.stage([(run["data_run"] + "-train", [
                python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                "-m", "latent_working_memory.v1.train", "--phase", "pretrain",
                "--config", spec["config"], "--data-selection", str(plan / "data-selection.json"),
                "--data-run", run["data_run"], "--output-dir", str(directory),
                "--max-steps", str(spec["max_steps"]), "--save-every", str(spec["save_every"]),
                "--swanlab-mode", "online", *groups,
            ], spec["gpus"])])
            jobs = []
            for source, gpu in zip(selection["sources"], spec["gpus"], strict=True):
                destination = output / "eval" / run["evaluation_name"] / source
                checkpoint = directory / f"checkpoints/pretrain-step-{spec['max_steps']:06d}.pt"
                jobs.append((run["data_run"] + "-test-" + source, [
                    python, "-m", "latent_working_memory.v1.evaluate", "--checkpoint", str(checkpoint),
                    "--data-selection", str(plan / "data-selection.json"),
                    "--evaluation-source", source, "--output-dir", str(destination),
                    "--split", "test", "--examples", str(spec["evaluation"]["examples"]),
                    "--generation-examples", str(spec["evaluation"]["generation_examples"]),
                    "--swanlab-mode", "disabled",
                ], [gpu]))
                reports.append({"training_source": run["data_run"], "evaluation_source": source,
                                "report": str((destination / f"test-step-{spec['max_steps']:06d}.json").resolve())})
            execution.stage(jobs)
            training_runs += ["--training-run", run["data_run"], str(directory)]
        (plan / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
        execution.stage([("publish-comparison", [
            python, "-m", "latent_working_memory.v1.publish_reports",
            "--reports", str(plan / "reports.json"),
            "--output-dir", str(output / "compare" / spec["comparison_name"]),
            "--swanlab-mode", "online", *groups, *training_runs,
        ], spec["gpus"])])
        execution.finish()
    except BaseException as error:
        execution.finish(error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("use a new experiment directory")
    spec = json.loads(args.spec.read_text())
    if len(spec["gpus"]) != 2 or len(set(spec["gpus"])) != 2:
        raise ValueError("data comparison requires two distinct physical GPUs")
    run_series(spec, args.output_dir)


if __name__ == "__main__":
    main()
