"""运行独立动态实验：准备选样和评估记录、训练、最终 test。"""

import json
from pathlib import Path
import sys

from latent_working_memory.v1.dynamic.config import load_dynamic_config
from latent_working_memory.v1.experiment_execution import (
    experiment_parser, load_experiment, run_experiments,
)


def load_dynamic_experiment(path):
    spec = load_experiment(path, {"checkpoint"})
    recipe = load_dynamic_config(Path(spec["model"]))
    if recipe.global_batch_size % len(spec["gpus"]):
        raise ValueError("global batch size must divide evenly across GPUs")
    selection = json.loads(Path(spec["selection"]).read_text())
    if set(selection) != {"dataset", "dataset_dir"} or selection["dataset"] not in {"squad", "personamem"}:
        raise ValueError("dynamic selection requires dataset and dataset_dir")
    if not isinstance(spec["checkpoint"], str) or not spec["checkpoint"]:
        raise ValueError("dynamic experiment requires an initialization checkpoint")
    return spec


def build_stages(spec, original, output, args):
    name = spec["name"]
    recipe = load_dynamic_config(Path(original["model"]))
    selection = json.loads(Path(original["selection"]).read_text())
    plan = output / "plan" / name
    train = output / "train" / name
    python = sys.executable
    common = ["--config", spec["model"], "--dataset", selection["dataset"],
              "--dataset-dir", selection["dataset_dir"]]
    prepare = [python, "-m", "latent_working_memory.v1.dynamic.prepare", *common,
               "--checkpoint", spec["checkpoint"], "--output-dir", str(plan)]
    distributed = [python, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={len(spec['gpus'])}", "-m", "latent_working_memory.v1.dynamic.run"]
    tracking = ["--swanlab-mode", args.swanlab_mode, "--swanlab-project", spec["swanlab_project"]]
    if args.swanlab_group:
        tracking += ["--swanlab-group", args.swanlab_group]
    for tag in args.swanlab_tag:
        tracking += ["--swanlab-tag", tag]
    shared = [*common, "--evaluation-plan", str(plan / "evaluation-plan.json"), *tracking]
    training = [*distributed, "train", *shared, "--checkpoint", spec["checkpoint"],
                "--output-dir", str(train)]
    steps = recipe.epochs * recipe.micro_epochs_per_epoch * recipe.steps_per_micro_epoch
    checkpoint = train / f"checkpoints/dynamic-step-{steps:06d}.pt"
    evaluation = [*distributed, "evaluate", *shared, "--checkpoint", str(checkpoint),
                  "--split", "test", "--output-dir", str(output / "eval" / f"{name}_eval")]
    reports = [{"name": name,
                "report": str(output / "eval" / f"{name}_eval" / f"test-step-{steps:06d}.json")}]
    return ([[(name + "-prepare", prepare, spec["gpus"][:1])],
             [(name + "-train", training, spec["gpus"])],
             [(name + "-test", evaluation, spec["gpus"])]], reports)


def main(argv=None):
    args = experiment_parser(__doc__).parse_args(argv)
    run_experiments(args, load_dynamic_experiment, build_stages)


if __name__ == "__main__":
    main()
