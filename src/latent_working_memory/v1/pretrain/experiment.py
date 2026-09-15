"""运行一个或多个独立预训练实验，完成训练及最终 test。"""

import json
from pathlib import Path
import sys

from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.experiment_execution import (
    experiment_parser,
    load_experiment,
    run_experiments,
)
from latent_working_memory.v1.pretrain.data_selection import validate_selection
from latent_working_memory.v1.pretrain.curriculum import validate_curriculum


def load_pretrain_experiment(path):
    spec = load_experiment(path, {"epochs", "max_samples_per_epoch", "save_every", "evaluation"})
    raw = json.loads(Path(spec["model"]).read_text())
    unrelated = {
        "dynamic_k_first",
        "growth_actions",
        "exploration_probs",
        "bptt_tokens",
        "pretrain_dataset",
        "pretrain_subset",
        "split_fractions",
        "ae_weight",
        "lm_weight",
        "pretrain_balanced_batches",
        "pretrain_ae_warmup_steps",
        "input_length_weights",
        "input_length_weights_end",
        "input_length_curriculum_steps",
        "ratio_curriculum_steps",
        "lr_decay_steps",
    }
    if raw.keys() & unrelated:
        raise ValueError("pretrain model config contains data construction or other-stage fields")
    config = ExperimentConfig.from_mapping(raw)
    selection = json.loads(Path(spec["selection"]).read_text())
    validate_selection(selection)
    validate_curriculum(
        selection["training"], selection["sources"], config.input_length_bounds, config.max_input_tokens
    )
    for key in ("epochs", "save_every"):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    limit = spec["max_samples_per_epoch"]
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("max_samples_per_epoch must be a positive integer or null")
    evaluation = spec["evaluation"]
    if set(evaluation) != {"examples", "generation_examples", "prefix_tokens"}:
        raise ValueError("evaluation requires examples, generation_examples and prefix_tokens")
    if (
        type(evaluation["examples"]) is not int
        or evaluation["examples"] < 2
        or type(evaluation["generation_examples"]) is not int
        or evaluation["generation_examples"] < 0
        or any(type(n) is not int or n <= 0 for n in evaluation["prefix_tokens"])
    ):
        raise ValueError("invalid final evaluation counts or prefix lengths")
    return spec


def build_stages(spec, original, output, args):
    name = spec["name"]
    train = output / "train" / name
    evaluate = output / "eval" / f"{name}_eval"
    python = sys.executable
    tracking = ["--swanlab-mode", args.swanlab_mode, "--swanlab-project", spec["swanlab_project"]]
    if args.swanlab_group:
        tracking += ["--swanlab-group", args.swanlab_group]
    for tag in args.swanlab_tag:
        tracking += ["--swanlab-tag", tag]
    training = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(spec['gpus'])}",
        "-m",
        "latent_working_memory.v1.pretrain.train",
        "--phase",
        "pretrain",
        "--config",
        spec["model"],
        "--data-selection",
        spec["selection"],
        "--output-dir",
        str(train),
        "--epochs",
        str(spec["epochs"]),
        "--tokenizer-workers",
        str(args.tokenizer_workers),
        "--tokenization-batch-size",
        str(args.tokenization_batch_size),
        "--prefetch-batches",
        str(args.prefetch_batches),
        "--save-every",
        str(spec["save_every"]),
        *tracking,
    ]
    if spec["max_samples_per_epoch"] is not None:
        training += ["--max-samples-per-epoch", str(spec["max_samples_per_epoch"])]
    result_path = train / "training-result.json"
    evaluation = [
        python,
        "-m",
        "latent_working_memory.v1.pretrain.evaluate",
        "--training-result",
        str(result_path),
        "--data-selection",
        spec["selection"],
        "--output-dir",
        str(evaluate),
        "--split",
        "test",
        "--tokenizer-workers",
        str(args.tokenizer_workers),
        "--tokenization-batch-size",
        str(args.tokenization_batch_size),
        "--examples",
        str(spec["evaluation"]["examples"]),
        "--generation-examples",
        str(spec["evaluation"]["generation_examples"]),
        *tracking,
    ]
    if spec["evaluation"]["prefix_tokens"]:
        evaluation += ["--prefix-tokens", *map(str, spec["evaluation"]["prefix_tokens"])]
    if args.swanlab_mode == "online":
        evaluation += ["--training-run", str(train)]
    # One evaluation invocation appends all sources together to the training run.
    reports = [
        {
            "training_source": name,
            "evaluation_source": source,
            "report": str(evaluate / source / "test-step-<resolved-step>.json"),
        }
        for source in json.loads(Path(original["selection"]).read_text())["sources"]
    ]
    return (
        [
            [(name + "-train", training, spec["gpus"])],
            [(name + "-test", evaluation, spec["gpus"][:1])],
        ],
        reports,
    )


def main(argv=None):
    parser = experiment_parser(__doc__)
    parser.add_argument("--tokenizer-workers", type=int, default=4)
    parser.add_argument("--tokenization-batch-size", type=int, default=256)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    args = parser.parse_args(argv)
    if args.tokenizer_workers < 0 or args.tokenization_batch_size <= 0 or args.prefetch_batches < 0:
        raise ValueError("invalid CPU tokenization settings")
    specs = run_experiments(args, load_pretrain_experiment, build_stages)
    if not args.plan_only:
        reports = []
        for spec in specs:
            result = json.loads(
                (args.output_dir / "train" / spec["name"] / "training-result.json").read_text()
            )
            for source in json.loads(Path(spec["selection"]).read_text())["sources"]:
                reports.append(
                    {
                        "training_source": spec["name"],
                        "evaluation_source": source,
                        "report": str(
                            (
                                args.output_dir
                                / "eval"
                                / f"{spec['name']}_eval"
                                / source
                                / f"test-step-{result['completed_steps']:06d}.json"
                            ).resolve()
                        ),
                    }
                )
        (args.output_dir / "plan/reports.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
