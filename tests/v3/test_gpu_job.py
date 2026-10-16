"""验证任务编排和 shell 契约；子进程替身不运行 GPU、模型或云端请求。"""

import json
import os
from dataclasses import replace
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import torch

from latent_working_memory import tracking_credentials
from latent_working_memory.v3 import gpu_job, job_plan
from latent_working_memory.v3.config import DYNAMIC_METHODS, METHODS, load_experiment, load_preset


@pytest.fixture(autouse=True)
def explicit_test_credentials(monkeypatch):
    # 编排测试不读取开发者的真实凭据；真实解析流程另行使用临时目录测试。
    monkeypatch.setattr(gpu_job, "swanlab_api_key", lambda: "test-api-key")


def arguments(tmp_path, *options, run_id="unit-job", checkpoint=True):
    # 后训练用例默认显式提供CLI checkpoint；缺失契约用checkpoint=False单独覆盖。
    method = next(
        (options[i + 1] for i in range(len(options) - 1, -1, -1) if options[i] == "--method"),
        "memory_change",
    )
    initialization = []
    if checkpoint and method in DYNAMIC_METHODS and "--checkpoint" not in options:
        source = tmp_path / "argument source"
        existing = (
            source / "source/dynamic-pretrain-k64_unit-job/pretrain/checkpoints/global_step_2"
        )
        initial = existing if existing.is_dir() else external_checkpoint(source)
        initialization = ["--checkpoint", str(initial)]
    return gpu_job.parse_args(
        [
            "--mode",
            "smoke",
            "--method",
            "memory_change",
            *(["--run-id", run_id] if run_id is not None else []),
            "--output-root",
            str(tmp_path / "outputs"),
            "--pretrain-data",
            str(tmp_path / "pretraining data"),
            "--qa-data",
            str(tmp_path / "qa data"),
            *initialization,
            *options,
        ]
    )


def prepare_dataset_entries(args):
    # 这里只测试编排入口的文件要求；内容解析由真正的训练/评估测试覆盖。
    for directory in (args.pretrain_data, args.qa_data):
        directory.mkdir(parents=True)
        for split in ("train", "dev", "test"):
            (directory / f"{split}.jsonl").write_text("")
    (args.pretrain_data / "preparation.json").write_text("{}")
    (args.qa_data / "preparation.json").write_text("{}")


def install_preset_stages(tmp_path, monkeypatch, filename, stages, initialization=None):
    preset = json.loads((Path("configs/v3") / filename).read_text())
    preset["objective"]["stages"] = list(stages)
    if initialization is not None:
        preset["training"]["init_checkpoint"] = str(initialization)
    path = tmp_path / filename
    path.write_text(json.dumps(preset))
    monkeypatch.setattr(
        job_plan,
        "load_preset",
        lambda requested: load_preset(path if requested.name == filename else requested),
    )


def custom_preset_file(tmp_path, method, **sections):
    preset = json.loads((Path("configs/v3") / f"{method}.json").read_text())
    for section, values in sections.items():
        preset[section].update(values)
    path = tmp_path / "custom experiment.json"
    path.write_text(json.dumps(preset))
    return path


def training_limit(config):
    return (
        config.training.max_pretrain_train_samples
        if config.objective.stage in {"pretrain", "lm"}
        else config.training.max_qa_train_samples
    )


def value(command, flag):
    return command[command.index(flag) + 1]


def config_paths(command):
    return command[command.index("--config") + 1 : command.index("--device")]


def external_checkpoint(tmp_path, experiment_id="unit-job", method="dynamic", stage="pretrain"):
    root = tmp_path / "source" / f"dynamic-pretrain-k64_{experiment_id}"
    output = root / stage
    (output / "checkpoints").mkdir(parents=True)
    checkpoint = output / "checkpoints" / "global_step_2"
    checkpoint.mkdir()
    (checkpoint / "state.pt").write_bytes(b"opaque external pretraining checkpoint fixture")
    (output / "run.json").write_text(
        json.dumps(
            {
                "config": {
                    "objective": {"method": method, "stage": stage},
                    "training": {
                        "experiment_id": experiment_id,
                        "experiment_dir": str(root),
                        "output_dir": str(output),
                    },
                }
            }
        )
    )
    return checkpoint


class Commands:
    def __init__(self, fail_at=None, omit_checkpoint=False, fail_stage=None):
        self.calls = []
        self.fail_at = fail_at
        self.omit_checkpoint = omit_checkpoint
        self.fail_stage = fail_stage

    def __call__(self, command, environment, log_path, append=False):
        call = {"command": command, "environment": dict(environment), "log_path": log_path}
        self.calls.append(call)
        with log_path.open("a" if append else "w") as log:
            log.write("orchestration test subprocess log\n")
        if self.fail_at == len(self.calls):
            raise subprocess.CalledProcessError(7, command)
        if "latent_working_memory.v3.train" in command:
            call["configs"], call["checkpoints"] = [], []
            for config_path in config_paths(command):
                config = load_experiment(config_path)
                if call["checkpoints"]:
                    config = replace(
                        config,
                        training=replace(
                            config.training, init_checkpoint=str(call["checkpoints"][-1])
                        ),
                    )
                call["configs"].append(config)
                output = Path(config.training.output_dir)
                (output / "checkpoints").mkdir(parents=True)
                (output / "config.json").write_text(json.dumps(config.to_dict()))
                steps = (
                    int(value(command, "--max-steps"))
                    if "--max-steps" in command
                    else 80
                )
                run = {
                    "config": config.to_dict(),
                    "step_offset": sum(
                        int(path.name.removeprefix("global_step_"))
                        for path in call["checkpoints"]
                    ),
                }
                (output / "run.json").write_text(json.dumps(run))
                checkpoint = output / "checkpoints" / f"global_step_{steps}"
                if not self.omit_checkpoint:
                    # 编排器读取身份；权重与优化器状态留空，不用于模型加载。
                    checkpoint.mkdir()
                    torch.save(
                        {
                            "run": run,
                            "trainable": {},
                            "optimizer": {},
                            "cursor": {"step": steps},
                            "rng": [],
                        },
                        checkpoint / "state.pt",
                    )
                    retained = sorted(
                        Path(config.training.experiment_dir).glob(
                            "*/checkpoints/global_step_*/state.pt"
                        ),
                        key=lambda path: (
                            json.loads((path.parents[2] / "run.json").read_text())["step_offset"]
                            + int(path.parent.name.removeprefix("global_step_"))
                        ),
                    )
                    for state in retained[: -config.training.save_total_limit]:
                        shutil.rmtree(state.parent)
                call["checkpoints"].append(checkpoint)
                if config.objective.stage == self.fail_stage:
                    raise subprocess.CalledProcessError(7, command)
                (output / "training-result.json").write_text(
                    json.dumps(
                        {
                            "complete": "--max-steps" not in command,
                            "completed_steps": steps,
                            "total_steps": 80,
                            "completed_epochs": 1 if steps == 80 else 0,
                            "sample_visits": steps
                            * config.training.global_batch_size(
                                len(environment["CUDA_VISIBLE_DEVICES"].split(","))
                            ),
                            "max_steps": steps if steps != 80 else None,
                            "checkpoint": str(checkpoint),
                        }
                    )
                )
        elif "latent_working_memory.v3.evaluate" in command:
            call["evaluation_config"] = json.loads(Path(value(command, "--config")).read_text())
            output = Path(value(command, "--output-dir"))
            output.mkdir(parents=True, exist_ok=True)
            checkpoint = Path(value(command, "--checkpoint")).resolve()
            run = json.loads((checkpoint.parent.parent / "run.json").read_text())
            objective = run["config"]["objective"]
            stage_step = int(checkpoint.name.removeprefix("global_step_"))
            summary = {
                "split": "test",
                "trajectories": 2,
                "method": objective["method"],
                "stage": objective["stage"],
                "dataset_signature": "orchestration-test-task-pool",
                "objective": objective,
                "qa_sampling": {"gate_signature": "orchestration-test-gate-pool"},
                "max_new_tokens": call["evaluation_config"]["max_new_tokens"],
                "metadata": {
                    "checkpoint": str(checkpoint),
                    "global_step": run["step_offset"] + stage_step,
                    "stage_step": stage_step,
                    "evaluation_config": call["evaluation_config"],
                },
            }
            (output / "summary.json").write_text(json.dumps(summary))
            gpu_job.EvaluationHistory(output, call["evaluation_config"]).record(summary)
        else:
            assert "latent_working_memory.v3.compare" in command
            Path(value(command, "--output-dir")).mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("smoke", (16, 4, 2, 2)),
        ("full", (None, None, None, None)),
    ],
)
def test_modes_apply_bounded_or_complete_training_and_evaluation(tmp_path, mode, expected):
    args = arguments(tmp_path, "--mode", mode)
    assert args.gpus == "0,1"
    assert args.max_new_tokens is None
    directory, level, jobs = job_plan.build_jobs(args)
    assert (
        tuple(
            level[name]
            for name in (
                "train_samples",
                "max_dev_samples",
                "max_steps",
                "max_test_samples",
            )
        )
        == expected
    )
    assert directory == args.output_root / "unit-job"
    for job in jobs:
        stage = job.config.objective.stage
        assert training_limit(job.config) == (
            expected[0] if mode != "full" else 12800 if stage == "pretrain" else None
        )
        assert job.config.training.max_dev_samples == expected[1]
        assert job.config.training.init_checkpoint == (
            str(args.checkpoint) if stage == "warmup" else None
        )
        assert job.config.training.swanlab_project == "latent-working-memory-v3"
        assert job.config.training.tags == (f"study:{'main' if mode == 'full' else mode}",)
        shared = job.config.objective.stage == "pretrain"
        assert job.config.training.micro_batch_size_per_gpu == 4
        assert job.config.training.gradient_accumulation_steps == 2
        assert job.config.training.global_batch_size(2) == 16
        assert job.config.training.global_batch_size(4) == 32
        assert job.config.model.memory_slots == 512
        assert job.config.objective.append_slots == (8 if shared else 32)
        assert job.config.training.experiment_id == "unit-job"
        assert job.config.training.group == "unit-job"
        root = Path(job.config.training.experiment_dir)
        assert Path(job.config.training.output_dir) == root / job.config.objective.stage
        capacity = "-k512" if shared else "-k512+32-local"
        assert root.name.endswith(capacity if mode == "full" else f"{capacity}_{mode}")
        assert job.config.training.experiment_name == f"{root.name}_unit-job"
        assert (
            job.config_path == directory / "plan" / root.name / f"{job.config.objective.stage}.json"
        )
        assert "global_batch_size" not in job.config.to_dict()["training"]
        command = gpu_job.training_command([job], 4, level["max_steps"])
        assert "--nproc_per_node=4" in command
        assert ("--max-steps" in command) == (mode != "full")
    evaluation_path = jobs[-1].config_path.parent / "eval.json"
    evaluation = gpu_job.evaluation_command(
        args, evaluation_path, Path("final.pt"), Path("evaluation")
    )
    assert value(evaluation, "--config") == str(evaluation_path)
    assert "--log-to-swanlab" in evaluation


@pytest.mark.parametrize("method", DYNAMIC_METHODS)
@pytest.mark.parametrize("mode", ["full", "smoke"])
def test_dynamic_capacity_names_use_cli_append_slots_in_every_mode(tmp_path, method, mode):
    args = arguments(tmp_path, "--method", method, "--mode", mode, "--append-slots", "16")

    directory, _, jobs = job_plan.build_jobs(args)

    suffix = "" if mode == "full" else f"_{mode}"
    for job in jobs:
        shared = job.method == "dynamic_pretrain"
        capacity = "k512" if shared else "k512+16-local"
        name = f"{job.method.replace('_', '-')}-{capacity}{suffix}"
        assert job.config.objective.append_slots == (8 if shared else 16)
        assert Path(job.config.training.experiment_dir) == directory / "train" / name
        assert job.config.training.output_dir == str(
            directory / "train" / name / job.config.objective.stage
        )
        assert job.config_path == directory / "plan" / name / f"{job.config.objective.stage}.json"
        assert job.config.training.experiment_name == f"{name}_unit-job"
        assert job.config.training.group == job.config.training.experiment_id == "unit-job"


def test_dynamic_stage_training_parameters_come_from_their_own_presets(tmp_path, monkeypatch):
    calls = []

    def preset(path):
        settings = {
            "dynamic_pretrain.json": (3, 1e-4),
            "memory_change.json": (4, 2e-4),
            "information_loss.json": (5, 3e-4),
        }
        configs = load_preset(path)
        calls.append(path.name)
        epochs, learning_rate = settings[path.name]
        return tuple(
            replace(
                config,
                training=replace(config.training, epochs=epochs, learning_rate=learning_rate),
            )
            for config in configs
        )

    monkeypatch.setattr(job_plan, "load_preset", preset)
    jobs = []
    for method in ("dynamic_pretrain", *DYNAMIC_METHODS):
        _, _, current = job_plan.build_jobs(arguments(tmp_path, "--method", method))
        jobs.extend(current)
    assert calls == ["dynamic_pretrain.json", "memory_change.json", "information_loss.json"]
    assert [job.config.training.epochs for job in jobs] == [3, 4, 4, 5, 5]
    assert [job.config.training.learning_rate for job in jobs] == [1e-4, 2e-4, 2e-4, 3e-4, 3e-4]


@pytest.mark.parametrize("method", ["autocompressors", *DYNAMIC_METHODS])
@pytest.mark.parametrize("steps", ["null", 1, 2, 7])
def test_bptt_cli_affects_only_ac_lm_and_dynamic_qa_and_null_restores_full_bptt(
    tmp_path, monkeypatch, method, steps
):
    def preset(path):
        return tuple(
            replace(config, objective=replace(config.objective, bptt_steps=3))
            if config.objective.method == "autocompressors"
            or config.objective.method in DYNAMIC_METHODS
            and config.objective.stage == "warmup"
            else config
            for config in load_preset(path)
        )

    monkeypatch.setattr(job_plan, "load_preset", preset)
    args = arguments(tmp_path, "--method", method, "--bptt-steps", str(steps))
    _, _, jobs = job_plan.build_jobs(args)
    for job in jobs:
        objective = job.config.objective
        uses_bptt = objective.method == "autocompressors" or (
            objective.method in DYNAMIC_METHODS and objective.stage in {"warmup", "policy"}
        )
        assert objective.bptt_steps == (steps if uses_bptt and steps != "null" else None)


def test_dynamic_qa_presets_default_to_full_bptt(tmp_path):
    for method in ("autocompressors", *DYNAMIC_METHODS):
        _, _, jobs = job_plan.build_jobs(arguments(tmp_path, "--method", method))
        for job in jobs:
            assert job.config.objective.bptt_steps == (2 if method == "autocompressors" else None)


@pytest.mark.parametrize("method", ["autocompressors", *DYNAMIC_METHODS])
@pytest.mark.parametrize("configured", [None, 3])
@pytest.mark.parametrize("override", [None, "null", 1])
def test_bptt_custom_config_is_preserved_unless_cli_overrides_it(
    tmp_path, method, configured, override
):
    path = custom_preset_file(tmp_path, method, objective={"bptt_steps": configured})
    options = () if override is None else ("--bptt-steps", str(override))
    _, _, jobs = job_plan.build_jobs(
        arguments(tmp_path, "--method", method, "--config", str(path), *options)
    )
    expected = configured if override is None else None if override == "null" else override
    for job in jobs:
        if job.method == "dynamic_pretrain":
            assert job.config.objective.bptt_steps is None
        else:
            assert job.config.objective.bptt_steps == expected


def test_full_plan_preserves_each_stage_training_budget(tmp_path):
    for method in (*METHODS, "dynamic_pretrain"):
        _, _, jobs = job_plan.build_jobs(arguments(tmp_path, "--method", method, "--mode", "full"))
        for job in jobs:
            assert training_limit(job.config) == (
                12800 if job.config.objective.stage == "pretrain" else None
            )


@pytest.mark.parametrize("category", ["pretrain", "qa"])
@pytest.mark.parametrize("limit", ["null", 7])
def test_training_limit_overrides_only_selected_data_category(tmp_path, category, limit):
    for method in METHODS:
        _, _, defaults = job_plan.build_jobs(
            arguments(tmp_path, "--method", method, "--mode", "full")
        )
        _, _, overridden = job_plan.build_jobs(
            arguments(
                tmp_path,
                "--method",
                method,
                "--mode",
                "full",
                f"--max-{category}-train-samples",
                str(limit),
            )
        )
        for before, after in zip(defaults, overridden, strict=True):
            stage_category = (
                "pretrain" if after.config.objective.stage in {"pretrain", "lm"} else "qa"
            )
            assert training_limit(after.config) == (
                (None if limit == "null" else limit)
                if stage_category == category
                else training_limit(before.config)
            )


@pytest.mark.parametrize("mode,profile_limit", [("smoke", 16)])
@pytest.mark.parametrize("limit", ["null", 7, 3000])
def test_trial_profiles_cap_shared_qa_training_budget(tmp_path, mode, profile_limit, limit):
    _, _, jobs = job_plan.build_jobs(
        arguments(tmp_path, "--mode", mode, "--max-qa-train-samples", str(limit))
    )
    warmup, policy = jobs
    assert (
        training_limit(warmup.config)
        == training_limit(policy.config)
        == (profile_limit if limit == "null" else min(limit, profile_limit))
    )


def test_null_input_limit_removes_upper_bound_only_for_ae_lm(tmp_path):
    for method in (*METHODS, "dynamic_pretrain"):
        _, _, defaults = job_plan.build_jobs(arguments(tmp_path, "--method", method))
        _, _, overridden = job_plan.build_jobs(
            arguments(tmp_path, "--method", method, "--max-pretrain-data-tokens", "null")
        )
        for before, after in zip(defaults, overridden, strict=True):
            assert after.config.training.max_pretrain_data_tokens == (
                None
                if after.config.objective.stage == "pretrain"
                else before.config.training.max_pretrain_data_tokens
            )
            assert (
                after.config.training.max_qa_data_tokens
                == before.config.training.max_qa_data_tokens
            )


@pytest.mark.parametrize(
    "method", ["icae_single", "icae_multi", "autocompressors", *DYNAMIC_METHODS]
)
@pytest.mark.parametrize("limit", ["null", 4096])
def test_factqa_input_limit_is_saved_without_changing_pretraining_limit(tmp_path, method, limit):
    _, _, defaults = job_plan.build_jobs(arguments(tmp_path, "--method", method))
    _, _, overridden = job_plan.build_jobs(
        arguments(tmp_path, "--method", method, "--max-qa-data-tokens", str(limit))
    )
    for before, after in zip(defaults, overridden, strict=True):
        assert after.config.training.max_qa_data_tokens == (None if limit == "null" else limit)
        assert (
            after.config.training.max_pretrain_data_tokens
            == before.config.training.max_pretrain_data_tokens
        )


def test_removed_shared_training_limit_is_rejected():
    with pytest.raises(SystemExit):
        gpu_job.parse_args(["--train-samples", "10"])


def test_default_pretraining_data_is_the_shared_multisegment_root():
    args = gpu_job.parse_args(["--method", "dynamic_pretrain"])
    assert args.config is None
    assert args.pretrain_data == Path("data/fineweb-multisegment-k512-seg1to3x_train32k_20261008")


@pytest.mark.parametrize("options", [[], ["--method", "all"]])
def test_custom_config_requires_an_explicit_single_method(options):
    with pytest.raises(SystemExit):
        gpu_job.parse_args(["--config", "custom experiment.json", *options])


@pytest.mark.parametrize(
    "options",
    [[], ["--config", "custom.json"], ["--checkpoint", "parent.pt"], ["--bptt-steps", "2"]],
)
def test_removed_dynamic_method_is_rejected(options):
    with pytest.raises(SystemExit):
        gpu_job.parse_args(["--method", "dynamic", *options])


@pytest.mark.parametrize(
    "method", ["icae_single", "icae_multi", "autocompressors", "memory_change", "information_loss"]
)
@pytest.mark.parametrize("override", [False, True])
def test_custom_method_preset_preserves_parameters_and_launcher_overrides(
    tmp_path, method, override
):
    model = (
        {"gradient_checkpointing": False}
        if method in DYNAMIC_METHODS
        else {"model_name_or_path": "models/custom model", "lora_rank": 16}
    )
    path = custom_preset_file(
        tmp_path,
        method,
        model=model,
        objective={
            "qa_batch_size": 5,
            "task_qa_count": 6,
            "gate_qa_count": 4,
            "task_new_ratio": 0.5,
        },
        training={
            "epochs": 3,
            "learning_rate": 2e-4,
            "micro_batch_size_per_gpu": 3,
            "gradient_accumulation_steps": 5,
            "dataset_dir": "ignored/preset-data",
            "max_pretrain_train_samples": 999,
            "max_qa_train_samples": 999,
            "max_dev_samples": 888,
            "dev_every": 17,
            "save_every": 19,
            "swanlab_project": "ignored-project",
            "group": "ignored-group",
            "tags": ["study:ignored"],
        },
    )
    options = (
        [
            "--epochs",
            "4",
            "--micro-batch-size-per-gpu",
            "2",
            "--gradient-accumulation-steps",
            "3",
            "--qa-batch-size",
            "7",
            "--task-qa-count",
            "4",
            "--gate-qa-count",
            "2",
            "--task-new-ratio",
            "0.75",
            "--model-path",
            "models/CLI model",
        ]
        if override
        else []
    )
    args = arguments(tmp_path, "--method", method, "--config", str(path), *options)

    directory, level, jobs = job_plan.build_jobs(args)

    assert args.config == path
    assert directory == args.output_root / "unit-job"
    assert level["max_steps"] == 2
    own_jobs = [job for job in jobs if job.method == method]
    assert len(own_jobs) == (1 if method == "autocompressors" else 2)
    for job in own_jobs:
        config = job.config
        preset_model = (
            str(Path.home() / "models/Qwen3-4B-Instruct-2507")
            if method in DYNAMIC_METHODS
            else "models/custom model"
        )
        assert config.model.model_name_or_path == ("models/CLI model" if override else preset_model)
        assert config.model.lora_rank == (128 if method in DYNAMIC_METHODS else 16)
        assert config.objective.qa_batch_size == (7 if override else 5)
        assert config.objective.task_qa_count == (4 if override else 6)
        assert config.objective.gate_qa_count == (2 if override else 4)
        assert config.objective.task_new_ratio == (0.75 if override else 0.5)
        assert config.training.epochs == (4 if override else 3)
        assert config.training.learning_rate == 2e-4
        assert config.training.micro_batch_size_per_gpu == (2 if override else 3)
        assert config.training.gradient_accumulation_steps == (3 if override else 5)
        assert training_limit(config) == 16
        assert config.training.max_dev_samples == 4
        assert config.training.dev_every == config.training.save_every == 1
        assert config.training.swanlab_project == "latent-working-memory-v3"
        assert config.training.group == config.training.experiment_id == "unit-job"
        assert config.training.tags == ("study:smoke",)
        dataset = args.pretrain_data if config.objective.stage == "pretrain" else args.qa_data
        assert Path(config.training.dataset_dir) == dataset
        capacity = "k512+32-local" if method in DYNAMIC_METHODS else "k512"
        run_name = f"{method.replace('_', '-')}-{capacity}_smoke"
        assert Path(config.training.experiment_dir) == directory / "train" / run_name
        assert config.training.experiment_name == f"{run_name}_unit-job"
        assert job.config_path == directory / "plan" / run_name / f"{config.objective.stage}.json"


@pytest.mark.parametrize(
    "method,stage,evaluate",
    [
        ("icae_single", "pretrain", False),
        ("icae_multi", "qa", True),
        ("autocompressors", "pretrain", True),
        ("memory_change", "warmup", False),
        ("information_loss", "policy", True),
        ("memory_change", "pretrain", False),
        ("information_loss", "pretrain", False),
    ],
)
def test_custom_method_preset_runs_only_its_declared_stages(tmp_path, method, stage, evaluate):
    path = custom_preset_file(tmp_path, method, objective={"stages": [stage]})
    args = arguments(tmp_path, "--method", method, "--config", str(path))

    _, _, jobs = job_plan.build_jobs(args)

    own_jobs = [job for job in jobs if job.method == method]
    assert len(own_jobs) == 1
    assert own_jobs[0].config.objective.stage == stage
    assert own_jobs[0].evaluate == evaluate
    assert own_jobs[0].initialize_from is None
    assert [job.method for job in jobs] == [method]


@pytest.mark.parametrize("method", DYNAMIC_METHODS)
def test_custom_dynamic_posttraining_uses_only_its_requested_preset(tmp_path, method):
    path = custom_preset_file(
        tmp_path,
        method,
        model={"gradient_checkpointing": False},
        objective={"append_slots": 16, "qa_batch_size": 5},
        training={"epochs": 3, "learning_rate": 2e-4},
    )
    args = arguments(tmp_path, "--method", method, "--config", str(path))
    _, _, jobs = job_plan.build_jobs(args)
    assert [job.config.objective.stage for job in jobs] == ["warmup", "policy"]
    assert jobs[0].config.training.init_checkpoint == str(args.checkpoint)
    assert jobs[1].initialize_from == jobs[0].key
    assert all(
        job.config.training.learning_rate == 2e-4 and job.config.training.epochs == 3
        for job in jobs
    )
    assert all(not job.config.model.gradient_checkpointing for job in jobs)
    assert all(job.config.objective.append_slots == 16 for job in jobs)
    assert all(
        Path(job.config.training.experiment_dir).name
        == f"{method.replace('_', '-')}-k512+16-local_smoke"
        for job in jobs
    )


@pytest.mark.parametrize("method", DYNAMIC_METHODS)
@pytest.mark.parametrize("configured", [False, True])
def test_dynamic_posttraining_requires_explicit_cli_checkpoint(
    tmp_path, monkeypatch, method, configured
):
    source = external_checkpoint(tmp_path)
    path = custom_preset_file(
        tmp_path, method, training={"init_checkpoint": str(source) if configured else None}
    )
    args = arguments(tmp_path, "--method", method, "--config", str(path), checkpoint=False)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    with pytest.raises(ValueError, match="checkpoint"):
        gpu_job.run_job(args)
    assert not commands.calls
    assert not args.output_root.exists()


@pytest.mark.parametrize("method", DYNAMIC_METHODS)
@pytest.mark.parametrize("writer_mode", ["local", "tag", "mask", "dual_lora"])
@pytest.mark.parametrize("supplied", [False, True])
def test_full_dynamic_pipeline_uses_its_config_and_ignores_initialization(
    tmp_path, capsys, method, writer_mode, supplied
):
    path = custom_preset_file(
        tmp_path,
        method,
        model={"model_name_or_path": "models/custom model"},
        objective={
            "stages": ["pretrain", "warmup", "policy"],
            "writer_mode": writer_mode,
            "bptt_steps": 2,
        },
        training={"init_checkpoint": "missing/configured checkpoint"},
    )
    args = arguments(
        tmp_path,
        "--method",
        method,
        "--config",
        str(path),
        "--model-path",
        "models/CLI model",
        *(["--checkpoint", "missing/CLI checkpoint"] if supplied else []),
        run_id=None,
        checkpoint=False,
    )
    directory, _, jobs = job_plan.build_jobs(args)
    assert re.fullmatch(r"\d{8}-\d{6}", args.run_id)
    assert [job.config.objective.stage for job in jobs] == ["pretrain", "warmup", "policy"]
    assert all(job.method == method for job in jobs)
    assert jobs[0].config.training.init_checkpoint is None
    assert jobs[0].initialize_from is None
    assert jobs[1].initialize_from == jobs[0].key
    assert jobs[2].initialize_from == jobs[1].key
    assert jobs[0].config.objective.writer_mode == "local"
    assert jobs[0].config.objective.bptt_steps is None
    assert all(
        job.config.objective.writer_mode == writer_mode and job.config.objective.bptt_steps == 2
        for job in jobs[1:]
    )
    assert len({job.config.training.experiment_dir for job in jobs}) == 1
    assert len({job.config.training.experiment_name for job in jobs}) == 1
    assert all(job.config.model.model_name_or_path == "models/CLI model" for job in jobs)
    assert len({job.config.model for job in jobs}) == 1
    assert all(
        job.config_path.parent == directory / "plan" / Path(job.config.training.experiment_dir).name
        for job in jobs
    )
    output = capsys.readouterr().out
    assert output.count("[ignored]") == int(supplied)
    assert args.checkpoint is None


@pytest.mark.parametrize("supplied", [False, True])
def test_custom_dynamic_pretraining_uses_its_requested_file(tmp_path, capsys, supplied):
    path = custom_preset_file(
        tmp_path,
        "dynamic_pretrain",
        model={"memory_slots": 64, "lora_rank": 16},
        objective={"method": "dynamic", "append_slots": 16},
        training={"epochs": 3, "learning_rate": 2e-4},
    )
    args = arguments(
        tmp_path,
        "--method",
        "dynamic_pretrain",
        "--config",
        str(path),
        *(["--checkpoint", "missing/checkpoint"] if supplied else []),
    )

    directory, _, jobs = job_plan.build_jobs(args)

    assert len(jobs) == 1
    job = jobs[0]
    assert job.key == "dynamic-pretrain" and job.method == "dynamic_pretrain"
    assert job.config.objective.method == "dynamic"
    assert job.config.objective.stage == "pretrain"
    assert job.config.model.memory_slots == 64 and job.config.model.lora_rank == 16
    assert job.config.training.epochs == 3 and job.config.training.learning_rate == 2e-4
    assert job.config.objective.append_slots == 16
    assert job.config_path == directory / "plan/dynamic-pretrain-k64_smoke/pretrain.json"
    assert job.config.training.experiment_name == "dynamic-pretrain-k64_smoke_unit-job"
    assert job.initialize_from is None and not job.evaluate
    assert args.checkpoint is None
    assert capsys.readouterr().out.count("[ignored]") == int(supplied)


def test_saved_autocompressors_lm_preset_keeps_its_stage_and_uses_pretrain_budget(tmp_path):
    raw = json.loads(Path("configs/v3/autocompressors.json").read_text())
    raw["objective"]["stages"] = ["lm"]
    raw["training"]["max_pretrain_train_samples"] = 12800
    path = tmp_path / "original-autocompressors.json"
    path.write_text(json.dumps(raw))
    args = arguments(
        tmp_path,
        "--method",
        "autocompressors",
        "--config",
        str(path),
        "--mode",
        "full",
        "--max-pretrain-train-samples",
        "7",
    )

    _, _, jobs = job_plan.build_jobs(args)

    (job,) = jobs
    assert job.config.objective.stage == "lm"
    assert training_limit(job.config) == 7
    assert job.config_path.name == "lm.json"
    assert Path(job.config.training.output_dir).name == "lm"
    assert job.evaluate


@pytest.mark.parametrize(
    "method,source_method",
    [
        ("icae_single", "icae_multi"),
        ("icae_multi", "icae_single"),
        ("autocompressors", "icae_single"),
        ("memory_change", "information_loss"),
        ("information_loss", "memory_change"),
        ("dynamic_pretrain", "icae_single"),
    ],
)
def test_custom_preset_method_conflicts_fail_before_execution(
    tmp_path, monkeypatch, method, source_method
):
    path = custom_preset_file(
        tmp_path,
        source_method,
        objective={"stages": ["pretrain"]} if method == "dynamic_pretrain" else {},
    )
    args = arguments(tmp_path, "--method", method, "--config", str(path))
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    with pytest.raises(ValueError):
        gpu_job.run_job(args)

    assert not commands.calls
    assert not args.output_root.exists()


@pytest.mark.parametrize("stages", [["warmup"], ["policy"], ["warmup", "policy"]])
def test_custom_dynamic_pretraining_rejects_posttraining_stages(tmp_path, stages):
    path = custom_preset_file(tmp_path, "memory_change", objective={"stages": stages})
    args = arguments(tmp_path, "--method", "dynamic_pretrain", "--config", str(path))

    with pytest.raises(ValueError, match="pretrain"):
        gpu_job.run_job(args)

    assert not args.output_root.exists()


@pytest.mark.parametrize("invalid", ["schema", "runtime-stage", "stage-order", "parameter", "json"])
def test_invalid_custom_presets_fail_before_execution(tmp_path, monkeypatch, invalid):
    path = custom_preset_file(tmp_path, "icae_single")
    raw = json.loads(path.read_text())
    if invalid == "schema":
        del raw["training"]
    elif invalid == "runtime-stage":
        raw["objective"].pop("stages")
        raw["objective"]["stage"] = "pretrain"
    elif invalid == "stage-order":
        raw["objective"]["stages"] = ["qa", "pretrain"]
    elif invalid == "parameter":
        raw["training"]["epochs"] = 0
    path.write_text("{" if invalid == "json" else json.dumps(raw))
    args = arguments(tmp_path, "--method", "icae_single", "--config", str(path))
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    with pytest.raises(ValueError):
        gpu_job.run_job(args)

    assert not commands.calls
    assert not args.output_root.exists()


@pytest.mark.parametrize("method", ["icae_single", "dynamic_pretrain"])
def test_missing_custom_preset_fails_without_falling_back_to_defaults(
    tmp_path, monkeypatch, method
):
    path = tmp_path / "missing preset.json"
    args = arguments(
        tmp_path,
        "--method",
        method,
        "--config",
        str(path),
    )
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    with pytest.raises(FileNotFoundError):
        gpu_job.run_job(args)

    assert not commands.calls
    assert not args.output_root.exists()


def test_custom_preset_execution_records_resolved_configs_and_evaluates_final_stage(
    tmp_path, monkeypatch
):
    path = custom_preset_file(
        tmp_path, "icae_single", objective={"qa_batch_size": 5}, training={"learning_rate": 2e-4}
    )
    original = path.read_bytes()
    args = arguments(
        tmp_path,
        "--method",
        "icae_single",
        "--config",
        str(path),
        "--mode",
        "smoke",
        "--save-total-limit",
        "1",
    )
    prepare_dataset_entries(args)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    result = gpu_job.run_job(args)

    assert result["status"] == "finished"
    assert set(result["checkpoints"]) == {"icae-single-qa"}
    training, evaluation = commands.calls
    assert [config.objective.stage for config in training["configs"]] == ["pretrain", "qa"]
    assert value(training["command"], "--max-steps") == "2"
    assert training["configs"][1].training.init_checkpoint == str(training["checkpoints"][0])
    assert value(evaluation["command"], "--checkpoint") == str(training["checkpoints"][-1])
    plan_directory = args.output_root / "unit-job/plan/icae-single-k512_smoke"
    plan = json.loads((plan_directory / "job.json").read_text())
    assert plan["method"] == "icae_single"
    assert plan["training_runs"][0]["command"] == training["command"]
    assert [job["initialize_from"] for job in plan["jobs"]] == [None, "icae-single-pretrain"]
    assert all(job["config"]["training"]["init_checkpoint"] is None for job in plan["jobs"])
    for job, trained in zip(plan["jobs"], training["configs"], strict=True):
        config = load_experiment(job["config_path"])
        assert config == trained
        assert job["config"]["training"]["learning_rate"] == 2e-4
        assert job["config"]["objective"]["qa_batch_size"] == 5
        assert config.training.learning_rate == 2e-4
        assert config.objective.qa_batch_size == 5
        assert training_limit(config) == 16
        assert config.training.max_dev_samples == 4
        assert config.training.dev_every == config.training.save_every == 1
    assert json.loads((plan_directory / "result.json").read_text()) == result
    assert path.read_bytes() == original


@pytest.mark.parametrize("mode", ["smoke", "full"])
@pytest.mark.parametrize(
    "method", ["icae_single", "icae_multi", "autocompressors", "dynamic_pretrain"]
)
def test_minimal_command_builds_one_method_with_preset_batch_settings(mode, method):
    with pytest.raises(SystemExit):
        gpu_job.parse_args(["--mode", mode])
    args = gpu_job.parse_args(["--method", method, "--mode", mode, "--gpus", "4,5"])
    assert args.method == method and args.run_id is None
    for action in ("--resume", "--evaluate-only"):
        restored = gpu_job.parse_args([action, "--checkpoint", "existing/checkpoint"])
        assert restored.method is None
        assert restored.checkpoint == (
            [Path("existing/checkpoint")]
            if action == "--evaluate-only"
            else Path("existing/checkpoint")
        )
    directory, _, jobs = job_plan.build_jobs(args)
    assert re.fullmatch(r"\d{8}-\d{6}", args.run_id)
    assert directory == args.output_root.resolve() / args.run_id
    assert {job.method for job in jobs} == {method}
    assert len(jobs) == (2 if method in {"icae_single", "icae_multi"} else 1)
    for job in jobs:
        assert job.config.training.group == args.run_id
        assert job.config.model.model_name_or_path == str(
            Path.home() / "models/Qwen3-4B-Instruct-2507"
        )
        assert job.config.training.micro_batch_size_per_gpu == 4
        assert job.config.training.gradient_accumulation_steps == 2
        assert job.config.training.global_batch_size(2) == 16
        assert job.config.objective.qa_batch_size == 8


@pytest.mark.parametrize("override", [False, True])
def test_cli_only_overrides_explicitly_supplied_preset_values(tmp_path, monkeypatch, override):
    def preset(path):
        return tuple(
            replace(
                config,
                model=replace(config.model, model_name_or_path="local/preset-model"),
                objective=replace(config.objective, qa_batch_size=5, append_slots=12),
                training=replace(
                    config.training,
                    micro_batch_size_per_gpu=3,
                    gradient_accumulation_steps=5,
                    seed=71,
                ),
            )
            for config in load_preset(path)
        )

    monkeypatch.setattr(job_plan, "load_preset", preset)
    options = (
        [
            "--micro-batch-size-per-gpu",
            "4",
            "--model-path",
            "~/models/override-model",
            "--append-slots",
            "16",
            "--seed",
            "0",
        ]
        if override
        else []
    )
    args = arguments(tmp_path, "--mode", "full", *options)
    _, level, jobs = job_plan.build_jobs(args)
    assert args.method == "memory_change"
    for job in jobs:
        assert job.config.training.micro_batch_size_per_gpu == (4 if override else 3)
        assert job.config.training.gradient_accumulation_steps == 5
        assert job.config.training.seed == (0 if override else 71)
        assert job.config.objective.qa_batch_size == 5
        assert job.config.objective.append_slots == (16 if override else 12)
        name = f"memory-change-k512+{16 if override else 12}-local"
        assert Path(job.config.training.experiment_dir).name == name
        assert job.config.training.experiment_name == f"{name}_{args.run_id}"
        assert job.config.model.model_name_or_path == (
            str(Path.home() / "models/override-model") if override else "local/preset-model"
        )
        evaluation_config = gpu_job.resolve_evaluation_config(
            args, job.config_path.parent / "eval.json", level
        )
        assert evaluation_config.seed == (args.seed if args.seed is not None else 20261004)


@pytest.mark.parametrize("method", DYNAMIC_METHODS)
def test_auto_from_external_pretraining_checkpoint_skips_pretraining_in_plan(tmp_path, method):
    checkpoint = external_checkpoint(tmp_path)
    args = arguments(tmp_path, "--method", method, "--checkpoint", str(checkpoint))

    _, _, jobs = job_plan.build_jobs(args)

    assert len(jobs) == 2
    warmup, policy = jobs
    prefix = method.replace("_", "-")
    assert warmup.key == f"{prefix}-warmup"
    assert policy.key == f"{prefix}-policy"
    assert warmup.config.objective.method == policy.config.objective.method == method
    assert warmup.config.objective.stage == "warmup"
    assert policy.config.objective.stage == "policy"
    assert warmup.config.training.init_checkpoint == str(checkpoint.resolve())
    assert warmup.initialize_from is None
    assert policy.config.training.init_checkpoint is None
    assert policy.initialize_from == warmup.key
    assert not warmup.evaluate and policy.evaluate
    assert Path(warmup.config.training.dataset_dir) == args.qa_data
    assert Path(policy.config.training.dataset_dir) == args.qa_data
    assert not args.output_root.exists()


@pytest.mark.parametrize("method", ["icae_single", "icae_multi"])
def test_icae_pretrain_only_preset_needs_no_qa_or_final_evaluation(tmp_path, monkeypatch, method):
    install_preset_stages(tmp_path, monkeypatch, f"{method}.json", ["pretrain"])
    args = arguments(tmp_path, "--method", method)
    args.pretrain_data.mkdir(parents=True)
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "preparation.json"):
        (args.pretrain_data / name).write_text("")
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    _, _, jobs = job_plan.build_jobs(args)
    result = gpu_job.run_job(args)

    assert len(jobs) == 1
    assert jobs[0].config.objective.stage == "pretrain"
    assert jobs[0].initialize_from is None and not jobs[0].evaluate
    assert len(commands.calls) == 1
    assert [config.objective.stage for config in commands.calls[0]["configs"]] == ["pretrain"]
    assert set(result["checkpoints"]) == {f"{method.replace('_', '-')}-pretrain"}
    assert not result["summaries"]
    assert not args.qa_data.exists()
    assert not (args.output_root / "unit-job" / "eval").exists()


def test_icae_qa_only_preset_uses_its_configured_initialization(tmp_path, monkeypatch):
    source = external_checkpoint(tmp_path, method="icae_single")
    install_preset_stages(tmp_path, monkeypatch, "icae_single.json", ["qa"], source)
    args = arguments(tmp_path, "--method", "icae_single")
    args.qa_data.mkdir()
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "preparation.json"):
        (args.qa_data / name).write_text("")
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    result = gpu_job.run_job(args)

    training, evaluation = commands.calls
    assert len(training["configs"]) == 1
    config = training["configs"][0]
    assert config.objective.stage == "qa"
    assert config.training.init_checkpoint == str(source)
    assert value(evaluation["command"], "--checkpoint") == str(training["checkpoints"][0])
    assert set(result["checkpoints"]) == {"icae-single-qa"}
    assert len(result["summaries"]) == 1
    assert not args.pretrain_data.exists()


@pytest.mark.parametrize("method", DYNAMIC_METHODS)
@pytest.mark.parametrize("stage", ["warmup", "policy"])
@pytest.mark.parametrize("external", [False, True])
def test_dynamic_single_stage_preset_executes_only_declared_stage(
    tmp_path, monkeypatch, method, stage, external
):
    install_preset_stages(tmp_path, monkeypatch, f"{method}.json", [stage])
    source = external_checkpoint(tmp_path) if external else None
    args = arguments(
        tmp_path,
        "--method",
        method,
        *(["--checkpoint", str(source)] if external else []),
        checkpoint=False,
    )
    prepare_dataset_entries(args)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    if not external:
        with pytest.raises(ValueError, match="checkpoint"):
            gpu_job.run_job(args)
        assert not commands.calls and not args.output_root.exists()
        return
    _, _, jobs = job_plan.build_jobs(args)
    result = gpu_job.run_job(args)
    assert len(jobs) == 1 and jobs[0].config.objective.stage == stage
    assert jobs[0].initialize_from is None
    assert jobs[0].config.training.init_checkpoint == str(source)
    training = commands.calls[0]
    assert [config.objective.stage for config in training["configs"]] == [stage]
    assert len(commands.calls) == 1 + int(stage == "policy")
    assert len(result["summaries"]) == int(stage == "policy")
    assert set(result["checkpoints"]) == {f"{method.replace('_', '-')}-{stage}"}


def test_dynamic_cli_initialization_inherits_source_id_without_repeating_pretraining(
    tmp_path, monkeypatch
):
    source = external_checkpoint(tmp_path, experiment_id="configured-source")
    install_preset_stages(tmp_path, monkeypatch, "memory_change.json", ["warmup", "policy"], source)
    args = arguments(
        tmp_path, "--method", "memory_change", "--checkpoint", str(source), run_id=None
    )
    args.qa_data.mkdir()
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "preparation.json"):
        (args.qa_data / name).write_text("")
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    result = gpu_job.run_job(args)
    assert args.run_id == result["experiment_id"] == "configured-source"
    assert len(commands.calls) == 2
    assert [config.objective.stage for config in commands.calls[0]["configs"]] == [
        "warmup",
        "policy",
    ]
    assert commands.calls[0]["configs"][0].training.init_checkpoint == str(source)
    assert set(result["checkpoints"]) == {"memory-change-warmup", "memory-change-policy"}
    assert not args.pretrain_data.exists()
    assert {path.name for path in (args.output_root / "configured-source" / "plan").iterdir()} == {
        "memory-change-k512+32-local_smoke"
    }


@pytest.mark.parametrize(
    "method,stages",
    [
        ("icae_single", ["qa", "pretrain"]),
        ("icae_multi", ["pretrain", "pretrain"]),
        ("memory_change", ["policy", "warmup"]),
        ("information_loss", ["warmup", "warmup"]),
    ],
)
def test_invalid_preset_stage_order_or_duplicates_stop_before_execution(
    tmp_path, monkeypatch, method, stages
):
    install_preset_stages(tmp_path, monkeypatch, f"{method}.json", stages)
    args = arguments(tmp_path, "--method", method)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    with pytest.raises(ValueError, match="stage"):
        gpu_job.run_job(args)

    assert not commands.calls
    assert not args.output_root.exists()


@pytest.mark.parametrize("stages", [["warmup"], ["policy"], ["pretrain", "warmup", "policy"]])
def test_standalone_dynamic_pretraining_rejects_posttraining_stages(tmp_path, monkeypatch, stages):
    install_preset_stages(tmp_path, monkeypatch, "dynamic_pretrain.json", stages)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    with pytest.raises(ValueError, match="pretrain|stage"):
        gpu_job.run_job(arguments(tmp_path, "--method", "dynamic_pretrain"))
    assert not commands.calls
    assert not (tmp_path / "outputs").exists()


def test_pretraining_sampling_overrides_do_not_change_qa_or_autocompressors_tasks(tmp_path):
    for method in (*METHODS, "dynamic_pretrain"):
        args = arguments(
            tmp_path,
            "--method",
            method,
            "--lm-ratio",
            "0.75",
            "--max-pretrain-data-tokens",
            "2048",
            "--lm-target-tokens",
            "256",
        )
        _, _, jobs = job_plan.build_jobs(args)
        for job in jobs:
            stage = job.config.objective.stage
            assert job.config.training.lm_ratio == (
                0.75 if stage == "pretrain" and method != "autocompressors" else 0.5
            )
            assert job.config.training.max_pretrain_data_tokens == (
                2048 if stage == "pretrain" else None if stage in {"warmup", "policy"} else 12288
            )
            assert job.config.training.lm_target_tokens == (256 if stage == "pretrain" else 512)


@pytest.mark.parametrize(
    "method",
    ["icae_multi", "icae_single", "memory_change", "information_loss", "autocompressors"],
)
def test_icae_segment_count_overrides_change_both_multi_stages(tmp_path, method):
    _, _, defaults = job_plan.build_jobs(arguments(tmp_path, "--method", method))
    _, _, overridden = job_plan.build_jobs(
        arguments(
            tmp_path, "--method", method, "--icae-min-segments", "2", "--icae-max-segments", "8"
        )
    )
    for before, after in zip(defaults, overridden, strict=True):
        multi = after.config.objective.method == "icae_multi"
        expected = (
            replace(before.config.objective, icae_min_segments=2, icae_max_segments=8)
            if multi
            else before.config.objective
        )
        assert after.config.objective == expected
        assert after.config.training == before.config.training
        assert after.config.model == before.config.model


def test_icae_segment_count_defaults_preserve_the_preset(tmp_path, monkeypatch):
    def preset(path):
        return tuple(
            replace(
                config,
                objective=replace(config.objective, icae_min_segments=2, icae_max_segments=8),
            )
            for config in load_preset(path)
        )

    monkeypatch.setattr(job_plan, "load_preset", preset)
    _, _, jobs = job_plan.build_jobs(arguments(tmp_path, "--method", "icae_multi"))
    assert all(
        (job.config.objective.icae_min_segments, job.config.objective.icae_max_segments) == (2, 8)
        for job in jobs
    )


@pytest.mark.parametrize("method", ["autocompressors"])
def test_autocompressors_overrides_apply_only_to_autocompressors(tmp_path, method):
    _, _, defaults = job_plan.build_jobs(arguments(tmp_path, "--method", method))
    _, _, overridden = job_plan.build_jobs(
        arguments(tmp_path, "--method", method, "--ac-num-segments", "5")
    )
    for before, after in zip(defaults, overridden, strict=True):
        expected = (
            replace(before.config.objective, ac_num_segments=5)
            if after.config.objective.method == "autocompressors"
            else before.config.objective
        )
        assert after.config.objective == expected
        assert after.config.training == before.config.training
        assert after.config.model == before.config.model


def test_autocompressors_defaults_preserve_the_preset(tmp_path, monkeypatch):
    def preset(path):
        return tuple(
            replace(config, objective=replace(config.objective, ac_num_segments=5, bptt_steps=3))
            for config in load_preset(path)
        )

    monkeypatch.setattr(job_plan, "load_preset", preset)
    _, _, jobs = job_plan.build_jobs(arguments(tmp_path, "--method", "autocompressors"))
    assert len(jobs) == 1
    assert (jobs[0].config.objective.ac_num_segments, jobs[0].config.objective.bptt_steps) == (
        5,
        3,
    )


def test_explicit_overrides_and_null_remove_profile_limits(tmp_path):
    checkpoint = external_checkpoint(tmp_path)
    args = arguments(
        tmp_path,
        "--method",
        "memory_change",
        "--mode",
        "full",
        "--checkpoint",
        str(checkpoint),
        "--max-qa-train-samples",
        "null",
        "--max-dev-samples",
        "6",
        "--max-steps",
        "null",
        "--max-test-samples",
        "null",
        "--epochs",
        "3",
        "--micro-batch-size-per-gpu",
        "2",
        "--gradient-accumulation-steps",
        "3",
        "--qa-batch-size",
        "2",
        "--threshold-i",
        "0.25",
        "--dev-every",
        "2",
        "--save-every",
        "3",
        "--save-total-limit",
        "4",
        "--tracking",
        "disabled",
        "--group",
        "chosen-group",
        "--model-path",
        "models/local Qwen3",
    )
    _, level, jobs = job_plan.build_jobs(args)
    assert len(jobs) == 2
    warmup, job = jobs
    assert warmup.config.training.init_checkpoint == str(checkpoint.resolve())
    assert job.initialize_from == warmup.key and job.evaluate
    assert job.config.training.init_checkpoint is None
    assert training_limit(job.config) is None
    assert job.config.training.max_dev_samples == 6
    assert job.config.training.epochs == 3
    assert job.config.training.micro_batch_size_per_gpu == 2
    assert job.config.training.gradient_accumulation_steps == 3
    assert job.config.training.global_batch_size(2) == 12
    assert all(item.config.training.save_total_limit == 4 for item in jobs)
    assert job.config.training.group == "chosen-group"
    assert job.config.training.swanlab_project is None
    assert job.config.objective.qa_batch_size == 2
    assert job.config.objective.threshold_i == 0.25
    assert job.config.model.model_name_or_path == "models/local Qwen3"
    assert level["max_steps"] is None
    evaluation_config = gpu_job.resolve_evaluation_config(
        args, job.config_path.parent / "eval.json", level
    )
    assert evaluation_config.max_test_samples is None
    assert level["dev_every"] == 2 and level["save_every"] == 3
    assert "--log-to-swanlab" not in gpu_job.evaluation_command(
        args, job.config_path.parent / "eval.json", Path("a"), Path("b")
    )


def test_plan_derives_each_global_batch_from_selected_gpus_microbatch_and_accumulation(tmp_path):
    args = arguments(
        tmp_path,
        "--method",
        "icae_single",
        "--gpus",
        "4,5,6,7",
        "--micro-batch-size-per-gpu",
        "2",
        "--gradient-accumulation-steps",
        "3",
    )
    _, _, jobs = job_plan.build_jobs(args)
    assert len(jobs) == 2
    assert all(
        job.config.training.micro_batch_size_per_gpu == 2
        and job.config.training.gradient_accumulation_steps == 3
        and job.config.training.global_batch_size(4) == 24
        for job in jobs
    )
    assert all("global_batch_size" not in job.config.to_dict()["training"] for job in jobs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "options",
    [
        ["--gpus", "-1"],
        ["--gpus", "0,0"],
        ["--gpus", "4,4"],
        ["--gpus", "4,other"],
        ["--gpus", "4,04"],
        ["--gpus", "4,"],
        ["--gpus", ""],
        ["--max-pretrain-train-samples", "-1"],
        ["--lm-train-samples", "10"],
        ["--epochs", "0"],
        ["--micro-batch-size-per-gpu", "0"],
        ["--micro-batch-size-per-gpu", "-1"],
        ["--gradient-accumulation-steps", "0"],
        ["--gradient-accumulation-steps", "-1"],
        ["--append-slots", "0"],
        ["--append-slots", "-1"],
        ["--lm-ratio", "-0.1"],
        ["--lm-ratio", "1.1"],
        ["--lm-ratio", "nan"],
        ["--lm-ratio", "inf"],
        ["--max-pretrain-data-tokens", "-1"],
        ["--max-qa-data-tokens", "-1"],
        ["--lm-target-tokens", "0"],
        ["--lm-target-tokens", "-1"],
        ["--save-total-limit", "0"],
        ["--save-total-limit", "-1"],
        ["--icae-min-segments", "0"],
        ["--icae-min-segments", "-1"],
        ["--icae-max-segments", "1.5"],
        ["--ac-num-segments", "0"],
        ["--ac-num-segments", "-1"],
        ["--ac-num-segments", "1.5"],
        ["--ac-bptt-steps", "0"],
        ["--ac-bptt-steps", "-1"],
        ["--method", "autocompressors", "--ac-bptt-steps", "2"],
        ["--method", "icae_single", "--ac-num-segments", "4"],
        ["--method", "icae_multi", "--ac-num-segments", "4"],
        ["--method", "memory_change", "--ac-bptt-steps", "2"],
        ["--method", "information_loss", "--ac-bptt-steps", "2"],
        ["--method", "dynamic_pretrain", "--ac-num-segments", "4"],
        ["--bptt-steps", "-1"],
        ["--method", "icae_single", "--bptt-steps", "2"],
        ["--method", "icae_multi", "--bptt-steps", "2"],
        ["--method", "dynamic_pretrain", "--bptt-steps", "2"],
        ["--method", "icae_single", "--append-slots", "8"],
        ["--method", "icae_multi", "--append-slots", "8"],
        ["--method", "autocompressors", "--append-slots", "8"],
        ["--method", "dynamic_pretrain", "--append-slots", "16"],
        ["--global-batch-size", "8"],
        ["--long-pretrain-data", "old-data"],
        ["--run-id", "../escape"],
        ["--stage", "auto"],
        ["--stage", "warmup"],
        ["--method", "memory_change", "--stage", "policy"],
        ["--method", "icae_single", "--checkpoint", "parent.pt"],
        ["--method", "icae_multi", "--checkpoint", "parent.pt"],
        ["--method", "autocompressors", "--checkpoint", "parent.pt"],
        ["--method", "all", "--checkpoint", "parent.pt"],
        ["--method", "memory_change", "--stage", "pretrain", "--checkpoint", "parent.pt"],
    ],
)
def test_invalid_gpu_scope_or_stage_arguments_are_rejected(tmp_path, options):
    with pytest.raises(SystemExit):
        arguments(tmp_path, *options)


@pytest.mark.parametrize("gpus", ["0,1", "4,5,6,7", "2,4", "7,4", "6"])
@pytest.mark.parametrize("writer_mode", ["local", "tag", "mask", "dual_lora"])
def test_execution_writes_resolved_configs_and_chains_actual_checkpoints(
    tmp_path, monkeypatch, gpus, writer_mode
):
    path = custom_preset_file(
        tmp_path,
        "memory_change",
        objective={"stages": ["pretrain", "warmup", "policy"], "writer_mode": writer_mode},
    )
    args = arguments(tmp_path, "--config", str(path), "--gpus", gpus, checkpoint=False)
    prepare_dataset_entries(args)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    result = gpu_job.run_job(args)
    assert result["status"] == "finished"
    assert all(call["environment"]["SWANLAB_API_KEY"] == "test-api-key" for call in commands.calls)
    assert all(call["environment"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" for call in commands.calls)
    training = [call for call in commands.calls if "configs" in call]
    assert [config.objective.stage for call in training for config in call["configs"]] == [
        "pretrain",
        "warmup",
        "policy",
    ]
    trained = [config for call in training for config in call["configs"]]
    checkpoints = [checkpoint for call in training for checkpoint in call["checkpoints"]]
    assert len(training) == 1
    assert trained[0].training.init_checkpoint is None
    for config, previous in zip(trained[1:], checkpoints[:-1], strict=True):
        assert config.training.init_checkpoint == str(previous)
    for call in training:
        assert call["environment"]["CUDA_VISIBLE_DEVICES"] == gpus
        assert f"--nproc_per_node={len(gpus.split(','))}" in call["command"]
        assert value(call["command"], "--max-steps") == "2"
        for config_path, config in zip(config_paths(call["command"]), call["configs"], strict=True):
            assert json.loads(Path(config_path).read_text()) == config.to_dict()
    evaluation = commands.calls[-1]
    assert "latent_working_memory.v3.evaluate" in evaluation["command"]
    assert value(evaluation["command"], "--checkpoint") == str(checkpoints[-1])
    assert evaluation["environment"]["CUDA_VISIBLE_DEVICES"] == gpus.split(",")[0]
    assert evaluation["evaluation_config"]["max_test_samples"] == 2
    plan_directory = (
        args.output_root / "unit-job" / "plan" / Path(trained[0].training.experiment_dir).name
    )
    plan = json.loads((plan_directory / "job.json").read_text())
    assert len({config.training.experiment_dir for config in trained}) == 1
    assert plan["evaluation_config_path"] == str(plan_directory / "eval.json")
    for job in plan["jobs"]:
        assert job["batching"]["global_batch_size"] == len(gpus.split(",")) * 8
    assert not (args.output_root / "unit-job" / "compare").exists()
    with pytest.raises(ValueError, match="already exists"):
        gpu_job.run_job(args)


@pytest.mark.parametrize("method", DYNAMIC_METHODS)
def test_auto_from_external_checkpoint_needs_only_qa_and_chains_warmup_into_policy(
    tmp_path, monkeypatch, method
):
    checkpoint = external_checkpoint(tmp_path)
    args = arguments(tmp_path, "--method", method, "--checkpoint", str(checkpoint))
    args.qa_data.mkdir()
    for filename in ("train.jsonl", "dev.jsonl", "test.jsonl", "preparation.json"):
        (args.qa_data / filename).write_text("")
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    result = gpu_job.run_job(args)

    assert result["status"] == "finished"
    assert not args.pretrain_data.exists()
    assert len(commands.calls) == 2
    training, evaluation = commands.calls
    warmup, policy = training["configs"]
    assert warmup.objective.method == policy.objective.method == method
    assert warmup.objective.stage == "warmup"
    assert policy.objective.stage == "policy"
    assert warmup.training.init_checkpoint == str(checkpoint.resolve())
    assert policy.training.init_checkpoint == str(training["checkpoints"][0])
    assert policy.training.init_checkpoint != str(checkpoint.resolve())
    assert "latent_working_memory.v3.evaluate" in evaluation["command"]
    assert value(evaluation["command"], "--checkpoint") == str(training["checkpoints"][1])
    assert "--log-to-swanlab" in evaluation["command"]
    prefix = method.replace("_", "-")
    assert result["checkpoints"][f"{prefix}-warmup"] == str(training["checkpoints"][0])
    assert result["checkpoints"][f"{prefix}-policy"] == str(training["checkpoints"][1])
    assert len(result["summaries"]) == 1
    assert "dynamic-pretrain" not in result["checkpoints"]


@pytest.mark.parametrize("method", DYNAMIC_METHODS)
def test_new_capacity_names_reuse_old_pretraining_paths_and_preserve_source_identity(
    tmp_path, monkeypatch, method
):
    checkpoint = external_checkpoint(tmp_path, experiment_id="legacy-series")
    source_directory = checkpoint.parent.parent
    identity_path = source_directory.parent / "swanlab.json"
    identity_path.write_text(
        json.dumps(
            {
                "id": "old-cloud-id",
                "url": "https://swanlab.cn/@test-user/latent-working-memory-v3/runs/old-cloud-id",
                "project": "latent-working-memory-v3",
                "group": "legacy-series",
                "name": "dynamic-pretrain-k64_legacy-series",
            }
        )
    )
    original_files = {
        path: path.read_bytes()
        for path in (checkpoint / "state.pt", source_directory / "run.json", identity_path)
    }
    args = arguments(
        tmp_path,
        "--method",
        method,
        "--mode",
        "full",
        "--append-slots",
        "16",
        "--checkpoint",
        str(checkpoint),
        run_id=None,
    )
    prepare_dataset_entries(args)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    result = gpu_job.run_job(args)

    assert args.run_id == result["experiment_id"] == "legacy-series"
    assert source_directory.parent.name == "dynamic-pretrain-k64_legacy-series"
    name = f"{method.replace('_', '-')}-k512+16-local"
    training, evaluation = commands.calls
    for config in training["configs"]:
        assert config.training.experiment_name == f"{name}_legacy-series"
        assert config.training.experiment_dir == str(
            args.output_root / "legacy-series" / "train" / name
        )
        assert config.training.group == "legacy-series"
    assert training["configs"][0].training.init_checkpoint == str(checkpoint.resolve())
    assert Path(value(evaluation["command"], "--output-dir")).parent.parent.name == name
    assert "dynamic-pretrain" not in result["checkpoints"]
    assert {path: path.read_bytes() for path in original_files} == original_files


def test_auto_from_missing_checkpoint_fails_before_starting_training(tmp_path, monkeypatch):
    args = arguments(tmp_path, "--checkpoint", str(tmp_path / "missing.pt"))
    args.qa_data.mkdir()
    for filename in ("train.jsonl", "dev.jsonl", "test.jsonl", "preparation.json"):
        (args.qa_data / filename).write_text("")
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    with pytest.raises(FileNotFoundError) as error:
        gpu_job.run_job(args)

    assert "missing.pt" in str(error.value)
    assert not commands.calls
    assert not args.output_root.exists()


def test_dynamic_pretraining_entry_requires_only_ae_lm_data(tmp_path, monkeypatch):
    args = arguments(tmp_path, "--mode", "full", "--method", "dynamic_pretrain")
    args.pretrain_data.mkdir(parents=True)
    (args.pretrain_data / "preparation.json").write_text("{}")
    for split in ("train", "dev", "test"):
        (args.pretrain_data / f"{split}.jsonl").write_text("")
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    result = gpu_job.run_job(args)

    assert len(commands.calls) == 1
    assert result["summaries"] == []
    config = commands.calls[0]["configs"][0]
    assert (config.objective.method, config.objective.stage) == ("dynamic", "pretrain")
    assert config.training.experiment_id == "unit-job"
    assert Path(config.training.experiment_dir).name == "dynamic-pretrain-k512"
    assert config.training.experiment_name == "dynamic-pretrain-k512_unit-job"
    assert Path(config.training.output_dir) == Path(config.training.experiment_dir) / "pretrain"
    assert not args.qa_data.exists()
    assert not (args.pretrain_data / "single").exists()


@pytest.mark.parametrize("method", ["icae_single", "icae_multi", "autocompressors"])
def test_baselines_use_the_same_multisegment_root(tmp_path, monkeypatch, method):
    args = arguments(tmp_path, "--method", method)
    prepare_dataset_entries(args)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    result = gpu_job.run_job(args)

    assert result["status"] == "finished"
    trained = [config for call in commands.calls for config in call.get("configs", [])]
    assert Path(trained[0].training.dataset_dir) == args.pretrain_data


@pytest.mark.parametrize("missing", ["preparation.json", "train.jsonl", "dev.jsonl", "test.jsonl"])
def test_method_requires_multisegment_root_entries_before_execution(tmp_path, monkeypatch, missing):
    args = arguments(tmp_path, "--method", "icae_single")
    prepare_dataset_entries(args)
    (args.pretrain_data / missing).unlink()
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    with pytest.raises(FileNotFoundError, match="required dataset entry is missing") as error:
        gpu_job.run_job(args)

    assert str(args.pretrain_data / missing) in str(error.value)
    assert not commands.calls
    assert not args.output_root.exists()


def test_separate_methods_share_source_id_and_coexist_without_overwriting_series(
    tmp_path, monkeypatch
):
    shared = arguments(tmp_path, "--mode", "full", "--method", "dynamic_pretrain")
    prepare_dataset_entries(shared)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    source_result = gpu_job.run_job(shared)
    source_checkpoint = source_result["checkpoints"]["dynamic-pretrain"]
    source_run = commands.calls[0]["configs"][0]
    source_metadata = Path(source_run.training.output_dir) / "run.json"
    original_source = source_metadata.read_bytes()
    descendants = []
    for method in ("memory_change", "information_loss"):
        args = arguments(
            tmp_path,
            "--mode",
            "full",
            "--method",
            method,
            "--checkpoint",
            source_checkpoint,
            run_id=None,
        )
        assert args.run_id is None
        result = gpu_job.run_job(args)
        assert args.run_id == result["experiment_id"] == "unit-job"
        descendants.append(args)
    baseline = arguments(tmp_path, "--mode", "full", "--method", "icae_single")
    gpu_job.run_job(baseline)

    directory = shared.output_root / "unit-job"
    assert {path.name for path in (directory / "plan").iterdir()} == {
        "dynamic-pretrain-k512",
        "memory-change-k512+32-local",
        "information-loss-k512+32-local",
        "icae-single-k512",
    }
    assert {path.name for path in (directory / "train").iterdir()} == {
        "dynamic-pretrain-k512",
        "memory-change-k512+32-local",
        "information-loss-k512+32-local",
        "icae-single-k512",
    }
    assert source_metadata.read_bytes() == original_source
    for method in ("memory_change", "information_loss"):
        stages = [
            config
            for call in commands.calls
            for config in call.get("configs", [])
            if config.objective.method == method and config.objective.stage != "pretrain"
        ]
        assert [config.objective.stage for config in stages] == ["warmup", "policy"]
        assert len({config.training.experiment_dir for config in stages}) == 1
        assert stages[0].training.init_checkpoint == source_checkpoint
    before = len(commands.calls)
    with pytest.raises(ValueError, match="already exists"):
        gpu_job.run_job(descendants[0])
    assert len(commands.calls) == before
    assert source_metadata.read_bytes() == original_source


def test_separate_dynamic_methods_reuse_pretraining_without_changing_its_plan(
    tmp_path, monkeypatch
):
    shared = arguments(tmp_path, "--mode", "full", "--method", "dynamic_pretrain")
    prepare_dataset_entries(shared)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    source_result = gpu_job.run_job(shared)
    source_checkpoint = source_result["checkpoints"]["dynamic-pretrain"]
    shared_directory = shared.output_root / "unit-job" / "plan/dynamic-pretrain-k512"
    original_plan = {path.name: path.read_bytes() for path in shared_directory.iterdir()}

    results = []
    for method in DYNAMIC_METHODS:
        args = arguments(
            tmp_path, "--mode", "full", "--method", method, "--checkpoint", source_checkpoint
        )
        results.append(gpu_job.run_job(args))

    assert all(result["status"] == "finished" for result in results)
    assert {key for result in results for key in result["checkpoints"]} == {
        "memory-change-warmup",
        "memory-change-policy",
        "information-loss-warmup",
        "information-loss-policy",
    }
    assert {path.name for path in (shared.output_root / "unit-job" / "plan").iterdir()} == {
        "dynamic-pretrain-k512",
        "memory-change-k512+32-local",
        "information-loss-k512+32-local",
    }
    assert {path.name: path.read_bytes() for path in shared_directory.iterdir()} == original_plan
    training = [call for call in commands.calls if "configs" in call]
    assert len(training) == 3
    assert all(
        call["configs"][0].training.init_checkpoint == source_checkpoint for call in training[1:]
    )
    assert len(commands.calls) == 5
    assert not (shared.output_root / "unit-job/compare").exists()


@pytest.mark.parametrize("method", DYNAMIC_METHODS)
@pytest.mark.parametrize("writer_mode", ["local", "tag", "mask", "dual_lora"])
def test_explicit_id_reuses_pretraining_in_a_new_series(tmp_path, monkeypatch, method, writer_mode):
    checkpoint = external_checkpoint(tmp_path, experiment_id="original-id")
    source_paths = (checkpoint / "state.pt", checkpoint.parent.parent / "run.json")
    original_source = {path: path.read_bytes() for path in source_paths}
    args = arguments(
        tmp_path,
        "--method",
        method,
        "--writer-mode",
        writer_mode,
        "--checkpoint",
        str(checkpoint),
        run_id="writer-mode-comparison",
    )
    args.qa_data.mkdir()
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "preparation.json"):
        (args.qa_data / name).write_text("")
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    result = gpu_job.run_job(args)

    assert result["experiment_id"] == args.run_id == "writer-mode-comparison"
    method_directory = f"{method.replace('_', '-')}-k512+32-{writer_mode.replace('_', '-')}_smoke"
    assert result["status"] == "finished"
    training, evaluation = commands.calls
    warmup, policy = training["configs"]
    for config in (warmup, policy):
        assert config.training.experiment_id == args.run_id
        assert config.training.group == args.run_id
        assert config.training.experiment_name == f"{method_directory}_{args.run_id}"
        assert config.training.experiment_dir == str(
            args.output_root / args.run_id / "train" / method_directory
        )
    assert warmup.training.init_checkpoint == str(checkpoint)
    assert policy.training.init_checkpoint == str(training["checkpoints"][0])
    assert value(evaluation["command"], "--checkpoint") == str(training["checkpoints"][1])
    assert not (args.output_root / "original-id").exists()
    assert not args.pretrain_data.exists()
    assert {path: path.read_bytes() for path in source_paths} == original_source


@pytest.mark.parametrize(
    "method,stage",
    [("memory_change", "warmup"), ("information_loss", "policy"), ("icae_single", "pretrain")],
)
def test_initialization_must_come_from_shared_dynamic_pretraining(tmp_path, method, stage):
    checkpoint = external_checkpoint(tmp_path, method=method, stage=stage)
    args = arguments(tmp_path, "--checkpoint", str(checkpoint))
    with pytest.raises(ValueError, match="requires a shared dynamic pretraining run"):
        gpu_job.run_job(args)
    assert not args.output_root.exists()


def test_real_initialization_requires_source_sidecar_even_with_explicit_id(tmp_path):
    checkpoint = external_checkpoint(tmp_path)
    (checkpoint.parent.parent / "run.json").unlink()
    args = arguments(tmp_path, "--checkpoint", str(checkpoint))
    with pytest.raises(FileNotFoundError, match="source run.json required"):
        gpu_job.run_job(args)
    assert not args.output_root.exists()


@pytest.mark.parametrize("identity", [None, "../invalid"])
def test_initialization_requires_valid_source_experiment_id(tmp_path, identity):
    checkpoint = external_checkpoint(tmp_path, experiment_id=identity)
    args = arguments(tmp_path, "--checkpoint", str(checkpoint))
    with pytest.raises(ValueError, match="no valid experiment_id"):
        gpu_job.run_job(args)
    assert not args.output_root.exists()


@pytest.mark.parametrize("target", ["stage", "plan", "evaluation"])
def test_existing_owned_output_is_rejected_without_touching_it(tmp_path, monkeypatch, target):
    args = arguments(tmp_path, "--method", "icae_single")
    directory, _, jobs = job_plan.build_jobs(args)
    evaluated = next(job for job in jobs if job.evaluate)
    path = {
        "stage": Path(jobs[0].config.training.output_dir),
        "plan": jobs[0].config_path.parent,
        "evaluation": directory
        / "eval"
        / Path(evaluated.config.training.experiment_dir).name
        / evaluated.config.objective.stage,
    }[target]
    path.mkdir(parents=True)
    marker = path / "keep.txt"
    marker.write_text("existing output")
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        gpu_job.run_job(args)
    assert marker.read_text() == "existing output"
    assert not commands.calls


def test_full_execution_uses_its_stage_budget_without_smoke_checkpoint(tmp_path, monkeypatch):
    smoke = arguments(tmp_path, "--method", "dynamic_pretrain")
    prepare_dataset_entries(smoke)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    short_result = gpu_job.run_job(smoke)
    full = arguments(tmp_path, "--mode", "full", "--method", "dynamic_pretrain")
    full_result = gpu_job.run_job(full)
    assert len(commands.calls) == 2
    directory = full.output_root / "unit-job"
    assert {path.name for path in (directory / "plan").iterdir()} == {
        "dynamic-pretrain-k512_smoke",
        "dynamic-pretrain-k512",
    }
    assert short_result["experiment_id"] == full_result["experiment_id"] == "unit-job"
    assert set(short_result["checkpoints"].values()).isdisjoint(full_result["checkpoints"].values())
    full_call = commands.calls[-1]
    assert "--max-steps" not in full_call["command"]
    assert full_call["configs"][0].training.init_checkpoint is None
    assert training_limit(full_call["configs"][0]) == 12800
    assert full_call["configs"][0].training.max_dev_samples is None


@pytest.mark.parametrize("fail_at", [1, 2])
def test_failed_training_or_evaluation_preserves_method_result(tmp_path, monkeypatch, fail_at):
    args = arguments(tmp_path)
    prepare_dataset_entries(args)
    commands = Commands(fail_at=fail_at)
    monkeypatch.setattr(gpu_job, "execute", commands)
    with pytest.raises(subprocess.CalledProcessError):
        gpu_job.run_job(args)
    assert len(commands.calls) == fail_at
    result = json.loads(
        (
            args.output_root / "unit-job/plan/memory-change-k512+32-local_smoke/result.json"
        ).read_text()
    )
    assert result["status"] == "failed" and "error" in result
    assert set(result["checkpoints"]) == (
        set() if fail_at == 1 else {"memory-change-warmup", "memory-change-policy"}
    )
    assert not result["summaries"]


def test_second_stage_failure_preserves_latest_saved_checkpoint_in_result(tmp_path, monkeypatch):
    args = arguments(tmp_path, "--save-total-limit", "1")
    prepare_dataset_entries(args)
    commands = Commands(fail_stage="policy")
    monkeypatch.setattr(gpu_job, "execute", commands)
    with pytest.raises(subprocess.CalledProcessError):
        gpu_job.run_job(args)
    assert len(commands.calls) == 1
    failure = json.loads(
        (
            args.output_root / "unit-job/plan/memory-change-k512+32-local_smoke/result.json"
        ).read_text()
    )
    assert failure["status"] == "failed"
    assert set(failure["checkpoints"]) == {"memory-change-policy"}
    assert failure["checkpoints"]["memory-change-policy"] == str(
        commands.calls[0]["checkpoints"][-1]
    )
    assert not commands.calls[0]["checkpoints"][0].exists()
    policy = commands.calls[0]["configs"][-1]
    assert not (Path(policy.training.output_dir) / "training-result.json").exists()
    assert not failure["summaries"]


def test_training_must_save_the_reported_checkpoint_before_dependent_stage(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    prepare_dataset_entries(args)
    commands = Commands(omit_checkpoint=True)
    monkeypatch.setattr(gpu_job, "execute", commands)
    with pytest.raises(FileNotFoundError, match="reported checkpoint"):
        gpu_job.run_job(args)
    assert len(commands.calls) == 1


@pytest.mark.parametrize("module", ["job_plan", "gpu_job"])
def test_planning_imports_no_model_or_tracking_runtime(tmp_path, module):
    code = (
        f"import importlib, sys; importlib.import_module('latent_working_memory.v3.{module}'); "
        "assert not {'torch', 'transformers', 'swanlab'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("dotenv", [None, "SWANLAB_API_KEY=project-test-key\n"])
def test_selected_key_reaches_all_stages_without_being_persisted(
    tmp_path, monkeypatch, capsys, dotenv
):
    repository = Path.cwd()
    monkeypatch.setattr(job_plan, "load_preset", lambda path: load_preset(repository / path))
    monkeypatch.setattr(gpu_job, "swanlab_api_key", tracking_credentials.swanlab_api_key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWANLAB_API_KEY", "terminal-test-key")
    if dotenv is not None:
        (tmp_path / ".env").write_text(dotenv)
    args = arguments(tmp_path, "--method", "icae_single")
    prepare_dataset_entries(args)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    gpu_job.run_job(args)

    expected = "project-test-key" if dotenv is not None else "terminal-test-key"
    assert all(call["environment"]["SWANLAB_API_KEY"] == expected for call in commands.calls)
    assert os.environ["SWANLAB_API_KEY"] == "terminal-test-key"
    output = capsys.readouterr().out
    saved = "".join(path.read_text() for path in args.output_root.rglob("*.json"))
    command_text = str([call["command"] for call in commands.calls])
    for key in ("project-test-key", "terminal-test-key"):
        assert key not in output + saved + command_text


@pytest.mark.parametrize("tracking", ["online", "disabled"])
def test_missing_key_stops_online_job_before_creating_artifacts(tmp_path, monkeypatch, tracking):
    repository = Path.cwd()
    monkeypatch.setattr(job_plan, "load_preset", lambda path: load_preset(repository / path))
    monkeypatch.setattr(gpu_job, "swanlab_api_key", tracking_credentials.swanlab_api_key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    args = arguments(tmp_path, "--tracking", tracking)
    prepare_dataset_entries(args)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    if tracking == "online":
        with pytest.raises(ValueError, match="nonempty SWANLAB_API_KEY"):
            gpu_job.run_job(args)
        assert not commands.calls and not args.output_root.exists()
    else:
        gpu_job.run_job(args)
        assert commands.calls


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("launch_from", ["repository", "elsewhere"])
def test_shell_locates_repo_preserves_arguments_and_returns_process_status(
    tmp_path, exit_code, launch_from
):
    repository = tmp_path / "repository with spaces"
    repository.mkdir()
    script = repository / "src/latent_working_memory/v3/scripts/run.sh"
    script.parent.mkdir(parents=True)
    script.write_bytes(Path("src/latent_working_memory/v3/scripts/run.sh").read_bytes())
    binaries = tmp_path / "fake bin"
    binaries.mkdir()
    fake_uv = binaries / "uv"
    fake_uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:], "
        "'unbuffered': os.environ['PYTHONUNBUFFERED'], "
        "'tokenizers': os.environ['TOKENIZERS_PARALLELISM']}))\n"
        f"sys.exit({exit_code})\n"
    )
    fake_uv.chmod(0o755)
    options = [
        "--mode",
        "smoke",
        "--model-path",
        "models/name with spaces",
        "--group",
        "literal $name; echo unchanged",
    ]
    environment = dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}")
    working_directory = repository if launch_from == "repository" else tmp_path
    result = subprocess.run(
        ["bash", str(script), *options],
        cwd=working_directory,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code
    recorded = json.loads(result.stdout)
    assert recorded["cwd"] == str(repository)
    assert recorded["args"] == [
        "run",
        "--frozen",
        "python",
        "-m",
        "latent_working_memory.v3.gpu_job",
        *options,
    ]
    assert recorded["unbuffered"] == "1"
    assert recorded["tokenizers"] == "false"


def test_execute_propagates_real_child_failure_and_keeps_the_log(tmp_path):
    log = tmp_path / "child.log"
    command = [sys.executable, "-c", "print('visible failure'); raise SystemExit(3)"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        gpu_job.execute(command, dict(os.environ), log)
    assert error.value.returncode == 3
    assert log.read_text().strip() == "visible failure"
