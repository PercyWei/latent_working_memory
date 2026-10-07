"""验证任务编排和 shell 契约；子进程替身不运行 GPU、模型或云端请求。"""

import json
import os
from dataclasses import replace
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from latent_working_memory.v3 import gpu_job, tracking_credentials
from latent_working_memory.v3.config import load_experiment


@pytest.fixture(autouse=True)
def explicit_test_credentials(monkeypatch):
    # 编排测试不读取开发者的真实凭据；真实解析流程另行使用临时目录测试。
    monkeypatch.setattr(gpu_job, "swanlab_api_key", lambda: "test-api-key")


def arguments(tmp_path, *options, run_id="unit-job"):
    return gpu_job.parse_args(
        [
            "--method",
            "dynamic",
            *(["--run-id", run_id] if run_id is not None else []),
            "--output-root",
            str(tmp_path / "outputs"),
            "--pretrain-data",
            str(tmp_path / "pretraining data"),
            "--qa-data",
            str(tmp_path / "qa data"),
            *options,
        ]
    )


def prepare_dataset_entries(args):
    # 这里只测试编排入口的文件要求；内容解析由真正的训练/评估测试覆盖。
    for directory in (args.pretrain_data / "single", args.pretrain_data / "multi", args.qa_data):
        directory.mkdir(parents=True)
        for split in ("train", "dev", "test"):
            (directory / f"{split}.jsonl").write_text("")
    (args.pretrain_data / "preparation.json").write_text("{}")
    (args.qa_data / "preparation.json").write_text("{}")


def value(command, flag):
    return command[command.index(flag) + 1]


def external_checkpoint(
    tmp_path, experiment_id="unit-job", method="memory_change", stage="pretrain"
):
    root = tmp_path / "source" / f"shared-pretrain-k64_{experiment_id}"
    output = root / stage
    (output / "checkpoints").mkdir(parents=True)
    checkpoint = output / "checkpoints" / "step-000002.pt"
    checkpoint.write_bytes(b"opaque external pretraining checkpoint fixture")
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
    def __init__(self, fail_at=None, omit_checkpoint=False):
        self.calls = []
        self.fail_at = fail_at
        self.omit_checkpoint = omit_checkpoint

    def __call__(self, command, environment, log_path):
        call = {"command": command, "environment": dict(environment), "log_path": log_path}
        self.calls.append(call)
        if self.fail_at == len(self.calls):
            raise subprocess.CalledProcessError(7, command)
        if "latent_working_memory.v3.train" in command:
            config = load_experiment(value(command, "--config"))
            call["config"] = config
            output = Path(config.training.output_dir)
            (output / "checkpoints").mkdir(parents=True)
            (output / "run.json").write_text(json.dumps({"config": config.to_dict()}))
            steps = (
                int(value(command, "--stop-after-steps")) if "--stop-after-steps" in command else 80
            )
            checkpoint = output / "checkpoints" / f"step-{steps:06d}.pt"
            if not self.omit_checkpoint:
                # 编排器只传递此文件，不读取权重；这不是可加载的模型 checkpoint。
                checkpoint.write_bytes(b"opaque orchestration checkpoint fixture")
            (output / "training-result.json").write_text(
                json.dumps(
                    {
                        "complete": "--stop-after-steps" not in command,
                        "completed_steps": steps,
                        "total_steps": 80,
                        "completed_epochs": 1 if steps == 80 else 0,
                        "sample_visits": steps
                        * config.training.global_batch_size(
                            len(environment["CUDA_VISIBLE_DEVICES"].split(","))
                        ),
                        "stop_after_steps": steps if steps != 80 else None,
                        "checkpoint": str(checkpoint),
                    }
                )
            )
            call["checkpoint"] = checkpoint
        elif "latent_working_memory.v3.evaluate" in command:
            output = Path(value(command, "--output-dir"))
            output.mkdir(parents=True)
            (output / "summary.json").write_text(
                json.dumps({"split": value(command, "--split"), "trajectories": 2})
            )
        else:
            assert "latent_working_memory.v3.compare" in command
            Path(value(command, "--output-dir")).mkdir(parents=True)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("smoke", (16, 4, 2, 2, "dev")),
        ("pilot", (256, 32, 20, 16, "dev")),
        ("full", (None, None, None, None, "test")),
    ],
)
def test_modes_apply_bounded_or_complete_training_and_evaluation(tmp_path, mode, expected):
    args = arguments(tmp_path, "--mode", mode)
    assert args.gpus == "0,1"
    directory, level, jobs = gpu_job.build_jobs(args)
    assert (
        tuple(
            level[name]
            for name in (
                "train_samples",
                "dev_samples",
                "max_steps",
                "eval_trajectories",
                "eval_split",
            )
        )
        == expected
    )
    assert directory.name == (
        "capacity_unit-job" if mode == "full" else f"capacity-{mode}_unit-job"
    )
    for job in jobs:
        assert job.config.training.max_train_samples == expected[0]
        assert job.config.training.max_dev_samples == expected[1]
        assert job.config.training.init_checkpoint is None
        assert job.config.training.swanlab_project == "latent-working-memory-v3"
        assert job.config.training.tags == (f"study:{'main' if mode == 'full' else mode}",)
        assert job.config.training.micro_batch_size_per_gpu == 2
        assert job.config.training.gradient_accumulation_steps == 2
        assert job.config.training.global_batch_size(2) == 8
        assert job.config.training.global_batch_size(4) == 16
        assert job.config.training.experiment_id == "unit-job"
        root = Path(job.config.training.experiment_dir)
        assert Path(job.config.training.output_dir) == root / job.config.objective.stage
        assert root.name.endswith("-k64_unit-job" if mode == "full" else f"-k64_{mode}_unit-job")
        assert job.config_path.parent == directory / "plan" / "dynamic"
        assert "global_batch_size" not in job.config.to_dict()["training"]
        command = gpu_job.training_command(job, 4, level["max_steps"])
        assert "--nproc_per_node=4" in command
        assert ("--stop-after-steps" in command) == (mode != "full")
    evaluation = gpu_job.evaluation_command(args, level, Path("final.pt"), Path("evaluation"))
    assert ("--max-trajectories" in evaluation) == (mode != "full")
    assert value(evaluation, "--split") == expected[-1]
    assert "--log-to-swanlab" in evaluation


def test_dynamic_methods_share_only_pretraining_and_have_independent_warmup_policy(tmp_path):
    _, _, jobs = gpu_job.build_jobs(arguments(tmp_path))
    assert [job.key for job in jobs] == [
        "dynamic-pretrain",
        "memory-change-warmup",
        "memory-change-policy",
        "information-loss-warmup",
        "information-loss-policy",
    ]
    assert [job.initialize_from for job in jobs] == [
        None,
        "dynamic-pretrain",
        "memory-change-warmup",
        "dynamic-pretrain",
        "information-loss-warmup",
    ]
    assert [job.evaluate for job in jobs] == [False, False, True, False, True]
    assert len({job.config.training.output_dir for job in jobs}) == 5
    assert len({job.config.training.experiment_dir for job in jobs}) == 3
    assert Path(jobs[0].config.training.experiment_dir).name == "shared-pretrain-k64_smoke_unit-job"
    assert jobs[1].config.training.experiment_dir == jobs[2].config.training.experiment_dir
    assert jobs[3].config.training.experiment_dir == jobs[4].config.training.experiment_dir
    assert len({job.config.training.seed for job in jobs}) == 1
    assert len({job.config.training.group for job in jobs}) == 1


def test_default_pretraining_data_is_the_shared_reconstruction_root():
    args = gpu_job.parse_args([])
    assert args.pretrain_data == Path("data/fineweb-reconstruction-k512-doc100k_20260917")


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_minimal_command_builds_all_methods_with_shared_default_settings(mode):
    args = gpu_job.parse_args(["--mode", mode, "--gpus", "4,5"])
    assert args.method == "all"
    assert args.run_id is None
    _, _, jobs = gpu_job.build_jobs(args)
    assert len(jobs) == 10
    assert sum(job.key == "dynamic-pretrain" for job in jobs) == 1
    assert len({job.config.training.experiment_id for job in jobs}) == 1
    for job in jobs:
        assert job.config.model.model_name_or_path == str(
            Path.home() / "models/Qwen3-4B-Instruct-2507"
        )
        assert job.config.training.micro_batch_size_per_gpu == 2
        assert job.config.training.gradient_accumulation_steps == 2
        assert job.config.training.global_batch_size(2) == 8
        assert job.config.objective.qa_batch_size == 8


@pytest.mark.parametrize("override", [False, True])
def test_cli_only_overrides_explicitly_supplied_preset_values(monkeypatch, override):
    def preset(path):
        config = load_experiment(path)
        return replace(
            config,
            model=replace(config.model, model_name_or_path="local/preset-model"),
            objective=replace(config.objective, qa_batch_size=5),
            training=replace(
                config.training, micro_batch_size_per_gpu=3, gradient_accumulation_steps=5
            ),
        )

    monkeypatch.setattr(gpu_job, "load_experiment", preset)
    options = (
        ["--micro-batch-size-per-gpu", "4", "--model-path", "~/models/override-model"]
        if override
        else []
    )
    args = gpu_job.parse_args(options)
    assert args.method == "all" and args.mode == "smoke"
    _, _, jobs = gpu_job.build_jobs(args)
    for job in jobs:
        assert job.config.training.micro_batch_size_per_gpu == (4 if override else 3)
        assert job.config.training.gradient_accumulation_steps == 5
        assert job.config.objective.qa_batch_size == 5
        assert job.config.model.model_name_or_path == (
            str(Path.home() / "models/override-model") if override else "local/preset-model"
        )


@pytest.mark.parametrize("method", ["memory_change", "information_loss", "dynamic"])
def test_auto_from_external_pretraining_checkpoint_skips_pretraining_in_plan(tmp_path, method):
    checkpoint = tmp_path / "shared pretraining.pt"
    args = arguments(
        tmp_path, "--dry-run", "--method", method, "--init-checkpoint", str(checkpoint)
    )

    _, _, jobs = gpu_job.build_jobs(args)

    methods = ("memory_change", "information_loss") if method == "dynamic" else (method,)
    assert len(jobs) == 2 * len(methods)
    for index, name in enumerate(methods):
        warmup, policy = jobs[index * 2 : index * 2 + 2]
        prefix = name.replace("_", "-")
        assert warmup.key == f"{prefix}-warmup"
        assert policy.key == f"{prefix}-policy"
        assert warmup.config.objective.method == policy.config.objective.method == name
        assert warmup.config.objective.stage == "warmup"
        assert policy.config.objective.stage == "policy"
        assert warmup.config.training.init_checkpoint == str(checkpoint.resolve())
        assert warmup.initialize_from is None
        assert policy.config.training.init_checkpoint is None
        assert policy.initialize_from == warmup.key
        assert not warmup.evaluate and policy.evaluate
        assert Path(warmup.config.training.dataset_dir) == args.qa_data
        assert Path(policy.config.training.dataset_dir) == args.qa_data
    assert not checkpoint.exists()
    assert list(tmp_path.iterdir()) == []


def test_all_methods_build_a_complete_topologically_ordered_stage_graph(tmp_path):
    args = arguments(tmp_path, "--method", "all", "--mode", "full")
    _, _, jobs = gpu_job.build_jobs(args)
    assert len(jobs) == 10
    assert {
        (job.config.objective.method, job.config.objective.stage) for job in jobs if job.evaluate
    } == {
        ("icae_single", "qa"),
        ("icae_multi", "qa"),
        ("autocompressors", "lm"),
        ("memory_change", "policy"),
        ("information_loss", "policy"),
    }
    seen = set()
    for job in jobs:
        assert job.initialize_from is None or job.initialize_from in seen
        seen.add(job.key)
        if job.config.objective.stage in {"qa", "warmup", "policy"}:
            assert Path(job.config.training.dataset_dir) == args.qa_data
        else:
            assert Path(job.config.training.dataset_dir) == args.pretrain_data
            assert job.config.training.pretrain_data_view == (
                "reconstruction_first_write"
                if job.config.objective.method in {"memory_change", "information_loss"}
                else "reconstruction_single"
            )
    assert len({job.config.training.experiment_dir for job in jobs}) == 6
    for method in ("icae_single", "icae_multi"):
        stages = [job for job in jobs if job.config.objective.method == method]
        assert [job.config.objective.stage for job in stages] == ["pretrain", "qa"]
        assert stages[0].config.training.experiment_dir == stages[1].config.training.experiment_dir
        assert (
            Path(stages[0].config.training.experiment_dir).name
            == f"{method.replace('_', '-')}-k64_unit-job"
        )


def test_explicit_overrides_and_zero_remove_profile_limits(tmp_path):
    args = arguments(
        tmp_path,
        "--method",
        "memory_change",
        "--dry-run",
        "--init-checkpoint",
        "parent checkpoint.pt",
        "--train-samples",
        "0",
        "--dev-samples",
        "6",
        "--max-steps",
        "0",
        "--eval-trajectories",
        "0",
        "--eval-split",
        "test",
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
        "--eval-every",
        "2",
        "--save-every",
        "3",
        "--tracking",
        "disabled",
        "--group",
        "chosen-group",
        "--model-path",
        "models/local Qwen3",
    )
    _, level, jobs = gpu_job.build_jobs(args)
    assert len(jobs) == 2
    warmup, job = jobs
    assert warmup.config.training.init_checkpoint == str(Path("parent checkpoint.pt").resolve())
    assert job.initialize_from == warmup.key and job.evaluate
    assert job.config.training.init_checkpoint is None
    assert job.config.training.max_train_samples is None
    assert job.config.training.max_dev_samples == 6
    assert job.config.training.epochs == 3
    assert job.config.training.micro_batch_size_per_gpu == 2
    assert job.config.training.gradient_accumulation_steps == 3
    assert job.config.training.global_batch_size(2) == 12
    assert job.config.training.group == "chosen-group"
    assert job.config.training.swanlab_project is None
    assert job.config.objective.qa_batch_size == 2
    assert job.config.objective.threshold_i == 0.25
    assert job.config.model.model_name_or_path == "models/local Qwen3"
    assert level["max_steps"] is None and level["eval_trajectories"] is None
    assert level["eval_every"] == 2 and level["save_every"] == 3
    assert "--log-to-swanlab" not in gpu_job.evaluation_command(args, level, Path("a"), Path("b"))


def test_plan_derives_each_global_batch_from_selected_gpus_microbatch_and_accumulation(tmp_path):
    args = arguments(
        tmp_path,
        "--dry-run",
        "--gpus",
        "4,5,6,7",
        "--micro-batch-size-per-gpu",
        "2",
        "--gradient-accumulation-steps",
        "3",
    )

    plan = gpu_job.run_job(args)

    assert len(plan["jobs"]) == 5
    assert all(
        job["batching"]
        == {
            "world_size": 4,
            "micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 3,
            "global_batch_size": 24,
        }
        for job in plan["jobs"]
    )
    assert all("global_batch_size" not in job["config"]["training"] for job in plan["jobs"])
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
        ["--train-samples", "-1"],
        ["--epochs", "0"],
        ["--micro-batch-size-per-gpu", "0"],
        ["--micro-batch-size-per-gpu", "-1"],
        ["--gradient-accumulation-steps", "0"],
        ["--gradient-accumulation-steps", "-1"],
        ["--global-batch-size", "8"],
        ["--long-pretrain-data", "old-data"],
        ["--run-id", "../escape"],
        ["--stage", "auto"],
        ["--stage", "warmup"],
        ["--method", "memory_change", "--stage", "policy"],
        ["--method", "icae_single", "--init-checkpoint", "parent.pt"],
        ["--method", "icae_multi", "--init-checkpoint", "parent.pt"],
        ["--method", "autocompressors", "--init-checkpoint", "parent.pt"],
        ["--method", "all", "--init-checkpoint", "parent.pt"],
        ["--method", "shared_pretrain", "--init-checkpoint", "parent.pt"],
        ["--method", "memory_change", "--stage", "pretrain", "--init-checkpoint", "parent.pt"],
    ],
)
def test_invalid_gpu_scope_or_stage_arguments_are_rejected(tmp_path, options):
    with pytest.raises(SystemExit):
        arguments(tmp_path, *options)


@pytest.mark.parametrize("gpus", ["0,1", "4,5,6,7", "2,4", "7,4", "6"])
def test_execution_writes_resolved_configs_and_chains_actual_checkpoints(
    tmp_path, monkeypatch, gpus
):
    args = arguments(tmp_path, "--gpus", gpus)
    prepare_dataset_entries(args)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    result = gpu_job.run_job(args)

    assert result["status"] == "finished"
    assert all(call["environment"]["SWANLAB_API_KEY"] == "test-api-key" for call in commands.calls)
    assert all(call["environment"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" for call in commands.calls)
    training = [call for call in commands.calls if "config" in call]
    evaluations = [
        call for call in commands.calls if "latent_working_memory.v3.evaluate" in call["command"]
    ]
    assert len(training) == 5 and len(evaluations) == 2 and len(commands.calls) == 8
    assert training[0]["config"].training.init_checkpoint is None
    assert training[1]["config"].training.init_checkpoint == str(training[0]["checkpoint"])
    assert training[2]["config"].training.init_checkpoint == str(training[1]["checkpoint"])
    assert training[3]["config"].training.init_checkpoint == str(training[0]["checkpoint"])
    assert training[4]["config"].training.init_checkpoint == str(training[3]["checkpoint"])
    for call in training:
        assert call["environment"]["CUDA_VISIBLE_DEVICES"] == gpus
        assert f"--nproc_per_node={len(gpus.split(','))}" in call["command"]
        assert value(call["command"], "--stop-after-steps") == "2"
        assert (
            json.loads(Path(value(call["command"], "--config")).read_text())
            == call["config"].to_dict()
        )
    for call, trained in zip(evaluations, (training[2], training[4]), strict=True):
        assert value(call["command"], "--checkpoint") == str(trained["checkpoint"])
        assert call["environment"]["CUDA_VISIBLE_DEVICES"] == gpus.split(",")[0]
        assert value(call["command"], "--split") == "dev"
        assert value(call["command"], "--max-trajectories") == "2"
        assert "--log-to-swanlab" in call["command"]
    assert "latent_working_memory.v3.compare" in commands.calls[-1]["command"]
    assert all(str(path) in commands.calls[-1]["command"] for path in result["summaries"])
    directory, _, _ = gpu_job.build_jobs(args)
    assert json.loads((directory / "plan/dynamic/result.json").read_text()) == result
    plan = json.loads((directory / "plan/dynamic/job.json").read_text())
    world_size = len(gpus.split(","))
    for job in plan["jobs"]:
        assert job["batching"] == {
            "world_size": world_size,
            "micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "global_batch_size": world_size * 4,
        }
        assert "global_batch_size" not in job["config"]["training"]
    with pytest.raises(ValueError, match="already exists"):
        gpu_job.run_job(args)


@pytest.mark.parametrize("method", ["memory_change", "information_loss", "dynamic"])
def test_auto_from_external_checkpoint_needs_only_qa_and_chains_warmup_into_policy(
    tmp_path, monkeypatch, method
):
    checkpoint = external_checkpoint(tmp_path)
    args = arguments(tmp_path, "--method", method, "--init-checkpoint", str(checkpoint))
    args.qa_data.mkdir()
    for filename in ("train.jsonl", "dev.jsonl", "test.jsonl", "preparation.json"):
        (args.qa_data / filename).write_text("")
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    result = gpu_job.run_job(args)

    methods = ("memory_change", "information_loss") if method == "dynamic" else (method,)
    assert result["status"] == "finished"
    assert not args.pretrain_data.exists()
    assert len(commands.calls) == 3 * len(methods) + int(len(methods) > 1)
    for index, name in enumerate(methods):
        warmup, policy, evaluation = commands.calls[index * 3 : index * 3 + 3]
        assert warmup["config"].objective.method == policy["config"].objective.method == name
        assert warmup["config"].objective.stage == "warmup"
        assert policy["config"].objective.stage == "policy"
        assert warmup["config"].training.init_checkpoint == str(checkpoint.resolve())
        assert policy["config"].training.init_checkpoint == str(warmup["checkpoint"])
        assert policy["config"].training.init_checkpoint != str(checkpoint.resolve())
        assert "latent_working_memory.v3.evaluate" in evaluation["command"]
        assert value(evaluation["command"], "--checkpoint") == str(policy["checkpoint"])
        assert "--log-to-swanlab" in evaluation["command"]
        prefix = name.replace("_", "-")
        assert result["checkpoints"][f"{prefix}-warmup"] == str(warmup["checkpoint"])
        assert result["checkpoints"][f"{prefix}-policy"] == str(policy["checkpoint"])
    assert len(result["summaries"]) == len(methods)
    assert "dynamic-pretrain" not in result["checkpoints"]
    if len(methods) > 1:
        assert "latent_working_memory.v3.compare" in commands.calls[-1]["command"]


def test_auto_from_missing_checkpoint_fails_before_starting_training(tmp_path, monkeypatch):
    args = arguments(tmp_path, "--init-checkpoint", str(tmp_path / "missing.pt"))
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


def test_shared_pretraining_entry_requires_only_ae_lm_data(tmp_path, monkeypatch):
    args = arguments(tmp_path, "--mode", "full", "--method", "shared_pretrain")
    (args.pretrain_data / "multi").mkdir(parents=True)
    (args.pretrain_data / "preparation.json").write_text("{}")
    for split in ("train", "dev", "test"):
        (args.pretrain_data / "multi" / f"{split}.jsonl").write_text("")
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    result = gpu_job.run_job(args)

    assert len(commands.calls) == 1
    assert result["summaries"] == []
    config = commands.calls[0]["config"]
    assert config.objective.stage == "pretrain"
    assert config.training.experiment_id == "unit-job"
    assert Path(config.training.experiment_dir).name == "shared-pretrain-k64_unit-job"
    assert Path(config.training.output_dir) == Path(config.training.experiment_dir) / "pretrain"
    assert not args.qa_data.exists()
    assert not (args.pretrain_data / "single").exists()


@pytest.mark.parametrize("method", ["icae_single", "icae_multi", "autocompressors"])
def test_baselines_need_single_index_but_not_multi_index(tmp_path, monkeypatch, method):
    args = arguments(tmp_path, "--method", method)
    prepare_dataset_entries(args)
    for path in (args.pretrain_data / "multi").iterdir():
        path.unlink()
    (args.pretrain_data / "multi").rmdir()
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    result = gpu_job.run_job(args)

    assert result["status"] == "finished"
    trained = [call["config"] for call in commands.calls if "config" in call]
    assert trained[0].training.pretrain_data_view == "reconstruction_single"
    assert Path(trained[0].training.dataset_dir) == args.pretrain_data


@pytest.mark.parametrize("missing", ["preparation.json", "single/train.jsonl", "multi/dev.jsonl"])
def test_all_methods_require_both_selected_index_views_before_execution(
    tmp_path, monkeypatch, missing
):
    args = arguments(tmp_path, "--method", "all")
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
    shared = arguments(tmp_path, "--mode", "full", "--method", "shared_pretrain")
    prepare_dataset_entries(shared)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    source_result = gpu_job.run_job(shared)
    source_checkpoint = source_result["checkpoints"]["dynamic-pretrain"]
    source_run = commands.calls[0]["config"]
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
            "--init-checkpoint",
            source_checkpoint,
            run_id=None,
        )
        assert args.run_id is None
        result = gpu_job.run_job(args)
        assert args.run_id == result["experiment_id"] == "unit-job"
        descendants.append(args)
    baseline = arguments(tmp_path, "--mode", "full", "--method", "icae_single")
    gpu_job.run_job(baseline)

    directory = shared.output_root / "capacity_unit-job"
    assert {path.name for path in (directory / "plan").iterdir()} == {
        "shared-pretrain",
        "memory-change",
        "information-loss",
        "icae-single",
    }
    assert {path.name for path in (directory / "train").iterdir()} == {
        "shared-pretrain-k64_unit-job",
        "memory-change-k64_unit-job",
        "information-loss-k64_unit-job",
        "icae-single-k64_unit-job",
    }
    assert source_metadata.read_bytes() == original_source
    for method in ("memory_change", "information_loss"):
        stages = [
            call["config"]
            for call in commands.calls
            if "config" in call
            and call["config"].objective.method == method
            and call["config"].objective.stage != "pretrain"
        ]
        assert [config.objective.stage for config in stages] == ["warmup", "policy"]
        assert len({config.training.experiment_dir for config in stages}) == 1
        assert stages[0].training.init_checkpoint == source_checkpoint
    before = len(commands.calls)
    with pytest.raises(ValueError, match="already exists"):
        gpu_job.run_job(descendants[0])
    assert len(commands.calls) == before
    assert source_metadata.read_bytes() == original_source


@pytest.mark.parametrize("dry_run", [False, True])
def test_explicit_id_cannot_disagree_with_pretraining_source(tmp_path, monkeypatch, dry_run):
    checkpoint = external_checkpoint(tmp_path, experiment_id="original-id")
    args = arguments(
        tmp_path, "--init-checkpoint", str(checkpoint), *(["--dry-run"] if dry_run else [])
    )
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    with pytest.raises(ValueError, match="differs from source experiment_id"):
        gpu_job.run_job(args)
    assert not commands.calls
    assert not args.output_root.exists()


@pytest.mark.parametrize(
    "method,stage",
    [("memory_change", "warmup"), ("information_loss", "policy"), ("icae_single", "pretrain")],
)
def test_initialization_must_come_from_shared_dynamic_pretraining(tmp_path, method, stage):
    checkpoint = external_checkpoint(tmp_path, method=method, stage=stage)
    args = arguments(tmp_path, "--dry-run", "--init-checkpoint", str(checkpoint))
    with pytest.raises(ValueError, match="requires a shared dynamic pretraining run"):
        gpu_job.run_job(args)
    assert not args.output_root.exists()


def test_real_initialization_requires_source_sidecar_even_with_explicit_id(tmp_path):
    checkpoint = external_checkpoint(tmp_path)
    (checkpoint.parent.parent / "run.json").unlink()
    args = arguments(tmp_path, "--init-checkpoint", str(checkpoint))
    with pytest.raises(FileNotFoundError, match="source run.json required"):
        gpu_job.run_job(args)
    assert not args.output_root.exists()


@pytest.mark.parametrize("identity", [None, "../invalid"])
def test_initialization_requires_valid_source_experiment_id(tmp_path, identity):
    checkpoint = external_checkpoint(tmp_path, experiment_id=identity)
    args = arguments(tmp_path, "--dry-run", "--init-checkpoint", str(checkpoint))
    with pytest.raises(ValueError, match="no valid experiment_id"):
        gpu_job.run_job(args)
    assert not args.output_root.exists()


@pytest.mark.parametrize("target", ["stage", "plan", "evaluation", "compare"])
def test_existing_owned_output_is_rejected_without_touching_it(tmp_path, monkeypatch, target):
    args = arguments(tmp_path)
    directory, _, jobs = gpu_job.build_jobs(args)
    evaluated = next(job for job in jobs if job.evaluate)
    path = {
        "stage": Path(jobs[0].config.training.output_dir),
        "plan": directory / "plan" / "dynamic",
        "evaluation": directory
        / "eval"
        / Path(evaluated.config.training.experiment_dir).name
        / "policy",
        "compare": directory / "compare" / "dynamic",
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


def test_full_execution_starts_a_new_unbounded_run_without_smoke_checkpoint(tmp_path, monkeypatch):
    smoke = arguments(tmp_path, "--method", "shared_pretrain")
    prepare_dataset_entries(smoke)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    short_result = gpu_job.run_job(smoke)
    full = arguments(tmp_path, "--mode", "full", "--method", "shared_pretrain")
    full_result = gpu_job.run_job(full)
    assert set(short_result["checkpoints"].values()).isdisjoint(full_result["checkpoints"].values())
    full_call = commands.calls[-1]
    assert "--stop-after-steps" not in full_call["command"]
    assert full_call["config"].training.init_checkpoint is None
    assert full_call["config"].training.max_train_samples is None
    assert full_call["config"].training.max_dev_samples is None


@pytest.mark.parametrize("fail_at", [1, 4])
def test_failed_training_or_evaluation_stops_the_remaining_graph(tmp_path, monkeypatch, fail_at):
    args = arguments(tmp_path)
    prepare_dataset_entries(args)
    commands = Commands(fail_at=fail_at)
    monkeypatch.setattr(gpu_job, "execute", commands)
    with pytest.raises(subprocess.CalledProcessError):
        gpu_job.run_job(args)
    assert len(commands.calls) == fail_at
    directory, _, _ = gpu_job.build_jobs(args)
    failure = json.loads((directory / "plan/dynamic/result.json").read_text())
    assert failure["status"] == "failed"
    assert len(failure["checkpoints"]) == (0 if fail_at == 1 else 3)
    assert not failure["summaries"]


def test_training_must_save_the_reported_checkpoint_before_dependent_stage(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    prepare_dataset_entries(args)
    commands = Commands(omit_checkpoint=True)
    monkeypatch.setattr(gpu_job, "execute", commands)
    with pytest.raises(FileNotFoundError, match="reported checkpoint"):
        gpu_job.run_job(args)
    assert len(commands.calls) == 1


def test_dry_run_does_not_access_datasets_or_execute_or_create_artifacts(
    tmp_path, monkeypatch, capsys
):
    args = arguments(tmp_path, "--dry-run", "--method", "all")

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not execute commands or check dataset files")

    monkeypatch.setattr(gpu_job, "execute", forbidden)
    monkeypatch.setattr(Path, "is_file", forbidden)
    plan = gpu_job.run_job(args)
    assert len(plan["jobs"]) == 10
    assert json.loads(capsys.readouterr().out) == plan
    assert list(tmp_path.iterdir()) == []


def test_external_checkpoint_dry_run_does_not_read_or_check_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "unavailable shared pretraining.pt"
    args = arguments(tmp_path, "--dry-run", "--init-checkpoint", str(checkpoint))

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not inspect checkpoints, datasets or credentials")

    monkeypatch.setattr(Path, "is_file", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(gpu_job, "execute", forbidden)
    monkeypatch.setattr(gpu_job, "swanlab_api_key", forbidden)

    plan = gpu_job.run_job(args)

    assert len(plan["jobs"]) == 4
    assert all(job["config"]["objective"]["stage"] in {"warmup", "policy"} for job in plan["jobs"])
    assert plan["jobs"][0]["config"]["training"]["init_checkpoint"] == str(checkpoint)
    assert plan["jobs"][2]["config"]["training"]["init_checkpoint"] == str(checkpoint)
    assert list(tmp_path.iterdir()) == []


def test_preview_inherits_id_from_lightweight_sidecar_without_loading_weights(
    tmp_path, monkeypatch
):
    checkpoint = external_checkpoint(tmp_path, experiment_id="source-id")
    checkpoint.unlink()
    args = arguments(tmp_path, "--dry-run", "--init-checkpoint", str(checkpoint), run_id=None)

    def forbidden(*args, **kwargs):
        raise AssertionError("preview must not read model weights or start processes")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(gpu_job, "execute", forbidden)
    monkeypatch.setattr(gpu_job, "swanlab_api_key", forbidden)
    plan = gpu_job.run_job(args)
    assert plan["experiment_id"] == args.run_id == "source-id"
    assert all(job["config"]["training"]["experiment_id"] == "source-id" for job in plan["jobs"])
    assert not args.output_root.exists()


def test_preview_without_local_source_requires_explicit_id(tmp_path):
    args = arguments(
        tmp_path, "--dry-run", "--init-checkpoint", str(tmp_path / "unavailable.pt"), run_id=None
    )
    with pytest.raises(ValueError, match="requires explicit --run-id"):
        gpu_job.run_job(args)
    assert not args.output_root.exists()


def test_planning_imports_no_model_or_tracking_runtime(tmp_path):
    code = (
        "import sys; from latent_working_memory.v3.gpu_job import main; "
        f"main(['--dry-run', '--output-root', {str(tmp_path)!r}]); "
        "assert not {'torch', 'transformers', 'swanlab'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("dotenv", [None, "SWANLAB_API_KEY=project-test-key\n"])
def test_selected_key_reaches_all_stages_without_being_persisted(
    tmp_path, monkeypatch, capsys, dotenv
):
    repository = Path.cwd()
    monkeypatch.setattr(gpu_job, "load_experiment", lambda path: load_experiment(repository / path))
    monkeypatch.setattr(gpu_job, "swanlab_api_key", tracking_credentials.swanlab_api_key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWANLAB_API_KEY", "terminal-test-key")
    if dotenv is not None:
        (tmp_path / ".env").write_text(dotenv)
    args = arguments(tmp_path)
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


@pytest.mark.parametrize(
    "tracking,dry_run", [("online", False), ("disabled", False), ("online", True)]
)
def test_missing_key_stops_online_job_before_creating_artifacts(
    tmp_path, monkeypatch, tracking, dry_run
):
    repository = Path.cwd()
    monkeypatch.setattr(gpu_job, "load_experiment", lambda path: load_experiment(repository / path))
    monkeypatch.setattr(gpu_job, "swanlab_api_key", tracking_credentials.swanlab_api_key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    args = arguments(tmp_path, "--tracking", tracking, *(["--dry-run"] if dry_run else []))
    prepare_dataset_entries(args)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    if tracking == "online" and not dry_run:
        with pytest.raises(ValueError, match="nonempty SWANLAB_API_KEY"):
            gpu_job.run_job(args)
        assert not commands.calls and not args.output_root.exists()
    else:
        gpu_job.run_job(args)
        assert bool(commands.calls) != dry_run


@pytest.mark.parametrize("exit_code", [0, 7])
def test_shell_uses_configured_repo_preserves_arguments_and_returns_process_status(
    tmp_path, exit_code
):
    source = Path("src/latent_working_memory/v3/scripts/run_gpu.sh").read_text()
    repository = tmp_path / "repository with spaces"
    repository.mkdir()
    script = tmp_path / "run_gpu.sh"
    script.write_text(
        source.replace(
            'LWM_REPO_DIR="/data/zhangdw12/percyw/latent_working_memory"',
            f"LWM_REPO_DIR={shlex.quote(str(repository))}",
            1,
        )
    )
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
    environment["LWM_REPO_DIR"] = str(tmp_path / "ignored environment path")
    result = subprocess.run(
        ["bash", str(script), *options],
        cwd=tmp_path,
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
    assert list(repository.iterdir()) == []


def test_execute_propagates_real_child_failure_and_keeps_the_log(tmp_path):
    log = tmp_path / "child.log"
    command = [sys.executable, "-c", "print('visible failure'); raise SystemExit(3)"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        gpu_job.execute(command, dict(os.environ), log)
    assert error.value.returncode == 3
    assert log.read_text().strip() == "visible failure"
