from copy import deepcopy
from dataclasses import replace
import random
from types import SimpleNamespace

import numpy as np
import torch

from latent_working_memory.v3 import runtime, tracking
from latent_working_memory.v3.evaluate import _training_run_directory
from latent_working_memory.v3.runtime import load_initialization, prepare_training, train_loop
from latent_working_memory.v3.tracking import method_tracking_run, update_method_tracking
from latent_working_memory.v4.checkpoint import capture_rng
from .test_runtime import experiment_config, make_engine, make_splits, _run
import json

import pytest
from swanlab.sdk.internal.run.components.config import Config as SwanLabConfig

from latent_working_memory.v3.tracking import (
    TRAINING_METRICS,
    DEV_QA_METRICS,
    STAGE_NUMBERS,
    configure_training_metrics,
    training_metrics,
)


def test_training_panels_are_registered_before_log_and_use_optimizer_steps():
    class Run:
        def __init__(self):
            self.definitions = []

        def define_metric(self, name, **kwargs):
            self.definitions.append((name, kwargs))

    run = Run()
    configure_training_metrics(run, "dynamic", "pretrain")
    assert tuple(name for name, _ in run.definitions) == (
        *TRAINING_METRICS,
        "train/stage",
        "train/ae_lm_loss",
        "dev/ae_lm_loss",
    )
    assert all(options["x_axis"] == "_step" for _, options in run.definitions)
    assert all(options["section_name"] == name.split("/")[0] for name, options in run.definitions)
    assert not any("hidden" in options for _, options in run.definitions)
    configure_training_metrics(run, "memory_change", "warmup")
    assert tuple(name for name, _ in run.definitions[-2:]) == DEV_QA_METRICS
    configure_training_metrics(None, "dynamic", "pretrain")


def test_training_publishes_only_current_measured_core_values():
    record = {
        "step": 9,
        "stage": "warmup",
        "train/epoch": 0.45,
        "train/loss": 2.1,
        "train/grad_norm": 0.5,
        "train/slots_final": 128,
        "train/qa_old_count": 8,
        "train/gate_g": 0.01,
        "train/write_calls": 4,
        "dev/loss": None,
        "resources/peak_memory_allocated_bytes": 3 * 1024**3,
        "resources/optimizer_step_seconds": 4.2,
    }
    assert training_metrics(record, "memory_change") == {
        "train/stage": 2,
        "train/epoch": 0.45,
        "train/qa_loss": 2.1,
        "train/grad_norm": 0.5,
        "train/slots_final": 128,
        "resources/peak_memory_allocated_gib": 3,
        "resources/optimizer_step_seconds": 4.2,
    }
    assert "dev/loss" not in training_metrics(
        {"stage": "warmup", "train/loss": 1}, "memory_change"
    )
    assert record["train/qa_old_count"] == 8


def test_dev_qa_curves_skip_unmeasured_groups_and_keep_current_values():
    record = {
        "stage": "warmup",
        "dev/qa_old_nll": 0.0,
        "dev/qa_old_count": 0,
        "dev/qa_new_nll": 1.5,
        "dev/qa_new_count": 4,
    }
    assert training_metrics(record, "memory_change") == {"train/stage": 2, "dev/qa_new_nll": 1.5}
    record.update({"stage": "policy", "dev/qa_old_nll": 0.75, "dev/qa_old_count": 8})
    assert training_metrics(record, "memory_change") == {
        "train/stage": 3,
        "dev/qa_old_nll": 0.75,
        "dev/qa_new_nll": 1.5,
    }
    assert training_metrics({"stage": "policy"}, "memory_change") == {"train/stage": 3}


@pytest.mark.parametrize(
    "method,stage,metric",
    [
        ("dynamic", "pretrain", "ae_lm_loss"),
        ("icae_single", "pretrain", "ae_lm_loss"),
        ("icae_multi", "pretrain", "ae_lm_loss"),
        ("autocompressors", "pretrain", "lm_loss"),
        ("autocompressors", "lm", "lm_loss"),
        ("memory_change", "warmup", "qa_loss"),
        ("information_loss", "policy", "qa_loss"),
        ("icae_single", "qa", "qa_loss"),
    ],
)
def test_training_losses_are_separated_by_method_and_stage_without_empty_metrics(
    method, stage, metric
):
    values = training_metrics({"stage": stage, "train/loss": 2, "dev/loss": 3}, method)
    assert values == {
        "train/stage": STAGE_NUMBERS[stage],
        f"train/{metric}": 2,
        f"dev/{metric}": 3,
    }
    definitions = []
    run = SimpleNamespace(define_metric=lambda name, **options: definitions.append(name))
    configure_training_metrics(run, method, stage)
    assert f"train/{metric}" in definitions
    assert f"dev/{metric}" in definitions
    assert not any(name.endswith("_loss") and name not in values for name in definitions)


@pytest.fixture
def recorded_swanlab(monkeypatch):
    class Run:
        def __init__(self, record):
            self.record = record
            self.id = record["id"]
            self.url = record["url"]
            self.config = record["config"]

        def __enter__(self):
            return self

        def __exit__(self, kind, exception, traceback):
            self.record["state"] = "FINISHED" if kind is None else "CRASHED"

        def define_metric(self, name, **options):
            self.record["definitions"].append((name, options))

        def log(self, values, step):
            self.record["logs"].append((step, values))

    class Service:
        def __init__(self):
            self.runs = {}
            self.initializations = []
            self.api_reads = []

        def init(self, **settings):
            self.initializations.append(deepcopy(settings))
            run_id = settings["id"]
            if run_id is None:
                assert settings["resume"] == "never"
                run_id = f"run-{len(self.runs) + 1}"
                self.runs[run_id] = {
                    "id": run_id,
                    "url": f"https://swanlab.cn/@owner/{settings['project']}/runs/{run_id}",
                    "logs": [],
                    "definitions": [],
                }
            else:
                assert settings["resume"] == "must" and run_id in self.runs
            record = self.runs[run_id]
            sdk_config = SwanLabConfig()
            sdk_config.update(settings["config"])
            record.update(state="RUNNING", config=sdk_config, name=settings["name"])
            return Run(record)

        def api(self, api_key):
            assert api_key == "test-api-key"
            return SimpleNamespace(run=self.remote)

        def remote(self, path):
            self.api_reads.append(path)
            record = self.runs[path.rsplit("/", 1)[1]]
            return SimpleNamespace(
                state=record["state"],
                profile={
                    "config": {name: {"value": value} for name, value in record["config"].items()}
                },
            )

    service = Service()
    monkeypatch.setattr(tracking.swanlab, "init", service.init)
    monkeypatch.setattr(tracking.swanlab, "Api", service.api)
    monkeypatch.setattr(tracking.swanlab, "Settings", lambda **settings: settings)
    monkeypatch.setattr(runtime, "swanlab_api_key", lambda: "test-api-key")
    return service


def serialized_config(values):
    config = SwanLabConfig()
    config.update(values)
    return dict(config)


def online_config(root, stage, method=None, init=None):
    config = experiment_config(root, stage, method=method, init=init)
    return replace(
        config,
        training=replace(
            config.training,
            swanlab_project="test-project",
            group="trial-group",
            tags=("study:main",),
        ),
    )


def run_stage(config, steps, prefix="qa"):
    engine = make_engine(config)
    data = make_splits(prefix)
    initialization = (
        load_initialization(config.training.init_checkpoint, engine.model, config)
        if config.training.init_checkpoint is not None
        else None
    )
    run = _run(config, engine, data, initialization)
    stage = prepare_training(config, engine, data, run, stop_after_steps=steps)
    with runtime._tracking_context(config, engine, run) as active:
        result = train_loop(
            config, engine, data, run, stage, stop_after_steps=steps, tracking=active
        )
    return result, run


@pytest.mark.parametrize("explicit_name", [False, True])
def test_two_dynamic_methods_share_one_source_and_keep_one_run_per_method(
    tmp_path, recorded_swanlab, explicit_name
):
    source_root = tmp_path / (
        "dynamic-pretrain-k64" if explicit_name else "dynamic-pretrain-k64_trial"
    )
    source = online_config(source_root, "pretrain")
    if explicit_name:
        source = replace(
            source,
            training=replace(source.training, experiment_name="dynamic-pretrain-k64_trial"),
        )
    source_result, source_run = run_stage(source, 2, "pretrain")
    source_identity = json.loads((source_root / "swanlab.json").read_text())
    assert "method:dynamic-pretrain" in source_identity["tags"]
    assert recorded_swanlab.runs[source_identity["id"]]["name"] == "dynamic-pretrain-k64_trial"
    sources = []
    for method in ("memory_change", "information_loss"):
        method_name = method.replace("_", "-") if explicit_name else method
        root = tmp_path / (f"{method_name}-k64" if explicit_name else f"{method_name}-k64_trial")
        cloud_name = f"{method_name}-k64_trial"
        warmup = online_config(root, "warmup", method, source_result["checkpoint"])
        if explicit_name:
            warmup = replace(warmup, training=replace(warmup.training, experiment_name=cloud_name))
        warmup_result, warmup_run = run_stage(warmup, 2)
        identity_bytes = (root / "swanlab.json").read_bytes()
        identity = json.loads(identity_bytes)
        before = json.loads((root / "experiment.json").read_text())
        policy = online_config(root, "policy", method, warmup_result["checkpoint"])
        if explicit_name:
            policy = replace(policy, training=replace(policy.training, experiment_name=cloud_name))
        policy_result, policy_run = run_stage(policy, 2)
        combined = json.loads((root / "experiment.json").read_text())
        assert before["stages"]["warmup"] == combined["stages"]["warmup"]
        assert list(combined["stages"]) == ["warmup", "policy"]
        assert (root / "swanlab.json").read_bytes() == identity_bytes
        assert not (root / "warmup/swanlab.json").exists()
        assert not (root / "policy/swanlab.json").exists()
        assert policy_result["global_step"] == 4
        cloud = recorded_swanlab.runs[identity["id"]]
        assert cloud["config"] == serialized_config(combined)
        assert cloud["name"] == cloud_name
        assert [step for step, values in cloud["logs"] if "train/qa_loss" in values] == [1, 2, 3, 4]
        assert not any("train/ae_lm_loss" in values for _, values in cloud["logs"])
        assert [(step, values["train/stage"]) for step, values in cloud["logs"]] == [
            (1, 2),
            (2, 2),
            (3, 3),
            (4, 3),
        ]
        assert [(step, values["train/epoch"]) for step, values in cloud["logs"]] == [
            (1, 0.4),
            (2, 0.8),
            (3, 0.4),
            (4, 0.8),
        ]
        assert all("train/stages" not in values for _, values in cloud["logs"])
        assert "pretraining_sources" not in json.dumps(dict(cloud["config"]))
        assert "pretrain-train-0" not in json.dumps(dict(cloud["config"]))
        assert "test-api-key" not in json.dumps(dict(cloud["config"]))
        assert all(not tag.endswith((":warmup", ":policy")) for tag in identity["tags"])
        assert warmup_run["pretraining"] == policy_run["pretraining"]
        sources.append(policy_run["pretraining"])
    assert len(recorded_swanlab.runs) == 3
    assert len(recorded_swanlab.initializations) == 5
    assert sources[0] == sources[1]
    assert sources[0]["run_id"] == source_identity["id"]
    assert sources[0]["run_url"] == source_identity["url"]
    assert sources[0]["run_name"] == "dynamic-pretrain-k64_trial"
    assert sources[0]["run_dir"] == str(source_root.resolve())
    assert sources[0]["step"] == 2


@pytest.mark.parametrize("source_method", ["memory_change", "information_loss"])
def test_existing_shared_pretrain_identity_resumes_and_initializes_new_dynamic_method(
    tmp_path, recorded_swanlab, monkeypatch, source_method
):
    root = tmp_path / "shared-pretrain-k64_trial"
    source = online_config(root, "pretrain", source_method)
    # 按改名前的实现生成真实 checkpoint、本地身份和同一云端记录。
    with monkeypatch.context() as legacy:
        legacy.setattr(tracking, "tracking_method", lambda config, previous=None: "shared-pretrain")
        result, record = run_stage(source, 2, "pretrain")
    assert "experiment_name" not in record["config"]["training"]
    original_manifest = (root / "experiment.json").read_bytes()
    original_identity = (root / "swanlab.json").read_bytes()
    identity = json.loads(original_identity)
    cloud = recorded_swanlab.runs[identity["id"]]
    original_cloud_config = deepcopy(dict(cloud["config"]))
    assert original_cloud_config["method"] == "shared-pretrain"
    assert (
        original_cloud_config["stages"]["pretrain"]["config"]["objective"]["method"]
        == source_method
    )
    assert "method:shared-pretrain" in identity["tags"]
    checkpoint = runtime.read_checkpoint(result["checkpoint"])
    assert _training_run_directory(source, checkpoint["run"]) == root

    engine, data = make_engine(source), make_splits("pretrain")
    stage = prepare_training(
        source, engine, data, record, resume=result["checkpoint"], stop_after_steps=3
    )
    with method_tracking_run(source, record, engine.device, api_key="test-api-key") as active:
        resumed_result = train_loop(
            source, engine, data, record, stage, stop_after_steps=3, tracking=active
        )
    assert resumed_result["global_step"] == 3
    resumed_checkpoint = runtime.read_checkpoint(resumed_result["checkpoint"])
    assert resumed_checkpoint["run"]["config"]["objective"]["method"] == source_method
    assert (root / "experiment.json").read_bytes() == original_manifest
    assert (root / "swanlab.json").read_bytes() == original_identity
    assert dict(cloud["config"]) == original_cloud_config
    resumed = recorded_swanlab.initializations[-1]
    assert resumed["id"] == identity["id"]
    assert resumed["resume"] == "must"
    assert resumed["name"] == root.name
    assert resumed["tags"] == identity["tags"]
    assert len(recorded_swanlab.runs) == 1

    target = online_config(tmp_path / "memory-change-k64", "warmup", init=result["checkpoint"])
    target = replace(
        target,
        training=replace(target.training, experiment_name="memory-change-k64_trial"),
    )
    _, target_record = run_stage(target, 1)
    provenance = target_record["pretraining"]
    assert provenance["run_name"] == "shared-pretrain-k64_trial"
    assert provenance["run_dir"] == str(root.resolve())
    assert provenance["run_id"] == identity["id"]
    assert provenance["run_url"] == identity["url"]
    assert provenance["checkpoint"] == result["checkpoint"]
    assert (root / "experiment.json").read_bytes() == original_manifest
    assert (root / "swanlab.json").read_bytes() == original_identity
    assert dict(cloud["config"]) == original_cloud_config


@pytest.mark.parametrize("online", [False, True])
def test_live_method_run_continues_two_stages_without_reinitializing(
    tmp_path, recorded_swanlab, online
):
    root = tmp_path / "icae-single-k64_trial"
    config = online_config(root, "pretrain", "icae_single")
    if not online:
        config = replace(config, training=replace(config.training, swanlab_project=None))
    engine = make_engine(config)
    pretrain_data = make_splits("pretrain")
    run = _run(config, engine, pretrain_data)
    stage = prepare_training(config, engine, pretrain_data, run, stop_after_steps=2)
    with runtime._tracking_context(config, engine, run) as active:
        result = train_loop(
            config,
            engine,
            pretrain_data,
            run,
            stage,
            stop_after_steps=2,
            tracking=active,
        )
        before = json.loads((root / "experiment.json").read_text())
        target = online_config(root, "qa", "icae_single", result["checkpoint"])
        if not online:
            target = replace(target, training=replace(target.training, swanlab_project=None))
        qa_data = make_splits("qa")
        initialization = load_initialization(result["checkpoint"], engine.model, target)
        qa_run = _run(target, engine, qa_data, initialization)
        stage = prepare_training(target, engine, qa_data, qa_run, stop_after_steps=2)
        update_method_tracking(target, qa_run, active)
        combined = json.loads((root / "experiment.json").read_text())
        assert list(combined["stages"]) == ["pretrain", "qa"]
        assert combined["stages"]["pretrain"] == before["stages"]["pretrain"]
        assert recorded_swanlab.api_reads == []
        assert len(recorded_swanlab.initializations) == int(online)
        if online:
            assert active.record["state"] == "RUNNING"
            assert active.config == serialized_config(combined)
        result = train_loop(
            target, engine, qa_data, qa_run, stage, stop_after_steps=2, tracking=active
        )
        assert result["global_step"] == 4
    if online:
        identity = json.loads((root / "swanlab.json").read_text())
        remote = recorded_swanlab.runs[identity["id"]]
        assert remote["state"] == "FINISHED"
        assert remote["config"] == serialized_config(combined)
        assert [step for step, values in remote["logs"] if "train/ae_lm_loss" in values] == [1, 2]
        assert [step for step, values in remote["logs"] if "train/qa_loss" in values] == [3, 4]
        definitions = {name for name, _ in remote["definitions"]}
        assert {"train/ae_lm_loss", "train/qa_loss"} <= definitions
        assert [(step, values["train/stage"]) for step, values in remote["logs"]] == [
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
        ]
        assert all("train/stages" not in values for _, values in remote["logs"])
    else:
        assert not (root / "swanlab.json").exists()


@pytest.mark.parametrize("invalid", ["method", "model", "checkpoint", "step_offset"])
def test_live_stage_update_rejects_invalid_transition_before_mutation(
    tmp_path, recorded_swanlab, invalid
):
    root = tmp_path / "icae-single-k64_trial"
    config = online_config(root, "pretrain", "icae_single")
    engine, data = make_engine(config), make_splits("pretrain")
    run = _run(config, engine, data)
    stage = prepare_training(config, engine, data, run, stop_after_steps=1)
    with runtime._tracking_context(config, engine, run) as active:
        result = train_loop(config, engine, data, run, stage, stop_after_steps=1, tracking=active)
        target = online_config(root, "qa", "icae_single", result["checkpoint"])
        initialization = load_initialization(result["checkpoint"], engine.model, target)
        qa_run = _run(target, engine, make_splits("qa"), initialization)
        before = (root / "experiment.json").read_bytes()
        remote_before = deepcopy(dict(active.config))
        if invalid == "method":
            target = replace(target, objective=replace(target.objective, method="icae_multi"))
        elif invalid == "model":
            qa_run["config"]["model"]["memory_slots"] += 1
        elif invalid == "checkpoint":
            qa_run["initialization"]["checkpoint"] = str(tmp_path / "unrelated.pt")
        else:
            qa_run["step_offset"] = 0
        expected = (
            "identity or model" if invalid in {"method", "model"} else "last completed checkpoint"
        )
        with pytest.raises(ValueError, match=expected):
            update_method_tracking(target, qa_run, active)
        assert (root / "experiment.json").read_bytes() == before
        assert active.config == remote_before
        assert active.record["state"] == "RUNNING"
        assert recorded_swanlab.api_reads == []
        assert len(recorded_swanlab.initializations) == 1


@pytest.mark.parametrize(
    "state,allowed",
    [
        ("FINISHED", True),
        ("CRASHED", True),
        ("ABORTED", True),
        ("RUNNING", False),
        ("OFFLINE", False),
    ],
)
def test_same_stage_checkpoint_resume_obeys_session_state(
    tmp_path, recorded_swanlab, state, allowed
):
    root = tmp_path / "dynamic-pretrain-k64_trial"
    config = online_config(root, "pretrain")
    result, run = run_stage(config, 1, "pretrain")
    identity = json.loads((root / "swanlab.json").read_text())
    recorded_swanlab.runs[identity["id"]]["state"] = state
    engine = make_engine(config)
    data = make_splits()
    stage = prepare_training(
        config, engine, data, run, resume=result["checkpoint"], stop_after_steps=2
    )
    if not allowed:
        with pytest.raises(ValueError, match="session state"):
            with runtime._tracking_context(config, engine, run) as active:
                train_loop(
                    config,
                    engine,
                    data,
                    run,
                    stage,
                    stop_after_steps=2,
                    tracking=active,
                )
        assert len(recorded_swanlab.initializations) == 1
        return
    with runtime._tracking_context(config, engine, run) as active:
        result = train_loop(
            config,
            engine,
            data,
            run,
            stage,
            stop_after_steps=2,
            tracking=active,
        )
    assert result["global_step"] == 2
    assert recorded_swanlab.initializations[-1]["id"] == identity["id"]
    assert recorded_swanlab.initializations[-1]["resume"] == "must"


@pytest.mark.parametrize("state", ["RUNNING", "OFFLINE"])
def test_cross_stage_rejects_active_or_offline_session_without_changing_config(
    tmp_path, recorded_swanlab, state
):
    root = tmp_path / "icae-single-k64_trial"
    config = online_config(root, "pretrain", "icae_single")
    result, _ = run_stage(config, 1, "pretrain")
    before = (root / "experiment.json").read_bytes()
    identity = json.loads((root / "swanlab.json").read_text())
    remote = recorded_swanlab.runs[identity["id"]]
    remote["state"] = state
    target = online_config(root, "qa", "icae_single", result["checkpoint"])
    engine = make_engine(target)
    initialization = load_initialization(result["checkpoint"], engine.model, target)
    run = _run(target, engine, make_splits("qa"), initialization)
    with pytest.raises(ValueError, match="session state"):
        with method_tracking_run(target, run, engine.device, api_key="test-api-key"):
            pass
    assert (root / "experiment.json").read_bytes() == before
    assert len(recorded_swanlab.initializations) == 1
    remote["state"] = "FINISHED"
    with method_tracking_run(target, run, engine.device, api_key="test-api-key"):
        pass
    assert list(remote["config"]["stages"]) == ["pretrain", "qa"]


@pytest.mark.parametrize("state", ["CRASHED", "ABORTED"])
def test_cross_stage_recovers_failure_before_successor_registration(
    tmp_path, recorded_swanlab, state
):
    root = tmp_path / "icae-single-k64_trial"
    config = online_config(root, "pretrain", "icae_single")
    engine, data = make_engine(config), make_splits("pretrain")
    run = _run(config, engine, data)
    stage = prepare_training(config, engine, data, run, stop_after_steps=1)
    with pytest.raises(RuntimeError, match="QA data unavailable"):
        with runtime._tracking_context(config, engine, run) as active:
            result = train_loop(
                config, engine, data, run, stage, stop_after_steps=1, tracking=active
            )
            raise RuntimeError("QA data unavailable before stage registration")
    identity = json.loads((root / "swanlab.json").read_text())
    remote = recorded_swanlab.runs[identity["id"]]
    assert remote["state"] == "CRASHED"
    assert list(remote["config"]["stages"]) == ["pretrain"]
    assert list(json.loads((root / "experiment.json").read_text())["stages"]) == ["pretrain"]
    remote["state"] = state
    target = online_config(root, "qa", "icae_single", result["checkpoint"])
    resumed, _ = run_stage(target, 1)
    assert resumed["global_step"] == 2
    assert remote["state"] == "FINISHED"
    assert list(remote["config"]["stages"]) == ["pretrain", "qa"]
    assert len(recorded_swanlab.runs) == 1
    assert recorded_swanlab.initializations[-1]["id"] == identity["id"]
    assert recorded_swanlab.initializations[-1]["resume"] == "must"


def test_method_manifest_rejects_source_replacement_and_step_rewind_without_cloud_changes(
    tmp_path, recorded_swanlab
):
    source = online_config(tmp_path / "dynamic-pretrain-k64_trial", "pretrain")
    source_result, _ = run_stage(source, 2, "pretrain")
    root = tmp_path / "memory-change-k64_trial"
    warmup = online_config(root, "warmup", init=source_result["checkpoint"])
    result, warmup_run = run_stage(warmup, 2)
    before = (root / "experiment.json").read_bytes()
    policy = online_config(root, "policy", init=result["checkpoint"])
    engine = make_engine(policy)
    initialization = load_initialization(result["checkpoint"], engine.model, policy)
    policy_run = _run(policy, engine, make_splits("qa"), initialization)
    invalid = deepcopy(policy_run)
    invalid["pretraining"]["checkpoint"] = "different-source.pt"
    with pytest.raises(ValueError, match="original pretraining source"):
        with method_tracking_run(policy, invalid, engine.device, api_key="test-api-key"):
            pass
    invalid = deepcopy(policy_run)
    invalid["step_offset"] = 1
    with pytest.raises(ValueError, match="last completed checkpoint"):
        with method_tracking_run(policy, invalid, engine.device, api_key="test-api-key"):
            pass
    assert (root / "experiment.json").read_bytes() == before
    assert len(recorded_swanlab.initializations) == 2
    run_stage(policy, 1)
    with pytest.raises(ValueError, match="earlier stage"):
        with method_tracking_run(warmup, warmup_run, engine.device, api_key="test-api-key"):
            pass
    assert len(recorded_swanlab.initializations) == 3


def test_tracking_setup_preserves_random_state_in_api_and_sdk_calls(
    tmp_path, monkeypatch, recorded_swanlab
):
    root = tmp_path / "dynamic-pretrain-k64_trial"
    config = online_config(root, "pretrain")
    _, record = run_stage(config, 1, "pretrain")
    original_remote, original_init = recorded_swanlab.remote, recorded_swanlab.init

    def disturb():
        random.random()
        np.random.random()
        torch.rand(())

    def remote(path):
        disturb()
        return original_remote(path)

    def initialize(**settings):
        disturb()
        return original_init(**settings)

    monkeypatch.setattr(recorded_swanlab, "remote", remote)
    monkeypatch.setattr(tracking.swanlab, "init", initialize)
    before = capture_rng(torch.device("cpu"))
    with tracking.method_tracking_run(config, record, torch.device("cpu"), api_key="test-api-key"):
        assert random.getstate() == before["python"]
        assert np.array_equal(np.random.get_state()[1], before["numpy"][1])
        assert torch.equal(torch.get_rng_state(), before["torch"])


def test_sdk_serialized_empty_values_resume_without_changing_local_semantics(
    tmp_path, recorded_swanlab
):
    root = tmp_path / "icae-single-k64_trial"
    config = online_config(root, "pretrain", "icae_single")
    _, record = run_stage(config, 1, "pretrain")
    original = (root / "experiment.json").read_bytes()
    local = json.loads(original)
    identity = json.loads((root / "swanlab.json").read_text())
    remote = recorded_swanlab.runs[identity["id"]]
    assert local["pretraining"] is local["resolved_model_revision"] is None
    assert remote["config"]["pretraining"] == remote["config"]["resolved_model_revision"] == {}
    assert remote["config"]["model"]["revision"] is None
    assert local["model"]["lora_target_modules"] == remote["config"]["model"]["lora_target_modules"]
    with method_tracking_run(config, record, torch.device("cpu"), api_key="test-api-key"):
        pass
    assert (root / "experiment.json").read_bytes() == original
    assert len(recorded_swanlab.runs) == 1
    assert recorded_swanlab.initializations[-1]["id"] == identity["id"]


@pytest.mark.parametrize(
    "change,field",
    [
        ("missing", "pretraining"),
        ("source", "pretraining"),
        ("nested_null", "model"),
        ("slots", "model"),
        ("learning_rate", "stages"),
    ],
)
def test_serialized_config_comparison_still_rejects_missing_or_changed_fields(
    tmp_path, recorded_swanlab, change, field
):
    root = tmp_path / "icae-single-k64_trial"
    config = online_config(root, "pretrain", "icae_single")
    _, record = run_stage(config, 1, "pretrain")
    original = (root / "experiment.json").read_bytes()
    identity = json.loads((root / "swanlab.json").read_text())
    remote = recorded_swanlab.runs[identity["id"]]["config"]
    if change == "missing":
        del remote["pretraining"]
    elif change == "source":
        remote["pretraining"] = {"checkpoint": "unexpected-source.pt"}
    elif change == "nested_null":
        remote["model"]["revision"] = {}
    elif change == "slots":
        remote["model"]["memory_slots"] += 1
    else:
        remote["stages"]["pretrain"]["config"]["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match=f"saved method experiment: {field}$"):
        with method_tracking_run(config, record, torch.device("cpu"), api_key="test-api-key"):
            pass
    assert (root / "experiment.json").read_bytes() == original
    assert len(recorded_swanlab.initializations) == 1


@pytest.mark.parametrize("legacy_manifest", [False, True])
def test_cloud_annotations_remain_outside_local_experiment_contract(
    tmp_path, recorded_swanlab, legacy_manifest
):
    root = tmp_path / "icae-single-k64_trial"
    config = online_config(root, "pretrain", "icae_single")
    _, record = run_stage(config, 1, "pretrain")
    original = (root / "experiment.json").read_bytes()
    identity = json.loads((root / "swanlab.json").read_text())
    cloud = recorded_swanlab.runs[identity["id"]]
    if legacy_manifest:
        # 旧恢复代码曾将云端附加字段混入本地记录；它们不属于方法配置。
        (root / "experiment.json").write_text(json.dumps({**json.loads(original), "note": "old"}))
    for note in ("first note", "revised note", None):
        if note is None:
            del cloud["config"]["note"]
        else:
            cloud["config"]["note"] = note
        with method_tracking_run(config, record, torch.device("cpu"), api_key="test-api-key"):
            pass
        assert (root / "experiment.json").read_bytes() == original
        assert cloud["config"].get("note") == note
    assert len(recorded_swanlab.runs) == 1
