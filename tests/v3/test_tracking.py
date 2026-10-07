from copy import deepcopy
from dataclasses import replace
import random
from types import SimpleNamespace

import numpy as np
import torch

from latent_working_memory.v3 import runtime, tracking
from latent_working_memory.v3.runtime import load_initialization, prepare_training, train_loop
from latent_working_memory.v3.tracking import method_tracking_run, update_method_tracking
from latent_working_memory.v4.checkpoint import capture_rng
from .test_runtime import experiment_config, make_engine, make_splits, _run
import json

import pytest
import swanlab
from swanlab.sdk.internal.run.components.config import Config as SwanLabConfig

from latent_working_memory.v3.tracking import (
    TRAINING_METRICS,
    configure_training_metrics,
    evaluation_media,
    training_metrics,
)
from .test_evaluate import Task, trajectory
from latent_working_memory.v3.evaluate import evaluate


def test_training_panels_are_registered_before_log_and_use_optimizer_steps():
    class Run:
        def __init__(self):
            self.definitions = []

        def define_metric(self, name, **kwargs):
            self.definitions.append((name, kwargs))

    run = Run()
    configure_training_metrics(run, "pretrain")
    assert tuple(name for name, _ in run.definitions) == (
        *TRAINING_METRICS,
        "train/ae_lm_loss",
        "dev/ae_lm_loss",
    )
    assert all(options["x_axis"] == "_step" for _, options in run.definitions)
    assert all(options["section_name"] == name.split("/")[0] for name, options in run.definitions)
    assert not any("hidden" in options for _, options in run.definitions)
    configure_training_metrics(None, "pretrain")


def test_training_publishes_only_current_measured_core_values():
    record = {
        "step": 9,
        "stage": "warmup",
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
    assert training_metrics(record) == {
        "train/qa_loss": 2.1,
        "train/grad_norm": 0.5,
        "train/slots_final": 128,
        "resources/peak_memory_allocated_gib": 3,
        "resources/optimizer_step_seconds": 4.2,
    }
    assert "dev/loss" not in training_metrics({"stage": "warmup", "train/loss": 1})
    assert record["train/qa_old_count"] == 8


def test_evaluation_merges_quality_groups_and_keeps_diagnostics_in_tables(tmp_path):
    summary = evaluate(Task(), [trajectory()], tmp_path, "test", 8)
    rows = [json.loads(line) for line in (tmp_path / "trajectories.jsonl").read_text().splitlines()]
    media = evaluation_media(summary, rows)
    assert set(media) == {
        "evaluation/nll",
        "evaluation/em",
        "evaluation/f1",
        "evaluation/summary",
        "evaluation/details",
        "evaluation/examples",
    }
    for metric in ("nll", "em", "f1"):
        chart = media[f"evaluation/{metric}"]
        assert isinstance(chart, swanlab.echarts.Bar)
        assert chart.options["xAxis"][0]["data"] == ["all", "old", "new"]
        assert len(chart.options["series"]) == 1
        assert [point["value"] for point in chart.options["series"][0]["data"]] == pytest.approx(
            [summary["quality"][group][metric] for group in ("all", "old", "new")]
        )
    assert all(
        isinstance(media[f"evaluation/{name}"], swanlab.echarts.Table)
        for name in ("summary", "details", "examples")
    )
    assert "gate_qa_reads" in media["evaluation/details"].html_content
    assert "Hidden gate question" not in media["evaluation/examples"].html_content
    assert "The blue whale." in media["evaluation/examples"].html_content


@pytest.mark.parametrize(
    "stage,metric",
    [
        ("pretrain", "ae_lm_loss"),
        ("lm", "lm_loss"),
        ("warmup", "qa_loss"),
        ("policy", "qa_loss"),
        ("qa", "qa_loss"),
    ],
)
def test_training_losses_are_separated_by_objective_without_empty_metrics(stage, metric):
    values = training_metrics({"stage": stage, "train/loss": 2, "dev/loss": 3})
    assert values == {f"train/{metric}": 2, f"dev/{metric}": 3}


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


def online_config(root, stage, method="memory_change", init=None):
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
    cursor, checkpoint = prepare_training(config, engine, data, run, stop_after_steps=steps)
    with runtime._tracking_context(config, engine, run) as active:
        result = train_loop(
            config, engine, data, run, cursor, checkpoint, stop_after_steps=steps, tracking=active
        )
    return result, run


def test_two_dynamic_methods_share_one_source_and_keep_one_run_per_method(
    tmp_path, recorded_swanlab
):
    source = online_config(tmp_path / "shared-pretrain-k64_trial", "pretrain")
    source_result, source_run = run_stage(source, 2, "pretrain")
    source_identity = json.loads((tmp_path / "shared-pretrain-k64_trial/swanlab.json").read_text())
    assert "method:shared-pretrain" in source_identity["tags"]
    sources = []
    for method in ("memory_change", "information_loss"):
        root = tmp_path / f"{method}-k64_trial"
        warmup = online_config(root, "warmup", method, source_result["checkpoint"])
        warmup_result, warmup_run = run_stage(warmup, 2)
        identity_bytes = (root / "swanlab.json").read_bytes()
        identity = json.loads(identity_bytes)
        before = json.loads((root / "experiment.json").read_text())
        policy = online_config(root, "policy", method, warmup_result["checkpoint"])
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
        assert cloud["name"] == root.name
        assert [step for step, values in cloud["logs"] if "train/qa_loss" in values] == [1, 2, 3, 4]
        assert not any("train/ae_lm_loss" in values for _, values in cloud["logs"])
        boundaries = [
            values["train/stages"].html_content
            for _, values in cloud["logs"]
            if "train/stages" in values
        ]
        assert "warmup" in boundaries[-1] and "policy" in boundaries[-1]
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
    assert sources[0]["step"] == 2


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
    cursor, checkpoint = prepare_training(config, engine, pretrain_data, run, stop_after_steps=2)
    with runtime._tracking_context(config, engine, run) as active:
        result = train_loop(
            config,
            engine,
            pretrain_data,
            run,
            cursor,
            checkpoint,
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
        cursor, checkpoint = prepare_training(target, engine, qa_data, qa_run, stop_after_steps=2)
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
            target, engine, qa_data, qa_run, cursor, checkpoint, stop_after_steps=2, tracking=active
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
        boundaries = [
            values["train/stages"] for _, values in remote["logs"] if "train/stages" in values
        ]
        assert "pretrain" in boundaries[-1].html_content
        assert "qa" in boundaries[-1].html_content
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
    cursor, checkpoint = prepare_training(config, engine, data, run, stop_after_steps=1)
    with runtime._tracking_context(config, engine, run) as active:
        result = train_loop(
            config, engine, data, run, cursor, checkpoint, stop_after_steps=1, tracking=active
        )
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
    root = tmp_path / "shared-pretrain-k64_trial"
    config = online_config(root, "pretrain")
    result, run = run_stage(config, 1, "pretrain")
    identity = json.loads((root / "swanlab.json").read_text())
    recorded_swanlab.runs[identity["id"]]["state"] = state
    engine = make_engine(config)
    data = make_splits()
    cursor, checkpoint = prepare_training(
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
                    cursor,
                    checkpoint,
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
            cursor,
            checkpoint,
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
    cursor, checkpoint = prepare_training(config, engine, data, run, stop_after_steps=1)
    with pytest.raises(RuntimeError, match="QA data unavailable"):
        with runtime._tracking_context(config, engine, run) as active:
            result = train_loop(
                config, engine, data, run, cursor, checkpoint, stop_after_steps=1, tracking=active
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
    source = online_config(tmp_path / "shared-pretrain-k64_trial", "pretrain")
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
    root = tmp_path / "shared-pretrain-k64_trial"
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
