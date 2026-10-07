from dataclasses import replace
from contextlib import contextmanager
import csv
import json

import pytest
import torch
from torch import nn

from latent_working_memory.v3 import evaluate as evaluation
from latent_working_memory.v3.compare import compare
from latent_working_memory.v3.config import (
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainingConfig,
)
from latent_working_memory.v3.data import FactQATrajectory, QA, Segment, StepUsage
from latent_working_memory.v3.objective import TokenMemoryTask
from .test_model import build_model
from .test_objective import TinyTokenizer, trajectory as objective_trajectory


def trajectory(identifier="doc1", split="test", extra_question=False):
    text = "First.Second.Third."
    segments = tuple(Segment(f"s{i + 1}", (i * 6, (i + 1) * 6), (i + 3,)) for i in range(3))
    qas = {}
    rows = [
        ("old", "s1", "What animal?", "blue whale", "evaluation", (10, 11)),
        ("new", "s3", "What fish?", "red fish", "evaluation", (12, 13)),
        ("gate", "s1", "Hidden gate question?", "gate answer", "gate", (14,)),
    ]
    if extra_question:
        rows.append(("extra", "s2", "What place?", "Paris", "evaluation", (15,)))
    for name, segment, question, answer, role, answer_ids in rows:
        qid = f"{identifier}:{name}"
        qas[qid] = QA(
            qid,
            segment,
            f"{identifier}:{name}:fact",
            answer,
            question,
            answer,
            role,
            (0, 1),
            (0, 1),
            (3,),
            answer_ids,
        )
    return FactQATrajectory(
        identifier,
        identifier,
        f"cluster:{identifier}",
        split,
        text,
        {"file": "local.parquet", "row_group": 0, "row_index": 0},
        (0, len(text)),
        segments,
        qas,
        tuple(
            StepUsage(segment.segment_id, (), (), (f"{identifier}:gate",)) for segment in segments
        ),
        (3, 4, 5),
    )


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, values, skip_special_tokens):
        assert skip_special_tokens
        return {9: "The blue whale.", 10: "red", 11: "Paris"}[values[0]]


class Codec(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory_embeddings = nn.Parameter(torch.zeros(2, 3))
        self.generated = []
        self.loaded = None

    def generate(self, memory, prompt, max_new_tokens, eos_token_id, pad_token_id):
        assert not torch.is_grad_enabled()
        assert eos_token_id == 2 and pad_token_id == 0
        assert memory.shape[1] == 3
        self.generated.append((memory.shape, prompt.tolist(), max_new_tokens))
        return prompt

    def load_trainable_state_dict(self, state):
        self.loaded = state


class Task(nn.Module):
    def __init__(self, method="memory_change"):
        super().__init__()
        self.codec = Codec()
        self.tokenizer = Tokenizer()
        stage = {"icae_single": "qa", "icae_multi": "qa", "autocompressors": "lm"}.get(
            method, "policy"
        )
        self.cfg = ObjectiveConfig(method=method, stage=stage)
        self.loss_queries = []

    def build_memory(self, example, epoch, force_policy):
        assert epoch == 0 and force_policy
        assert not self.training and not torch.is_grad_enabled()
        states = [("initial", 2), ("append", 4), ("overwrite", 4)]
        if self.cfg.method == "icae_single":
            states = [("single", 2)]
        events = [
            {
                "step": index,
                "segment_id": f"s{index + 1}",
                "action": action,
                "slots": slots,
                "write_calls": 1,
                "write_seconds": 0.01,
                "gate_qa_reads": 6 if index else 0,
                "gate_seconds": 0.02 if index else 0,
                "scores": {} if not index else {"L0": 2.0, "Lrw": 3.0, "Lapp": 2.5},
            }
            for index, (action, slots) in enumerate(states)
        ]
        return [torch.zeros(2, 3) for _ in range(states[-1][1] // 2)], events

    def qa_losses(self, blocks, example, qa_ids):
        assert not torch.is_grad_enabled()
        self.loss_queries.append(list(qa_ids))
        return torch.tensor(
            [{"old": 1.0, "new": 3.0, "extra": 5.0}[qid.rsplit(":", 1)[1]] for qid in qa_ids]
        )

    def prompt_ids(self, question):
        return torch.tensor([{"What animal?": 9, "What fish?": 10, "What place?": 11}[question]])


def test_evaluation_excludes_gate_and_aggregates_final_quality_capacity_and_costs(tmp_path):
    task = Task("information_loss")
    examples = [trajectory(), trajectory("doc2", extra_question=True)]
    summary = evaluation.evaluate(task, examples, tmp_path, "test", 12)
    assert task.training
    assert summary["offline_oracle"] is True
    assert summary["quality"]["all"]["questions"] == 5
    assert summary["quality"]["all"]["nll"] == pytest.approx(13 / 5)
    assert summary["quality"]["all"]["em"] == pytest.approx(3 / 5)
    assert summary["quality"]["all"]["f1"] == pytest.approx((3 + 4 / 3) / 5)
    assert summary["quality"]["new"]["questions"] == 2
    assert summary["quality"]["old"]["questions"] == 3
    assert summary["quality"]["by_age"]["2"]["nll"] == 1
    assert summary["quality"]["by_age"]["1"]["nll"] == 5
    assert summary["capacity"] == {
        "final_slots": 4,
        "mean_slots": pytest.approx(10 / 3),
        "appends": 1,
        "overwrites": 1,
    }
    assert summary["costs"]["gate_qa_reads"] == 24
    assert summary["costs"]["write_calls"] == 6
    assert summary["costs"]["nll_qa_reads"] == 5
    assert summary["costs"]["generation_qa_reads"] == 5
    assert summary["costs"]["generated_tokens"] == 5
    assert len(task.codec.generated) == 5
    assert all(not qid.endswith(":gate") for batch in task.loss_queries for qid in batch)
    records = [
        json.loads(line) for line in (tmp_path / "trajectories.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[0]["events"][1]["scores"]["Lrw"] == 3.0
    assert [row["age"] for row in records[0]["questions"]] == [2, 0]
    assert records[0]["questions"][0]["prediction"] == "The blue whale."
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert not (tmp_path / "trajectories.jsonl.tmp").exists()


@pytest.mark.parametrize(
    "method", ["icae_single", "icae_multi", "autocompressors", "memory_change", "information_loss"]
)
def test_all_five_methods_evaluate_the_same_final_question_panel(tmp_path, method):
    task = Task(method)
    summary = evaluation.evaluate(task, [trajectory()], tmp_path / method, "test", 8)
    assert task.loss_queries == [["doc1:old", "doc1:new"]]
    assert summary["quality"]["all"]["questions"] == 2
    assert summary["offline_oracle"] == (method == "information_loss")
    if method == "icae_single":
        assert summary["capacity"] == {
            "final_slots": 2,
            "mean_slots": 2,
            "appends": 0,
            "overwrites": 0,
        }


def test_split_contract_and_training_state_restore_on_failure(tmp_path):
    task = Task()
    with pytest.raises(ValueError, match="dev or test"):
        evaluation.evaluate(task, [trajectory(split="train")], tmp_path, "train", 8)
    with pytest.raises(ValueError, match="requested split"):
        evaluation.evaluate(task, [trajectory(split="dev")], tmp_path, "test", 8)
    example = trajectory()
    example = replace(
        example, qas={qid: qa for qid, qa in example.qas.items() if qa.role == "gate"}
    )
    with pytest.raises(ValueError, match="no evaluation questions"):
        evaluation.evaluate(task, [example], tmp_path, "test", 8)
    assert task.training
    assert not (tmp_path / "summary.json").exists()


def test_comparison_keeps_unequal_capacity_and_exports_json_csv(tmp_path):
    example = trajectory()
    for method in ("icae_single", "information_loss"):
        evaluation.evaluate(Task(method), [example], tmp_path / method, "test", 8)
    paths = [tmp_path / method / "summary.json" for method in ("icae_single", "information_loss")]
    points = compare(paths, tmp_path / "comparison")
    assert [point["final_slots"] for point in points] == [2, 4]
    assert [point["offline_oracle"] for point in points] == [False, True]
    assert points[0]["threshold_d"] is None
    assert points[1]["threshold_d"] == 0.1
    assert json.loads((tmp_path / "comparison" / "points.json").read_text()) == points
    with (tmp_path / "comparison" / "points.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["method"] for row in rows] == ["icae_single", "information_loss"]


@pytest.mark.parametrize("difference", ["question", "gate", "limit", "split", "prompt"])
def test_comparison_rejects_changed_evaluation_conditions(tmp_path, difference):
    first = trajectory()
    second = trajectory()
    if difference in {"question", "gate"}:
        qa_id = "doc1:gate" if difference == "gate" else "doc1:old"
        updated = replace(second.qas[qa_id], answer="Changed reference")
        second = replace(second, qas={**second.qas, qa_id: updated})
    split = "dev" if difference == "split" else "test"
    second = replace(second, split=split)
    evaluation.evaluate(Task(), [first], tmp_path / "first", "test", 8)
    second_task = Task()
    if difference == "prompt":
        second_task.cfg = replace(
            second_task.cfg, qa_prompt="Different question: {question}\nAnswer:"
        )
    evaluation.evaluate(
        second_task, [second], tmp_path / "second", split, 9 if difference == "limit" else 8
    )
    with pytest.raises(
        ValueError, match="same split, evaluation dataset, QA prompt and generation limit"
    ):
        compare(
            [tmp_path / name / "summary.json" for name in ("first", "second")],
            tmp_path / "comparison",
        )


@pytest.mark.parametrize("changed_split", [False, True])
@pytest.mark.parametrize("log_to_swanlab", [False, True])
def test_cli_loads_canonical_checkpoint_and_pins_backbone_revision(
    tmp_path, monkeypatch, capsys, changed_split, log_to_swanlab
):
    config = ExperimentConfig(
        ModelConfig(model_name_or_path="local-tiny"),
        ObjectiveConfig(method="icae_single", stage="qa"),
        TrainingConfig(dataset_dir="train-data", output_dir="training"),
    )
    checkpoint = {
        "run": {
            "config": config.to_dict(),
            "resolved_model_revision": "resolved-commit",
            "step_offset": 20,
            "pretraining": None,
            "pretraining_sources": {"document_ids": [], "dedup_clusters": []},
            "data": {"test": evaluation.dataset_identity([trajectory()])},
            "source_data": {"test": evaluation.dataset_identity([trajectory()])},
        },
        "trainable": {"memory_embeddings": torch.zeros(2, 3), "adapter": {}},
        "optimizer": {},
        "cursor": {"step": 7},
        "rng": {},
    }
    if changed_split:
        checkpoint["run"]["source_data"]["test"]["fingerprint"] = "another-panel"
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    task = Task("icae_single")
    loaded = []

    def load_codec(model_config, device):
        loaded.append((model_config, device))
        return task.codec, task.tokenizer

    monkeypatch.setattr(evaluation, "load_codec", load_codec)
    monkeypatch.setattr(evaluation, "TokenMemoryTask", lambda codec, tokenizer, objective: task)
    monkeypatch.setattr(
        evaluation, "load_factqa", lambda directory, tokenizer: {"test": [trajectory()]}
    )
    published = []
    monkeypatch.setattr(evaluation, "_training_run_directory", lambda *args: tmp_path)
    monkeypatch.setattr(evaluation, "_append_evaluation", lambda *args: published.append(args))
    arguments = [
        "--checkpoint",
        str(path),
        "--dataset-dir",
        "evaluation-data",
        "--output-dir",
        str(tmp_path / "evaluation"),
        "--split",
        "test",
        "--device",
        "cpu",
        "--max-new-tokens",
        "8",
    ]
    if log_to_swanlab:
        arguments.append("--log-to-swanlab")
    if changed_split:
        with pytest.raises(ValueError, match="checkpoint's original split"):
            evaluation.main(arguments)
        return
    evaluation.main(arguments)
    summary = json.loads(capsys.readouterr().out)
    assert loaded[0][0].revision == "resolved-commit"
    assert loaded[0][1] == torch.device("cpu")
    assert task.codec.loaded is not None
    assert summary["metadata"]["resolved_model_revision"] == "resolved-commit"
    assert summary["metadata"]["global_step"] == 27
    assert summary["metadata"]["config"]["model"]["revision"] is None
    assert [entry[-1] for entry in published] == ([27] if log_to_swanlab else [])


@pytest.mark.parametrize(
    "field,value", [("document_ids", "doc1"), ("dedup_clusters", "cluster:doc1")]
)
def test_evaluation_sources_cannot_overlap_pretraining_documents_or_clusters(field, value):
    sources = {"document_ids": [], "dedup_clusters": []}
    evaluation._check_qa_sources([trajectory()], sources)
    sources[field] = [value]
    with pytest.raises(ValueError, match="overlap pretraining"):
        evaluation._check_qa_sources([trajectory()], sources)


def test_cli_rejects_distributed_launch_before_loading_or_writing(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single process"):
        evaluation.main(
            [
                "--checkpoint",
                "unused.pt",
                "--dataset-dir",
                "unused",
                "--output-dir",
                "unused",
                "--split",
                "test",
                "--device",
                "cpu",
            ]
        )


@pytest.mark.parametrize(
    "method", ["icae_single", "icae_multi", "autocompressors", "memory_change", "information_loss"]
)
def test_real_tiny_model_evaluates_all_methods_without_training_or_gate_leakage(tmp_path, method):
    class DecoderTokenizer(TinyTokenizer):
        def decode(self, values, skip_special_tokens):
            return " ".join(map(str, values))

    stage = {"icae_single": "qa", "icae_multi": "qa", "autocompressors": "lm"}.get(method, "policy")
    task = TokenMemoryTask(
        build_model(), DecoderTokenizer(), ObjectiveConfig(method=method, stage=stage)
    )
    example = objective_trajectory(split="test")
    summary = evaluation.evaluate(task, [example], tmp_path / method, "test", 2)
    assert summary["quality"]["all"]["questions"] == 3
    assert summary["quality"]["old"]["questions"] == 2
    assert summary["quality"]["all"]["nll"] > 0
    assert summary["costs"]["nll_qa_reads"] == 3
    assert summary["costs"]["gate_qa_reads"] == (6 if method == "information_loss" else 0)
    assert all(parameter.grad is None for parameter in task.parameters())


def test_swanlab_append_requires_original_training_identity(tmp_path):
    config = ExperimentConfig(
        ModelConfig(),
        ObjectiveConfig(),
        TrainingConfig(dataset_dir="data", output_dir=str(tmp_path)),
    )
    with pytest.raises(ValueError, match="configured for SwanLab"):
        evaluation._training_run_directory(config, {})
    config = replace(
        config, training=replace(config.training, swanlab_project="project", group="experiment")
    )
    with pytest.raises(ValueError, match="original training SwanLab identity"):
        evaluation._training_run_directory(config, {})
    (tmp_path / "run.json").write_text(json.dumps({"config": config.to_dict()}))
    (tmp_path / "swanlab.json").write_text(
        json.dumps(
            {
                "id": "original",
                "project": "project",
                "group": "experiment",
                "job_type": "train",
                "mode": "online",
            }
        )
    )
    with pytest.raises(ValueError, match="checkpoint run differs"):
        evaluation._training_run_directory(config, {})
    assert evaluation._training_run_directory(config, {"config": config.to_dict()}) == tmp_path


def test_evaluation_resolves_method_identity_but_verifies_stage_checkpoint(tmp_path):
    directory = tmp_path / "memory-change-k64_20261007-01"
    stage_directory = directory / "policy"
    stage_directory.mkdir(parents=True)
    config = ExperimentConfig(
        ModelConfig(),
        ObjectiveConfig(stage="policy"),
        TrainingConfig(
            dataset_dir="data",
            output_dir=str(stage_directory),
            experiment_dir=str(directory),
            experiment_id="20261007-01",
            swanlab_project="project",
            group="series",
        ),
    )
    run = {"config": config.to_dict(), "step_offset": 20, "pretraining": {"step": 100}}
    (stage_directory / "run.json").write_text(json.dumps(run))
    (directory / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": "20261007-01",
                "method": "memory_change",
                "stages": {"policy": run},
            }
        )
    )
    (directory / "swanlab.json").write_text(
        json.dumps(
            {
                "id": "one-method-run",
                "project": "project",
                "group": "series",
                "job_type": "train",
                "mode": "online",
            }
        )
    )
    assert evaluation._training_run_directory(config, run) == directory
    assert not (stage_directory / "swanlab.json").exists()
    with pytest.raises(ValueError, match="checkpoint run differs"):
        evaluation._training_run_directory(config, {**run, "step_offset": 0})
    changed_run = {**run, "pretraining": {"step": 200}}
    (stage_directory / "run.json").write_text(json.dumps(changed_run))
    with pytest.raises(ValueError, match="saved method experiment"):
        evaluation._training_run_directory(config, changed_run)


def test_swanlab_publication_uses_existing_run_and_keeps_training_config(tmp_path, monkeypatch):
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    identity = {"id": "original", "project": "project", "group": "experiment"}
    identity_path = training_dir / "swanlab.json"
    identity_path.write_text(json.dumps(identity))
    config_path = training_dir / "config.json"
    config_path.write_text('{"preserve": "training configuration"}')
    original_config = config_path.read_bytes()
    publications = []

    class Tracking:
        def log(self, values, step):
            publications.append((values, step))

    @contextmanager
    def resume(directory, api_key):
        assert directory == training_dir
        assert api_key == "evaluation-test-key"
        yield Tracking()

    monkeypatch.setattr(evaluation, "swanlab_api_key", lambda: "evaluation-test-key")
    monkeypatch.setattr(evaluation, "swanlab_training_run", resume)
    summary = evaluation.evaluate(Task(), [trajectory()], tmp_path / "evaluation", "test", 8)
    local_results = {
        name: (tmp_path / "evaluation" / name).read_bytes()
        for name in ("summary.json", "trajectories.jsonl")
    }
    evaluation._append_evaluation(summary, training_dir, tmp_path / "evaluation", 37)
    assert len(publications) == 1 and publications[0][1] == 37
    assert publications[0][0]["evaluation/nll"].options["series"][0]["data"][0]["value"] == 2
    assert publications[0][0]["evaluation/capacity"].options["xAxis"][0]["data"] == [
        "final_slots",
        "mean_slots",
    ]
    assert not any(
        name in publications[0][0]
        for name in ("evaluation/summary", "evaluation/details", "evaluation/examples")
    )
    assert all(name.startswith("evaluation/") for name in publications[0][0])
    assert all(
        (tmp_path / "evaluation" / name).read_bytes() == original
        for name, original in local_results.items()
    )
    assert config_path.read_bytes() == original_config
    assert json.loads(identity_path.read_text()) == identity
    record = json.loads((tmp_path / "evaluation" / "swanlab-publication.json").read_text())
    assert record["run_id"] == "original" and record["step"] == 37
    assert record["quality"] == summary["quality"]
    assert set(record["panels"]) == set(publications[0][0])


@pytest.mark.parametrize("problem", [None, "source_changed", "pretraining_overlap"])
def test_limited_cli_evaluation_verifies_complete_source_before_selecting(
    tmp_path, monkeypatch, capsys, problem
):
    examples = tuple(trajectory(f"doc{i}", split="dev") for i in range(6))
    config = ExperimentConfig(
        ModelConfig(model_name_or_path="local-tiny"),
        ObjectiveConfig(method="icae_single", stage="qa"),
        TrainingConfig(dataset_dir="train-data", output_dir="training", seed=42),
    )
    selected = evaluation.select_examples(examples, 2, 42, "dev", pretraining=False)
    unselected = next(row for row in examples if row not in selected)
    checkpoint = {
        "run": {
            "config": config.to_dict(),
            "resolved_model_revision": None,
            "step_offset": 0,
            "pretraining": None,
            "pretraining_sources": {"document_ids": [], "dedup_clusters": []},
            # 小规模训练的 dev 子集与完整源分别记录，评估校验完整源。
            "data": {"dev": evaluation.dataset_identity(selected)},
            "source_data": {"dev": evaluation.dataset_identity(examples)},
        },
        "trainable": {},
        "optimizer": {},
        "cursor": {"step": 7},
        "rng": {},
    }
    if problem == "source_changed":
        checkpoint["run"]["source_data"]["dev"]["fingerprint"] = "changed-unselected-record"
    elif problem == "pretraining_overlap":
        checkpoint["run"]["pretraining_sources"]["document_ids"] = [unselected.document_id]
    task = Task("icae_single")
    monkeypatch.setattr(evaluation, "read_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(evaluation, "load_codec", lambda *args: (task.codec, task.tokenizer))
    monkeypatch.setattr(evaluation, "TokenMemoryTask", lambda *args: task)
    monkeypatch.setattr(evaluation, "load_factqa", lambda *args: {"dev": examples})
    arguments = [
        "--checkpoint",
        "unused.pt",
        "--dataset-dir",
        "data",
        "--output-dir",
        str(tmp_path),
        "--split",
        "dev",
        "--device",
        "cpu",
        "--max-trajectories",
        "2",
    ]
    if problem is not None:
        message = (
            "checkpoint's original split" if problem == "source_changed" else "overlap pretraining"
        )
        with pytest.raises(ValueError, match=message):
            evaluation.main(arguments)
        assert not (tmp_path / "summary.json").exists()
        assert not task.codec.generated
        return
    evaluation.main(arguments)
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset_signature"] == evaluation.dataset_identity(selected)["fingerprint"]
    assert summary["metadata"]["selection"] == {"total": 6, "selected": 2, "limit": 2, "seed": 42}
    assert summary["quality"]["all"]["questions"] == 4
    records = [
        json.loads(line) for line in (tmp_path / "trajectories.jsonl").read_text().splitlines()
    ]
    assert [row["trajectory_id"] for row in records] == [row.trajectory_id for row in selected]
    assert all(row["segments"] == 3 for row in records)


@pytest.mark.parametrize("limit", ["0", "-1", "1.5"])
def test_cli_rejects_invalid_trajectory_limits_before_model_load(monkeypatch, limit):
    def unexpected_load(path):
        raise AssertionError("invalid CLI argument must not load a checkpoint")

    monkeypatch.setattr(evaluation, "read_checkpoint", unexpected_load)
    with pytest.raises(SystemExit):
        evaluation.main(
            [
                "--checkpoint",
                "unused",
                "--dataset-dir",
                "unused",
                "--output-dir",
                "unused",
                "--split",
                "dev",
                "--device",
                "cpu",
                "--max-trajectories",
                limit,
            ]
        )
