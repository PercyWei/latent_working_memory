"""真实小模型检验动态写入版本的动作、历史范围和训练生命周期。"""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from latent_working_memory.v3 import gpu_job, job_plan, objective
from latent_working_memory.v3.checkpoint import TokenMemoryCheckpointHandler, read_checkpoint
from latent_working_memory.v3.config import (
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainingConfig,
    load_experiment,
)
from latent_working_memory.v3.engine import TokenMemoryEngine
from latent_working_memory.v3.evaluate import evaluate
from latent_working_memory.v3.runtime import (
    dataset_identity,
    initialization_record,
    load_initialization,
    make_run,
)
from latent_working_memory.v3.segmentation import example_rng
from latent_working_memory.v3.training_data import make_data_loader
from .test_model import build_model, randomize_adapter
from .test_objective import TinyTokenizer, task, trajectory


MODES = ("tag", "mask", "dual_lora")


def traced_writes(codec):
    calls, original = [], codec.compress_batch

    def traced(tokens, histories=None, output_slots=None, actions=None):
        outputs = original(tokens, histories, output_slots, actions)
        histories = histories if histories is not None else [[] for _ in tokens]
        output_slots = output_slots if output_slots is not None else list(map(len, outputs))
        actions = actions if actions is not None else ["initial"] * len(tokens)
        for text, history, slots, action, output in zip(
            tokens, histories, output_slots, actions, outputs, strict=True
        ):
            calls.append((int(text[0]) - 3, tuple(history), slots, action, output))
        return outputs

    codec.compress_batch = traced
    return calls


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", ["memory_change", "information_loss"])
@pytest.mark.parametrize("stage", ["warmup", "policy"])
def test_new_writers_read_all_history_and_keep_action_specific_capacity(
    monkeypatch, mode, method, stage
):
    model = task(method, stage, memory_slots=6, append_slots=2, writer_mode=mode)
    randomize_adapter(model.codec)
    choices = iter([False, True, False, True])
    if stage == "warmup":

        class FixedActions:
            def random(self):
                return 0.0 if next(choices) else 1.0

        monkeypatch.setattr(objective, "example_rng", lambda *args: FixedActions())
    elif method == "memory_change":
        monkeypatch.setattr(
            objective,
            "memory_change_score",
            lambda *args: torch.tensor(float(next(choices))),
        )
    else:
        monkeypatch.setattr(objective, "damage_action", lambda *args: next(choices))
    calls = traced_writes(model.codec)
    output = model(trajectory(n=5))
    output["loss"].backward()
    assert output["metrics"]["slots_final"] == 10
    assert output["metrics"]["appends"] == output["metrics"]["overwrites"] == 2
    assert calls[0][1:4] == ((), 6, "initial")
    for step, history, slots, action, _ in calls[1:]:
        assert len(history) == [0, 1, 1, 2, 2][step]
        assert slots == (2 if action == "append" else 6 if step < 3 else 2)
    if stage == "warmup":
        assert [call[3] for call in calls] == [
            "initial",
            "overwrite",
            "append",
            "overwrite",
            "append",
        ]
    else:
        assert [call[3] for call in calls if call[3] != "append"] == [
            "initial",
            "overwrite",
            "overwrite",
            "overwrite",
            "overwrite",
        ]
    assert model.codec.memory_embeddings.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.codec.decoder.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in model.codec.language_model.named_parameters()
        if "lora_" in name
    )


def rows():
    return [
        replace(trajectory(n=n), trajectory_id=f"writer-row-{i}-{n}")
        for i, n in enumerate((2, 5, 3))
    ]


def trainable(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def assert_parameters(left, right, rtol=2e-4, atol=1e-6):
    for actual, expected in zip(trainable(left), trainable(right), strict=True):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", ["memory_change", "information_loss"])
def test_variable_length_microbatch_matches_independent_trajectories(mode, method):
    model = task(method, "warmup", writer_mode=mode, memory_slots=6, append_slots=2)
    randomize_adapter(model.codec)
    reference = deepcopy(model)
    batch = rows()
    expected = torch.stack([reference(row, epoch=3)["loss"] for row in batch]).mean()
    expected.backward()
    actual = model(batch, epoch=3, batched=True)["loss"]
    actual.backward()
    torch.testing.assert_close(actual, expected)
    assert_parameters(model, reference)


def training_config(microbatch):
    return TrainingConfig(
        dataset_dir="unused",
        micro_batch_size_per_gpu=microbatch,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip=100.0,
    )


def reference_step(model, optimizer, batch, epoch):
    """逐轨迹、逐反传窗口归约，作为 microbatch/DDP 的独立调度参照。"""
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for row in batch:
        state = None
        windows = (
            math.ceil(len(row.segments) / model.cfg.bptt_steps)
            if model.cfg.bptt_steps is not None
            else 1
        )
        for _ in range(windows):
            kwargs = (
                {"window_steps": model.cfg.bptt_steps, "qa_state": state}
                if model.cfg.bptt_steps is not None
                else {}
            )
            output = model(row, epoch=epoch, **kwargs)
            loss = output["loss"] / len(batch)
            total += float(loss.detach())
            loss.backward()
            if kwargs:
                state = output["qa_state"]
                assert all(not block.requires_grad for blocks in state.blocks for block in blocks)
    norm = torch.nn.utils.clip_grad_norm_(trainable(model), 100.0)
    optimizer.step()
    return total, float(norm)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stage", ["warmup", "policy"])
def test_new_writer_truncated_bptt_preserves_microbatch_mean_and_updates_once(mode, stage):
    model = task(
        "memory_change",
        stage,
        writer_mode=mode,
        memory_slots=6,
        append_slots=2,
        bptt_steps=2,
        threshold_i=0.0,
    )
    randomize_adapter(model.codec)
    reference = deepcopy(model)
    engine = TokenMemoryEngine(model, training_config(2), "cpu")
    engine.initialize()
    optimizer = torch.optim.AdamW(trainable(reference), lr=1e-3, weight_decay=0.0)
    expected_loss, expected_norm = reference_step(reference, optimizer, rows(), epoch=2)
    actual = engine.step(rows(), epoch=2)
    assert actual["loss"] == pytest.approx(expected_loss, rel=1e-5)
    assert actual["grad_norm"] == pytest.approx(expected_norm, rel=1e-5)
    assert actual["samples"] == 3
    assert_parameters(model, reference)
    assert all(state["step"].item() == 1 for state in engine.optimizer.state.values())


def action_row(append, suffix):
    for index in range(100):
        identifier = f"branch-{append}-{suffix}-{index}"
        if (example_rng(20261004, 0, identifier).random() < 0.5) is append:
            return replace(trajectory(n=2), trajectory_id=identifier)
    raise AssertionError("could not find deterministic action fixture")


def _dual_ddp_worker(rank, rendezvous, bptt_steps, writer_mode="dual_lora"):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        model = task(
            "memory_change",
            "warmup",
            writer_mode=writer_mode,
            memory_slots=6,
            append_slots=2,
            bptt_steps=bptt_steps,
        )
        randomize_adapter(model.codec)
        reference = deepcopy(model)
        engine = TokenMemoryEngine(model, training_config(1), "cpu")
        engine.initialize()
        optimizer = torch.optim.AdamW(trainable(reference), lr=1e-3, weight_decay=0.0)
        # 第一微批两 rank 分别覆盖/追加，末微批均追加且不使用覆盖 LoRA。
        batch = [
            action_row(False, 0),
            action_row(True, 1),
            action_row(True, 2),
            action_row(True, 3),
        ]
        batches = [batch, batch[-2:]]
        if writer_mode == "tag":
            # 首段无动作标记；此尾批包括空 rank，tag 梯度应保持 None。
            batches.append([trajectory(n=1)])
        for current in batches:
            expected_loss, expected_norm = reference_step(reference, optimizer, current, epoch=0)
            actual = engine.step(current, epoch=0)
            assert actual["loss"] == pytest.approx(expected_loss, rel=1e-5)
            assert actual["grad_norm"] == pytest.approx(expected_norm, rel=1e-5)
            assert_parameters(model, reference)
            for actual_parameter, expected_parameter in zip(
                trainable(model), trainable(reference), strict=True
            ):
                if expected_parameter in optimizer.state:
                    assert engine.optimizer.state[actual_parameter]["step"].item() == (
                        optimizer.state[expected_parameter]["step"].item()
                    )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("bptt_steps", [None, 1])
def test_dual_lora_ddp_accumulates_earlier_branch_gradients(tmp_path, bptt_steps):
    mp.spawn(_dual_ddp_worker, args=(str(tmp_path / "rendezvous"), bptt_steps), nprocs=2, join=True)


@pytest.mark.parametrize("bptt_steps", [None, 1])
def test_tag_writer_ddp_accumulates_rows_and_handles_unmarked_initial_tail(tmp_path, bptt_steps):
    mp.spawn(
        _dual_ddp_worker,
        args=(str(tmp_path / "rendezvous"), bptt_steps, "tag"),
        nprocs=2,
        join=True,
    )


def experiment(output, mode="local", stage="warmup", method="memory_change", tag_tokens=3):
    return ExperimentConfig(
        ModelConfig(
            model_name_or_path="tiny",
            memory_slots=6,
            lora_rank=2,
            lora_alpha=4,
            dtype="float32",
            attention_implementation="eager",
            lora_target_modules=("q_proj", "v_proj"),
        ),
        ObjectiveConfig(
            method=method, stage=stage, writer_mode=mode, append_slots=2, tag_tokens=tag_tokens
        ),
        TrainingConfig(dataset_dir="unused", output_dir=str(output)),
    )


def save_source_checkpoint(tmp_path, model, config):
    path = tmp_path / "checkpoints" / "global_step_1"
    path.mkdir(parents=True)
    run = {
        "config": config.to_dict(),
        "resolved_model_revision": None,
        "step_offset": 0,
        "pretraining_sources": {"document_ids": ["pre"], "dedup_clusters": ["pre-cluster"]},
        "pretraining": None,
    }
    torch.save(
        {
            "run": run,
            "trainable": model.trainable_state_dict(),
            "optimizer": {},
            "cursor": {"epoch": 0, "sample_offset": 1, "step": 1, "sample_visits": 1},
            "rng": [],
        },
        path / "state.pt",
    )
    return path, run


def test_old_pretraining_initializes_two_loras_without_changing_checkpoint_layout(tmp_path):
    source = build_model(memory_slots=6)
    randomize_adapter(source)
    config = experiment(tmp_path / "pre", stage="pretrain", method="dynamic")
    path, run = save_source_checkpoint(tmp_path / "pre", source, config)
    assert "writer_mode" not in run["config"]["objective"]
    checkpoint_before = torch.load(path / "state.pt", weights_only=False)
    target = build_model(memory_slots=6, writer_mode="dual_lora")
    target_config = experiment(tmp_path / "warmup", mode="dual_lora")
    initialization = load_initialization(path, target, target_config)
    assert initialization["pretraining"]["checkpoint"] == str(path.resolve())
    ids = torch.tensor([3, 4, 5])
    expected = source.compress(ids)
    for action in ("initial", "append", "overwrite"):
        actual = target.compress(ids, output_slots=6, action=action)
        torch.testing.assert_close(actual, expected)
    restored = build_model(memory_slots=6, writer_mode="dual_lora")
    restored.load_trainable_state_dict(target.trainable_state_dict())
    for key in ("memory_embeddings", "adapter"):
        assert key in restored.trainable_state_dict()
    saved_again = torch.load(path / "state.pt", weights_only=False)
    assert set(saved_again) == {"run", "trainable", "optimizer", "cursor", "rng"}
    assert set(saved_again["trainable"]) == {"memory_embeddings", "adapter"}
    assert saved_again["run"] == checkpoint_before["run"]
    assert set(saved_again["trainable"]["adapter"]) == set(
        checkpoint_before["trainable"]["adapter"]
    )


@pytest.mark.parametrize("tag_tokens", [1, 3, 5])
def test_old_pretraining_initializes_tag_writer_without_reinitializing_action_embeddings(
    tmp_path, tag_tokens
):
    source = build_model(memory_slots=6)
    randomize_adapter(source)
    source_config = experiment(tmp_path / "pre", stage="pretrain", method="dynamic")
    path, _ = save_source_checkpoint(tmp_path / "pre", source, source_config)
    source_bytes = (path / "state.pt").read_bytes()
    target = build_model(memory_slots=6, writer_mode="tag", tag_tokens=tag_tokens)
    assert target.tag_embeddings.shape == (2, tag_tokens, target.width)
    initial_tags = target.tag_embeddings.detach().clone()
    initialization = load_initialization(
        path, target, experiment(tmp_path / "warmup", mode="tag", tag_tokens=tag_tokens)
    )
    assert initialization["pretraining"]["checkpoint"] == str(path.resolve())
    torch.testing.assert_close(target.tag_embeddings, initial_tags)
    torch.testing.assert_close(target.memory_embeddings, source.memory_embeddings)
    source_state, target_state = source.trainable_state_dict(), target.trainable_state_dict()
    assert set(target_state) == {"memory_embeddings", "adapter"}
    assert set(target_state["adapter"]) == set(source_state["adapter"]) | {"tag_embeddings"}
    for name, tensor in source_state["adapter"].items():
        torch.testing.assert_close(target_state["adapter"][name], tensor)
    ids = torch.tensor([3, 4, 5])
    torch.testing.assert_close(target.compress(ids, action="initial"), source.compress(ids))
    assert (path / "state.pt").read_bytes() == source_bytes


@pytest.mark.parametrize("tag_tokens", [1, 3, 5])
def test_both_action_embeddings_are_trained_by_selected_append_and_overwrite(
    monkeypatch, tag_tokens
):
    model = task(
        "memory_change",
        "warmup",
        memory_slots=6,
        append_slots=2,
        writer_mode="tag",
        tag_tokens=tag_tokens,
    )
    randomize_adapter(model.codec)
    before = model.codec.tag_embeddings.detach().clone()

    class FixedActions:
        def __init__(self):
            self.choices = iter((0.0, 1.0))

        def random(self):
            return next(self.choices)

    monkeypatch.setattr(objective, "example_rng", lambda *args: FixedActions())
    engine = TokenMemoryEngine(model, training_config(1), "cpu")
    engine.initialize()
    assert any(parameter is model.codec.tag_embeddings for parameter in engine.parameters)
    engine.step([trajectory(n=3)])
    assert (model.codec.tag_embeddings.grad.abs().sum(dim=-1) > 0).all()
    assert (model.codec.tag_embeddings.detach() - before).abs().sum(dim=-1).gt(0).all()
    assert model.codec.tag_embeddings in engine.optimizer.state
    saved = model.trainable_state_dict()
    torch.testing.assert_close(saved["adapter"]["tag_embeddings"], model.codec.tag_embeddings)


@pytest.mark.parametrize("mode", ["tag", "dual_lora"])
@pytest.mark.parametrize("bptt_steps", [None, 2])
def test_writer_checkpoint_restores_optimizer_loader_and_next_update(tmp_path, mode, bptt_steps):
    tag_tokens = 5 if mode == "tag" else 3
    config = experiment(tmp_path, mode=mode, tag_tokens=tag_tokens)
    config = replace(
        config,
        objective=replace(config.objective, bptt_steps=bptt_steps),
        training=replace(training_config(1), output_dir=str(tmp_path)),
    )
    model = task(
        "memory_change",
        "warmup",
        memory_slots=6,
        append_slots=2,
        writer_mode=mode,
        tag_tokens=tag_tokens,
        bptt_steps=bptt_steps,
    )
    randomize_adapter(model.codec)
    engine = TokenMemoryEngine(model, config.training, "cpu")
    engine.initialize()
    splits = {split: tuple(rows()) for split in ("train", "dev", "test")}
    statistics = {
        "source_data": {split: dataset_identity(records) for split, records in splits.items()}
    }
    run = make_run(config, splits, statistics, "cpu", 1)
    loader = make_data_loader(splits["train"], engine.global_batch_size, config.training.seed)
    iterator = iter(loader)
    engine.step(next(iterator), epoch=0)
    cursor = {"epoch": 0, "sample_offset": 2, "step": 1, "sample_visits": 2}
    handler = TokenMemoryCheckpointHandler(engine, loader, tmp_path / "checkpoints", run, cursor)
    handler.save_checkpoint(1)
    checkpoint = tmp_path / "checkpoints" / "global_step_1"
    state = read_checkpoint(checkpoint)
    assert set(state) == {"run", "trainable", "optimizer", "cursor", "rng"}
    assert set(state["trainable"]) == {"memory_embeddings", "adapter"}
    assert all(torch.is_tensor(value) for value in state["trainable"]["adapter"].values())
    if mode == "tag":
        assert state["trainable"]["adapter"]["tag_embeddings"].shape == (2, 5, model.codec.width)
        assert state["run"]["config"]["objective"]["tag_tokens"] == 5
    assert (checkpoint / "data_0.pt").exists()
    restored = task(
        "memory_change",
        "warmup",
        memory_slots=6,
        append_slots=2,
        writer_mode=mode,
        tag_tokens=tag_tokens,
        bptt_steps=bptt_steps,
    )
    restored_engine = TokenMemoryEngine(restored, config.training, "cpu")
    restored_engine.initialize()
    restored_loader = make_data_loader(
        splits["train"], restored_engine.global_batch_size, config.training.seed
    )
    restored_cursor = {}
    restored_handler = TokenMemoryCheckpointHandler(
        restored_engine,
        restored_loader,
        tmp_path / "checkpoints",
        run,
        restored_cursor,
        resume_from_path=checkpoint,
    )
    assert restored_handler.load_checkpoint() == 1
    assert restored_cursor == cursor
    next_batch, resumed_batch = next(iterator), next(iter(restored_loader))
    assert next_batch == resumed_batch
    expected, actual = engine.step(next_batch), restored_engine.step(resumed_batch)
    assert actual["loss"] == pytest.approx(expected["loss"], rel=1e-6)
    assert actual["grad_norm"] == pytest.approx(expected["grad_norm"], rel=1e-6)
    assert_parameters(restored, model, rtol=1e-6, atol=1e-7)
    for observed, wanted in zip(
        restored_engine.optimizer.state.values(), engine.optimizer.state.values(), strict=True
    ):
        assert observed.keys() == wanted.keys()
        for key in observed:
            torch.testing.assert_close(observed[key], wanted[key], rtol=1e-6, atol=1e-7)


class EvaluationTokenizer(TinyTokenizer):
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", ["memory_change", "information_loss"])
def test_final_evaluation_uses_saved_writer_and_reproduces_memory_quality(tmp_path, mode, method):
    tag_tokens = 5 if mode == "tag" else 3
    model = task(
        method,
        "warmup",
        memory_slots=6,
        append_slots=2,
        writer_mode=mode,
        tag_tokens=tag_tokens,
        append_probability=0.0,
        threshold_i=0.0,
    )
    randomize_adapter(model.codec)
    model.tokenizer = EvaluationTokenizer()
    config = experiment(tmp_path, mode=mode, method=method)
    config = replace(config, objective=model.cfg)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    saved = load_experiment(config_path)
    restored = task(
        method,
        "warmup",
        memory_slots=6,
        append_slots=2,
        writer_mode=saved.objective.writer_mode,
        tag_tokens=saved.objective.tag_tokens,
    )
    restored.cfg = saved.objective
    restored.tokenizer = EvaluationTokenizer()
    restored.load_trainable_state_dict(model.trainable_state_dict())
    batch = [trajectory(split="test", n=3)]
    expected = evaluate(model, batch, tmp_path / "before", "test", 2)
    actual = evaluate(restored, batch, tmp_path / "after", "test", 2)
    assert actual["quality"] == expected["quality"]
    assert actual["capacity"] == expected["capacity"]
    if method == "memory_change":
        # 即使保存的是纯覆盖 warmup 日程，最终评估仍按门控策略追加。
        assert actual["capacity"]["final_slots"] == 10
    before = json.loads((tmp_path / "before" / "trajectories.jsonl").read_text())
    after = json.loads((tmp_path / "after" / "trajectories.jsonl").read_text())
    assert after["questions"] == before["questions"]
    assert [event["action"] for event in after["events"]] == [
        event["action"] for event in before["events"]
    ]


@pytest.mark.parametrize("mode", MODES)
def test_after_warmup_writer_mode_cannot_change(tmp_path, mode):
    source = task("memory_change", "warmup", memory_slots=6, append_slots=2, writer_mode=mode)
    config = experiment(tmp_path / "source", mode=mode)
    path, run = save_source_checkpoint(tmp_path / "source", source, config)
    target = experiment(tmp_path / "policy", mode="local", stage="policy")
    with pytest.raises(ValueError, match="writer"):
        initialization_record(path, run, 1, target)


def test_after_warmup_tag_token_count_cannot_change(tmp_path):
    source = task(
        "memory_change",
        "warmup",
        memory_slots=6,
        append_slots=2,
        writer_mode="tag",
        tag_tokens=3,
    )
    config = experiment(tmp_path / "source", mode="tag", tag_tokens=3)
    path, run = save_source_checkpoint(tmp_path / "source", source, config)
    target = experiment(tmp_path / "policy", mode="tag", stage="policy", tag_tokens=5)
    with pytest.raises(ValueError, match="tag_tokens"):
        initialization_record(path, run, 1, target)


@pytest.mark.parametrize("tag_tokens", [1, 3, 5])
def test_tag_token_count_roundtrip_and_default(tmp_path, tag_tokens):
    config = experiment(tmp_path, mode="tag", tag_tokens=tag_tokens)
    raw = config.to_dict()
    assert raw["objective"]["tag_tokens"] == tag_tokens
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    restored = load_experiment(path)
    assert restored.objective.tag_tokens == tag_tokens
    assert restored.to_dict() == raw
    assert (
        ObjectiveConfig(method="memory_change", stage="warmup", writer_mode="tag").tag_tokens == 3
    )


@pytest.mark.parametrize("mode", ["local", "mask", "dual_lora"])
def test_non_tag_configuration_omits_control_token_count(tmp_path, mode):
    raw = experiment(tmp_path, mode=mode).to_dict()
    assert "tag_tokens" not in raw["objective"]


@pytest.mark.parametrize("tag_tokens", [0, -1, True, 1.5])
def test_tag_token_count_is_a_positive_integer(tag_tokens):
    with pytest.raises(ValueError, match="tag_tokens"):
        ObjectiveConfig(
            method="memory_change", stage="warmup", writer_mode="tag", tag_tokens=tag_tokens
        )


def test_default_local_configuration_roundtrip_preserves_old_saved_identity(tmp_path):
    config = experiment(tmp_path)
    raw = config.to_dict()
    assert "writer_mode" not in raw["objective"]
    assert "tag_tokens" not in raw["objective"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    loaded = load_experiment(path)
    assert loaded.objective.writer_mode == "local"
    assert loaded.to_dict() == raw


@pytest.mark.parametrize(
    "method,stage",
    [("dynamic", "pretrain"), ("icae_single", "qa"), ("autocompressors", "pretrain")],
)
@pytest.mark.parametrize("mode", MODES)
def test_nonlocal_writer_modes_are_only_dynamic_posttraining(method, stage, mode):
    with pytest.raises(ValueError, match="writer_mode"):
        ObjectiveConfig(method=method, stage=stage, writer_mode=mode)


def test_fixed_text_action_writer_mode_is_not_kept_as_an_alias():
    with pytest.raises(ValueError, match="writer_mode"):
        ObjectiveConfig(method="memory_change", stage="warmup", writer_mode="action")
    with pytest.raises(SystemExit):
        gpu_job.parse_args(["--method", "memory_change", "--writer-mode", "action"])


def test_mask_writer_rejects_flash_attention_at_configuration_boundary(tmp_path):
    config = experiment(tmp_path, mode="mask")
    with pytest.raises(ValueError, match="mask writer"):
        replace(config, model=replace(config.model, attention_implementation="flash_attention_2"))


@pytest.mark.parametrize("mode", ("local", *MODES))
def test_cli_selects_only_dynamic_writers_and_gives_each_variant_its_own_directory(tmp_path, mode):
    args = gpu_job.parse_args(
        [
            "--method",
            "all",
            "--writer-mode",
            mode,
            "--run-id",
            "writers",
            "--mode",
            "smoke",
            "--output-root",
            str(tmp_path),
            "--tracking",
            "disabled",
            "--dry-run",
        ]
    )
    _, _, jobs = job_plan.build_jobs(args)
    for job in jobs:
        config = job.config
        if (
            config.objective.method in {"memory_change", "information_loss"}
            and config.objective.stage != "pretrain"
        ):
            assert config.objective.writer_mode == mode
            assert f"-k512+32-{mode.replace('_', '-')}_smoke" in config.training.experiment_dir
        else:
            assert config.objective.writer_mode == "local"
            assert "writer_mode" not in config.to_dict()["objective"]
            assert "tag_tokens" not in config.to_dict()["objective"]
    for method in ("dynamic_pretrain", "icae_single", "autocompressors"):
        with pytest.raises(SystemExit):
            gpu_job.parse_args(["--method", method, "--writer-mode", mode])


def test_custom_tag_count_cli_does_not_modify_pretraining_or_baselines(tmp_path):
    args = gpu_job.parse_args(
        [
            "--method",
            "all",
            "--writer-mode",
            "tag",
            "--tag-tokens",
            "5",
            "--run-id",
            "tag-count",
            "--mode",
            "smoke",
            "--output-root",
            str(tmp_path),
            "--tracking",
            "disabled",
            "--dry-run",
        ]
    )
    _, _, jobs = job_plan.build_jobs(args)
    for job in jobs:
        if job.config.objective.writer_mode == "tag":
            assert job.config.objective.tag_tokens == 5
            assert job.config.to_dict()["objective"]["tag_tokens"] == 5
        else:
            assert "tag_tokens" not in job.config.to_dict()["objective"]


@pytest.mark.parametrize("mode", ["local", "mask", "dual_lora"])
def test_tag_count_cli_requires_tag_writer(tmp_path, mode):
    args = gpu_job.parse_args(
        [
            "--method",
            "memory_change",
            "--writer-mode",
            mode,
            "--tag-tokens",
            "5",
            "--run-id",
            "tag-count",
            "--mode",
            "smoke",
            "--output-root",
            str(tmp_path),
            "--tracking",
            "disabled",
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="tag-tokens.*tag"):
        job_plan.build_jobs(args)
