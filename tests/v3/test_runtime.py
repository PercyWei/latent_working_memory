from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, set_seed

from latent_working_memory.data_preparation.segmentation import SegmentationConfig
from latent_working_memory.v3 import runtime
from latent_working_memory.v3.checkpoint import TokenMemoryCheckpointHandler
from latent_working_memory.v3.config import (
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainingConfig,
    load_experiment,
)
from latent_working_memory.v3.engine import TokenMemoryEngine
from latent_working_memory.v3.runtime import (
    dataset_identity,
    load_codec,
    load_initialization,
    load_splits,
    make_run,
    read_checkpoint,
    train_loop as stage_train_loop,
    validate_cursor,
)
from latent_working_memory.v3.training_data import make_data_loader
from latent_working_memory.v4.checkpoint import capture_rng
from .test_objective import trajectory


def train_loop(config, engine, splits, run, resume=None, stop_after_steps=None):
    stage = runtime.prepare_training(
        config, engine, splits, run, resume=resume, stop_after_steps=stop_after_steps
    )
    with runtime._tracking_context(config, engine, run) as tracking:
        return stage_train_loop(
            config,
            engine,
            splits,
            run,
            stage,
            tracking=tracking,
            stop_after_steps=stop_after_steps,
        )


@dataclass(frozen=True)
class Example:
    document_id: str
    dedup_cluster: str
    values: tuple[float, float]
    target: float


class TinyTask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = ObjectiveConfig()
        self.weight = torch.nn.Parameter(torch.tensor([0.4, -0.2]))

    def forward(self, example, epoch=0, differentiable=True, batched=False):
        rows = example if batched else [example]
        noise = torch.tensor(
            [
                float(torch.rand(())) + random.random() + float(np.random.random())
                if differentiable
                else 0.0
                for _ in rows
            ]
        )
        prediction = torch.tensor([row.values for row in rows]) @ self.weight + noise
        return {
            "loss": (prediction - torch.tensor([row.target for row in rows])).square().mean(),
            "metrics": {"input_length": sum(len(row.values) for row in rows) / len(rows)},
        }

    def trainable_state_dict(self):
        return {"weight": self.weight.detach().cpu().clone()}

    def load_trainable_state_dict(self, state):
        if set(state) != {"weight"} or state["weight"].shape != self.weight.shape:
            raise ValueError("invalid trainable state")
        with torch.no_grad():
            self.weight.copy_(state["weight"])


def make_config(output, stage="pretrain", init=None):
    return ExperimentConfig(
        ModelConfig(
            model_name_or_path="local-tiny",
            dtype="float32",
            attention_implementation="eager",
            memory_slots=2,
            lora_rank=2,
            lora_alpha=2,
        ),
        ObjectiveConfig(
            method="dynamic" if stage == "pretrain" else "memory_change",
            stage=stage,
            append_slots=1,
        ),
        TrainingConfig(
            dataset_dir="data/unit-fixture",
            output_dir=str(output),
            epochs=2,
            gradient_accumulation_steps=2,
            learning_rate=0.01,
            weight_decay=0.0,
            gradient_clip=10.0,
            eval_every=1,
            save_every=1,
            seed=123,
            init_checkpoint=init,
        ),
    )


def make_splits(prefix="pretrain"):
    return {
        split: tuple(
            Example(
                f"{prefix}-{split}-{i}", f"{prefix}-cluster-{split}-{i}", (i + 1.0, -0.5), i / 4
            )
            for i in range(count)
        )
        for split, count in (("train", 5), ("dev", 2), ("test", 1))
    }


def make_engine(config):
    engine = TokenMemoryEngine(TinyTask(), config.training, "cpu")
    engine.initialize()
    return engine


def _run(config, engine, splits, initialization=None):
    statistics = {"source_data": {name: dataset_identity(rows) for name, rows in splits.items()}}
    return make_run(config, splits, statistics, "cpu", 1, initialization=initialization)


def checkpoint_handler(engine, run, cursor, directory=None, resume=None, rows=None):
    loader = make_data_loader(
        make_splits()["train"] if rows is None else rows,
        engine.global_batch_size,
        engine.config.seed,
    )
    return TokenMemoryCheckpointHandler(
        engine,
        loader,
        directory or Path(engine.config.output_dir) / "checkpoints",
        run,
        cursor,
        resume_from_path=resume,
    )


def restore_checkpoint(path, engine, run):
    cursor = {}
    handler = checkpoint_handler(engine, run, cursor, directory=Path(path).parent, resume=path)
    assert handler.load_checkpoint() == cursor["step"]
    return cursor


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_checkpoint_retention_uses_numeric_steps_and_ignores_incomplete_directories(
    tmp_path, limit
):
    config = make_config(tmp_path / "run")
    config = replace(config, training=replace(config.training, save_total_limit=4))
    engine, splits = make_engine(config), make_splits()
    run = _run(config, engine, splits)
    directory = tmp_path / "run/checkpoints"
    unfinished = directory / "global_step_999"
    unfinished.mkdir(parents=True)
    temporary = unfinished / "state.tmp"
    temporary.write_bytes(b"unfinished checkpoint")
    cursor = {"epoch": 0, "sample_offset": 0, "step": 0, "sample_visits": 0}
    handler = checkpoint_handler(engine, run, cursor, directory)
    for step in (1, 2, 3, 10):
        cursor["step"] = step
        handler.save_checkpoint(step)
    engine.reset_optimizer(replace(config.training, save_total_limit=limit))
    cursor["step"] = 11
    checkpoint_handler(engine, run, cursor, directory).save_checkpoint(11)

    retained = sorted(
        read_checkpoint(path)["cursor"]["step"]
        for path in directory.glob("global_step_*")
        if (path / "state.pt").is_file()
    )
    assert retained == [1, 2, 3, 10, 11][-limit:]
    assert temporary.read_bytes() == b"unfinished checkpoint"


def test_checkpoint_retention_is_stage_local_and_uses_updated_engine_config(tmp_path):
    pretrain = make_config(tmp_path / "method/pretrain")
    pretrain = replace(pretrain, training=replace(pretrain.training, save_total_limit=3))
    engine, splits = make_engine(pretrain), make_splits()
    run = _run(pretrain, engine, splits)
    source = tmp_path / "method/pretrain/checkpoints"
    cursor = {"step": 0}
    handler = checkpoint_handler(engine, run, cursor, source)
    for step in (1, 2, 3):
        cursor["step"] = step
        handler.save_checkpoint(step)

    warmup = make_config(tmp_path / "method/warmup", "warmup", str(source / "global_step_3"))
    warmup = replace(warmup, training=replace(warmup.training, save_total_limit=1))
    engine.reset_optimizer(warmup.training)
    initialization = load_initialization(source / "global_step_3", engine.model, warmup)
    run = _run(warmup, engine, make_splits("qa"), initialization)
    target = tmp_path / "method/warmup/checkpoints"
    cursor = {"step": 0}
    handler = checkpoint_handler(engine, run, cursor, target)
    for step in (1, 2, 3):
        cursor["step"] = step
        handler.save_checkpoint(step)

    assert sorted(path.name for path in source.glob("global_step_*")) == [
        "global_step_1",
        "global_step_2",
        "global_step_3",
    ]
    assert [path.name for path in target.glob("global_step_*")] == ["global_step_3"]
    assert read_checkpoint(target / "global_step_3")["cursor"]["step"] == 3


def test_nonzero_rank_only_saves_its_loader_without_deleting_model_states(tmp_path, monkeypatch):
    config = make_config(tmp_path / "run")
    engine = make_engine(config)
    engine.rank, engine.world_size = 1, 2
    directory = tmp_path / "run/checkpoints"
    directory.mkdir(parents=True)
    for step in (1, 2, 3):
        checkpoint = directory / f"global_step_{step}"
        checkpoint.mkdir()
        (checkpoint / "state.pt").write_bytes(b"saved by rank zero")
    monkeypatch.setattr(runtime.dist, "all_gather_object", lambda result, value: None)
    monkeypatch.setattr(runtime.dist, "barrier", lambda: None)

    checkpoint_handler(engine, {}, {"step": 4}, directory).save_checkpoint(4)

    assert all(
        (directory / f"global_step_{step}/state.pt").read_bytes() == b"saved by rank zero"
        for step in (1, 2, 3)
    )
    assert not (directory / "global_step_4/state.pt").exists()
    assert (directory / "global_step_4/data_1.pt").is_file()


@pytest.mark.parametrize("failure", ["save", "replace", "loader"])
def test_failed_checkpoint_save_preserves_previous_complete_files(tmp_path, monkeypatch, failure):
    config = make_config(tmp_path / "run")
    engine, splits = make_engine(config), make_splits()
    run = _run(config, engine, splits)
    directory = tmp_path / "run/checkpoints"
    cursor = {"step": 0}
    handler = checkpoint_handler(engine, run, cursor, directory)
    for step in (1, 2):
        cursor["step"] = step
        handler.save_checkpoint(step)
    original = {
        str(path.relative_to(directory)): path.read_bytes()
        for checkpoint in (directory / "global_step_1", directory / "global_step_2")
        for path in checkpoint.iterdir()
    }
    original_save = torch.save

    if failure in {"save", "loader"}:

        def failed_save(value, path):
            target = "state.tmp" if failure == "save" else "data_0.pt"
            if Path(path).name != target:
                return original_save(value, path)
            Path(path).write_bytes(b"partial checkpoint")
            raise OSError("checkpoint save interrupted")

        monkeypatch.setattr(runtime.torch, "save", failed_save)
    else:

        def failed_replace(self, target):
            raise OSError("checkpoint replace interrupted")

        monkeypatch.setattr(Path, "replace", failed_replace)

    with pytest.raises(OSError, match="interrupted"):
        cursor["step"] = 3
        handler.save_checkpoint(3)

    assert {name: (directory / name).read_bytes() for name in original} == original
    assert (directory / "global_step_3").is_dir()
    assert [
        read_checkpoint(directory / f"global_step_{step}")["cursor"]["step"] for step in (1, 2)
    ] == [1, 2]


def test_interrupted_training_retains_two_recent_readable_checkpoints(tmp_path, monkeypatch):
    config, splits = make_config(tmp_path / "run"), make_splits()
    engine = make_engine(config)
    run = _run(config, engine, splits)
    original_step = engine.step
    calls = 0

    def interrupted_step(batch, epoch=0):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("training interrupted")
        return original_step(batch, epoch=epoch)

    monkeypatch.setattr(engine, "step", interrupted_step)
    with pytest.raises(RuntimeError, match="training interrupted"):
        train_loop(config, engine, splits, run)

    directory = tmp_path / "run/checkpoints"
    assert sorted(path.name for path in directory.glob("global_step_*")) == [
        "global_step_2",
        "global_step_3",
    ]
    for step in (2, 3):
        restored = make_engine(config)
        cursor = restore_checkpoint(directory / f"global_step_{step}", restored, run)
        assert cursor["step"] == step
        assert restored.optimizer.state


@pytest.mark.parametrize("stage", ["pretrain", "warmup", "policy"])
def test_checkpoint_requires_explicit_append_size_only_after_dynamic_pretraining(tmp_path, stage):
    config = make_config(tmp_path / "source")
    engine, splits = make_engine(config), make_splits()
    result = train_loop(config, engine, splits, _run(config, engine, splits), stop_after_steps=1)
    saved = read_checkpoint(result["checkpoint"])
    saved["run"]["config"]["objective"]["stage"] = stage
    saved["run"]["config"]["objective"]["method"] = "memory_change"
    del saved["run"]["config"]["objective"]["append_slots"]
    torch.save(saved, Path(result["checkpoint"]) / "state.pt")
    if stage == "pretrain":
        target = make_config(tmp_path / "target", "warmup", result["checkpoint"])
        restored = make_engine(target)
        load_initialization(result["checkpoint"], restored.model, target)
        torch.testing.assert_close(restored.model.weight, engine.model.weight)
    else:
        with pytest.raises(ValueError, match="explicit objective.append_slots"):
            read_checkpoint(result["checkpoint"])


@pytest.mark.parametrize("micro_batch_size,accumulation", [(1, 2), (2, 1)])
@pytest.mark.parametrize("stop_step", [2, 3])
def test_checkpoint_resume_matches_uninterrupted_rng_optimizer_and_tail_batch(
    tmp_path, micro_batch_size, accumulation, stop_step
):
    splits = make_splits()
    full_config = make_config(tmp_path / "full")
    full_config = replace(
        full_config,
        training=replace(
            full_config.training,
            micro_batch_size_per_gpu=micro_batch_size,
            gradient_accumulation_steps=accumulation,
        ),
    )
    set_seed(11)
    full = make_engine(full_config)
    expected = train_loop(full_config, full, splits, _run(full_config, full, splits))
    expected_rng = capture_rng(torch.device("cpu"))

    partial_config = make_config(tmp_path / "resumed")
    partial_config = replace(
        partial_config,
        training=replace(
            partial_config.training,
            micro_batch_size_per_gpu=micro_batch_size,
            gradient_accumulation_steps=accumulation,
        ),
    )
    set_seed(11)
    partial = make_engine(partial_config)
    run = _run(partial_config, partial, splits)
    stopped = train_loop(partial_config, partial, splits, run, stop_after_steps=stop_step)
    assert not stopped["complete"]
    checkpoint = read_checkpoint(stopped["checkpoint"])
    assert run["global_batch_size"] == checkpoint["run"]["global_batch_size"] == 2
    assert set(checkpoint) == {"run", "trainable", "optimizer", "cursor", "rng"}
    assert checkpoint["cursor"] == (
        {"epoch": 0, "sample_offset": 4, "step": 2, "sample_visits": 4}
        if stop_step == 2
        else {"epoch": 1, "sample_offset": 0, "step": 3, "sample_visits": 5}
    )

    set_seed(999)
    resumed = make_engine(partial_config)
    result = train_loop(partial_config, resumed, splits, run, resume=stopped["checkpoint"])
    actual_rng = capture_rng(torch.device("cpu"))
    assert result["complete"]
    assert result["completed_steps"] == expected["completed_steps"] == 6
    assert result["sample_visits"] == 10
    torch.testing.assert_close(resumed.model.weight, full.model.weight, rtol=0, atol=0)
    torch.testing.assert_close(
        resumed.optimizer.state_dict(), full.optimizer.state_dict(), rtol=0, atol=0
    )
    assert actual_rng["python"] == expected_rng["python"]
    np.testing.assert_equal(actual_rng["numpy"], expected_rng["numpy"])
    torch.testing.assert_close(actual_rng["torch"], expected_rng["torch"], rtol=0, atol=0)
    full_records = [
        json.loads(line) for line in (tmp_path / "full/metrics.jsonl").read_text().splitlines()
    ]
    resumed_records = [
        json.loads(line) for line in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()
    ]
    assert [row["train/epoch"] for row in full_records] == [0.4, 0.8, 1.0, 1.4, 1.8, 2.0]
    assert [row["epoch"] for row in full_records] == [1, 1, 1, 2, 2, 2]
    assert [
        {key: value for key, value in row.items() if not key.startswith("resources/")}
        for row in full_records
    ] == [
        {key: value for key, value in row.items() if not key.startswith("resources/")}
        for row in resumed_records
    ]
    assert all("train/loss" in row and "dev/loss" in row for row in resumed_records)
    assert all("train/train/loss" not in row for row in resumed_records)
    assert all(row["resources/optimizer_step_seconds"] > 0 for row in resumed_records)
    assert all("resources/peak_memory_allocated_bytes" not in row for row in resumed_records)
    assert not (tmp_path / "resumed/swanlab.json").exists()


def test_resume_from_saved_run_without_explicit_experiment_name(tmp_path):
    config = make_config(tmp_path / "source")
    engine, splits = make_engine(config), make_splits()
    run = _run(config, engine, splits)
    stopped = train_loop(config, engine, splits, run, stop_after_steps=1)
    output = tmp_path / "source"
    saved_config = json.loads((output / "config.json").read_text())
    saved_config["training"].pop("experiment_name", None)
    (output / "config.json").write_text(json.dumps(saved_config))
    saved_run = deepcopy(run)
    saved_run["config"] = saved_config
    (output / "run.json").write_text(json.dumps(saved_run))
    checkpoint = read_checkpoint(stopped["checkpoint"])
    checkpoint["run"] = saved_run
    torch.save(checkpoint, Path(stopped["checkpoint"]) / "state.pt")

    restored_config = load_experiment(output / "config.json")
    restored = make_engine(restored_config)
    result = train_loop(
        restored_config,
        restored,
        splits,
        _run(restored_config, restored, splits),
        resume=stopped["checkpoint"],
    )
    assert result["complete"]
    assert result["sample_visits"] == 10


def test_resume_rejects_changed_config_or_data_before_loading_weights(tmp_path):
    config, splits = make_config(tmp_path / "run"), make_splits()
    engine = make_engine(config)
    run = _run(config, engine, splits)
    result = train_loop(config, engine, splits, run, stop_after_steps=1)
    fresh = make_engine(config)
    before = fresh.model.weight.detach().clone()
    changed = replace(config, training=replace(config.training, learning_rate=0.02))
    with pytest.raises(ValueError, match="resume configuration"):
        restore_checkpoint(result["checkpoint"], fresh, _run(changed, fresh, splits))
    different = {**splits, "test": (replace(splits["test"][0], target=9.0),)}
    with pytest.raises(ValueError, match="resume configuration"):
        restore_checkpoint(result["checkpoint"], fresh, _run(config, fresh, different))
    torch.testing.assert_close(fresh.model.weight, before, rtol=0, atol=0)
    assert dataset_identity(splits["test"]) != dataset_identity(different["test"])


def test_early_stop_always_evaluates_and_saves_at_the_actual_last_step(tmp_path):
    config = make_config(tmp_path / "smoke")
    config = replace(config, training=replace(config.training, eval_every=25, save_every=25))
    splits = make_splits()
    engine = make_engine(config)

    result = train_loop(config, engine, splits, _run(config, engine, splits), stop_after_steps=2)

    assert not result["complete"]
    assert result["completed_steps"] == result["stop_after_steps"] == 2
    records = [
        json.loads(line) for line in (tmp_path / "smoke/metrics.jsonl").read_text().splitlines()
    ]
    assert "dev/loss" not in records[0]
    assert records[1]["step"] == 2 and records[1]["dev/samples"] == 2
    assert read_checkpoint(result["checkpoint"])["cursor"]["step"] == 2
    assert result["checkpoint"].endswith("global_step_2")


@pytest.mark.parametrize("source_checkpointing", [True, False, None])
def test_stage_initialization_loads_only_trainable_weights_and_preserves_ancestry(
    tmp_path, source_checkpointing
):
    pretrain, splits = make_config(tmp_path / "pretrain"), make_splits()
    engine = make_engine(pretrain)
    result = train_loop(
        pretrain, engine, splits, _run(pretrain, engine, splits), stop_after_steps=1
    )
    source = read_checkpoint(result["checkpoint"])
    if source_checkpointing is None:
        del source["run"]["config"]["model"]["gradient_checkpointing"]
    else:
        source["run"]["config"]["model"]["gradient_checkpointing"] = source_checkpointing
    torch.save(source, Path(result["checkpoint"]) / "state.pt")
    warmup = make_config(tmp_path / "warmup", "warmup", result["checkpoint"])
    fresh = make_engine(warmup)
    torch_rng, python_rng, numpy_rng = (
        torch.get_rng_state(),
        random.getstate(),
        np.random.get_state(),
    )
    initialization = load_initialization(result["checkpoint"], fresh.model, warmup)
    torch.testing.assert_close(fresh.model.weight, engine.model.weight)
    assert not fresh.optimizer.state
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    qa_splits = make_splits("qa")
    run = _run(warmup, fresh, qa_splits, initialization)
    assert run["pretraining_sources"]["document_ids"] == sorted(
        row.document_id for rows in splits.values() for row in rows
    )
    warmup_path = tmp_path / "warmup/checkpoints/global_step_0"
    checkpoint_handler(
        fresh, run, {"epoch": 0, "sample_offset": 0, "step": 0, "sample_visits": 0}
    ).save_checkpoint(0)
    policy = make_config(tmp_path / "policy", "policy", str(warmup_path))
    policy_init = load_initialization(warmup_path, TinyTask(), policy)
    assert policy_init["pretraining_sources"] == initialization["pretraining_sources"]
    assert (
        _run(policy, fresh, qa_splits, policy_init)["pretraining_sources"]
        == run["pretraining_sources"]
    )

    different_model = replace(warmup, model=replace(warmup.model, memory_slots=3))
    with pytest.raises(ValueError, match="model configuration"):
        load_initialization(result["checkpoint"], fresh.model, different_model)
    with pytest.raises(ValueError, match="warmup initialization requires"):
        load_initialization(warmup_path, fresh.model, warmup)


@pytest.mark.parametrize("field", ["document_id", "dedup_cluster"])
def test_source_isolation_checks_all_qa_splits_and_both_identity_keys(tmp_path, field):
    splits = make_splits()
    config = make_config(tmp_path / "qa", "warmup", "parent.pt")
    initialization = {
        "pretraining_sources": {"document_ids": ["old-document"], "dedup_clusters": ["old-cluster"]}
    }
    overlap = "old-document" if field == "document_id" else "old-cluster"
    splits["test"] = (replace(splits["test"][0], **{field: overlap}),)
    with pytest.raises(ValueError, match="QA sources overlap"):
        _run(config, make_engine(config), splits, initialization)


def test_finetuning_requires_initialization_and_cursor_contract(tmp_path):
    config, splits = make_config(tmp_path / "warmup", "warmup"), make_splits()
    engine = make_engine(config)
    with pytest.raises(ValueError, match="requires an initialization checkpoint"):
        train_loop(config, engine, splits, _run(config, engine, splits))
    with pytest.raises(ValueError, match="epoch/batch schedule"):
        validate_cursor(
            {"epoch": 0, "sample_offset": 3, "step": 1, "sample_visits": 3}, config.training, 5, 1
        )


def test_load_codec_uses_a_local_tiny_pretrained_base_without_network(tmp_path):
    model_dir = tmp_path / "tiny"
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    base.save_pretrained(model_dir)
    backend = Tokenizer(
        WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}, unk_token="[UNK]")
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(model_dir)
    model_config = replace(
        make_config(tmp_path / "unused").model, model_name_or_path=str(model_dir)
    )
    objective = ObjectiveConfig(method="icae_single", stage="pretrain")
    codec, loaded_tokenizer = load_codec(model_config, torch.device("cpu"), objective)
    assert codec.memory_embeddings.shape == (2, 16)
    assert codec.memory_embeddings.dtype == torch.float32
    assert loaded_tokenizer.eos_token_id == 2
    assert codec.language_model.get_base_model().is_gradient_checkpointing
    assert codec.decoder.is_gradient_checkpointing
    assert all(not parameter.requires_grad for parameter in codec.decoder.parameters())
    writer_embedding = codec.language_model.get_input_embeddings().weight
    reader_embedding = codec.decoder.get_input_embeddings().weight
    torch.testing.assert_close(writer_embedding, reader_embedding)
    assert writer_embedding.data_ptr() != reader_embedding.data_ptr()
    assert any(
        "lora_" in name and parameter.requires_grad for name, parameter in codec.named_parameters()
    )
    plain, _ = load_codec(
        replace(model_config, gradient_checkpointing=False), torch.device("cpu"), objective
    )
    assert not plain.language_model.get_base_model().is_gradient_checkpointing
    assert not plain.decoder.is_gradient_checkpointing
    multi, _ = load_codec(
        replace(model_config, memory_slots=5),
        torch.device("cpu"),
        ObjectiveConfig(method="icae_multi", icae_min_segments=3, icae_max_segments=5),
    )
    assert multi.memory_slots == 5
    assert multi.write_slots == 2
    assert multi.memory_embeddings.shape == (2, 16)
    ac, _ = load_codec(
        replace(model_config, memory_slots=5),
        torch.device("cpu"),
        ObjectiveConfig(method="autocompressors", stage="lm", ac_num_segments=4),
    )
    assert ac.memory_slots == 5
    assert ac.write_slots == 2
    assert ac.memory_embeddings.shape == (2, 16)
    tag, tag_tokenizer = load_codec(
        model_config,
        torch.device("cpu"),
        ObjectiveConfig(method="memory_change", stage="warmup", writer_mode="tag", tag_tokens=5),
    )
    assert tag.tag_embeddings.shape == (2, 5, 16)
    assert tag.tag_embeddings.dtype == torch.float32
    assert tag.tag_embeddings.requires_grad
    assert tag_tokenizer.get_vocab() == loaded_tokenizer.get_vocab()
    tag_embedding = tag.language_model.get_input_embeddings().weight
    torch.testing.assert_close(tag_embedding, writer_embedding)
    assert not tag_embedding.requires_grad
    assert all(not parameter.requires_grad for parameter in tag.decoder.parameters())
    tokenizer.eos_token = None
    tokenizer.save_pretrained(model_dir)
    with pytest.raises(ValueError, match="eos_token_id"):
        load_codec(model_config, torch.device("cpu"), objective)


def _distributed_training_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        config, splits = make_config(output), make_splits()
        config = replace(
            config, training=replace(config.training, epochs=1, gradient_accumulation_steps=1)
        )
        set_seed(11)
        engine = make_engine(config)
        statistics = {
            "source_data": {name: dataset_identity(rows) for name, rows in splits.items()}
        }
        run = make_run(config, splits, statistics, "cpu", 2)
        partial = train_loop(config, engine, splits, run, stop_after_steps=1)
        assert partial["completed_steps"] == 1
        checkpoint = read_checkpoint(partial["checkpoint"])
        assert len(checkpoint["rng"]) == 2
        with pytest.raises(ValueError, match="empty output directory"):
            train_loop(config, engine, splits, run)
        resumed = make_engine(config)
        complete = train_loop(config, resumed, splits, run, resume=partial["checkpoint"])
        assert complete["complete"]
        assert complete["completed_steps"] == 3
        assert complete["sample_visits"] == 5
        weights = [torch.empty_like(resumed.model.weight) for _ in range(2)]
        dist.all_gather(weights, resumed.model.weight.detach())
        torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_distributed_startup_checkpoint_and_resume_with_empty_tail_rank(tmp_path):
    mp.spawn(
        _distributed_training_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "distributed")),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("filtered_split", ["train", "dev"])
def test_pretraining_accepts_a_single_objective_after_actual_length_filtering(
    tmp_path, filtered_split
):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    for split in ("train", "dev", "test"):
        rows = []
        for task in ("ae", "continuation") if split != "test" else ("ae",):
            text = "a b c" if split == filtered_split and task == "continuation" else "a b"
            rows.append(pretraining_record(split, f"{split}-{task}", task, text))
        (tmp_path / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_pretraining_metadata(tmp_path)
    config = make_config(tmp_path / "output")
    config = replace(
        config,
        training=replace(
            config.training, dataset_dir=str(tmp_path), max_input_tokens=3, lm_target_tokens=1
        ),
    )
    splits, _ = load_splits(config, tokenizer, 64)
    assert {row.task for row in splits["test"]} == {"ae"}
    restricted = replace(config, training=replace(config.training, max_input_tokens=2))
    splits, _ = load_splits(restricted, tokenizer, 64)
    expected_tasks = {"ae"} if filtered_split == "train" else {"ae", "continuation"}
    assert {row.task for row in splits[filtered_split]} == expected_tasks
    lm = replace(
        restricted,
        objective=ObjectiveConfig(method="autocompressors", stage="lm", ac_num_segments=1),
    )
    splits, _ = load_splits(lm, tokenizer, 64)
    assert len(splits[filtered_split]) == (1 if filtered_split == "train" else 2)
    assert {row.task for row in splits[filtered_split]} == {"continuation"}


def test_tracking_context_uses_explicit_credentials_only_on_root_rank(tmp_path, monkeypatch):
    config = make_config(tmp_path / "output")
    config = replace(
        config,
        training=replace(
            config.training, swanlab_project="explicit-project", group="explicit-group"
        ),
    )
    run = _run(config, make_engine(config), make_splits())
    original = deepcopy(run)
    calls = []

    def capture_upload(settings, record, device, api_key):
        calls.append((settings, record, device, api_key))
        return nullcontext(None)

    monkeypatch.setattr(runtime, "method_tracking_run", capture_upload)
    monkeypatch.setattr(runtime, "swanlab_api_key", lambda: "training-test-key")
    with runtime._tracking_context(
        config, SimpleNamespace(rank=0, device=torch.device("cpu")), run
    ):
        pass
    assert calls == [(config, run, torch.device("cpu"), "training-test-key")]
    with runtime._tracking_context(config, SimpleNamespace(rank=1), run):
        pass
    assert len(calls) == 1
    assert "training-test-key" not in json.dumps(run)
    assert run == original


def pretraining_record(split, identifier, task, text="a b"):
    suffix = 0
    while True:
        trajectory_id = f"{identifier}-{suffix}"
        sampled_task = (
            "continuation"
            if random.Random(f"123:pretraining:{trajectory_id}").random() < 0.5
            else "ae"
        )
        if sampled_task == task:
            break
        suffix += 1
    first, rest = text.split(" ", 1)
    first, rest = first.ljust(4), rest.ljust(4)
    body = first + rest
    return {
        "trajectory_id": trajectory_id,
        "document_id": identifier,
        "dedup_cluster": f"{identifier}:cluster",
        "split": split,
        "source": {"file": f"{identifier}.parquet", "row_group": 0, "row_index": 0},
        "window_char_span": [0, len(body)],
        "text": body,
        "segments": [
            {"segment_id": "seg0", "char_span": [0, 4]},
            {"segment_id": "seg1", "char_span": [4, len(body)]},
        ],
        "continuation": "c".ljust(8),
        "text_char_length": len(body),
        "estimated_tokens": len(body) / 4,
        "estimated_tokens_rule": "len(text) / 4",
    }


def write_pretraining_metadata(path):
    window = SegmentationConfig(
        capacity=1,
        min_segment_ratio=1,
        max_segment_ratio=4,
        min_segments=2,
        max_segments=2,
        continuation_tokens=2,
    )
    (path / "preparation.json").write_text(json.dumps({"config": {"window": asdict(window)}}))


def pretraining_corpus(path):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    for split in ("train", "dev", "test"):
        records = []
        for task, count in (("ae", 8), ("continuation", 4)):
            for index in range(count):
                records.append(pretraining_record(split, f"{split}-{task}-{index}", task))
        (path / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    write_pretraining_metadata(path)
    return tokenizer


def test_sample_limits_keep_both_pretraining_tasks_and_full_test_source_identity(tmp_path):
    tokenizer = pretraining_corpus(tmp_path)
    config = make_config(tmp_path / "output")
    config = replace(
        config,
        training=replace(
            config.training,
            dataset_dir=str(tmp_path),
            lm_target_tokens=1,
            max_qa_input_tokens=1,
        ),
    )
    full, full_statistics = load_splits(config, tokenizer, 64)
    limited = replace(
        config,
        training=replace(config.training, max_train_samples=4, max_dev_samples=3),
    )
    splits, statistics = load_splits(limited, tokenizer, 64)

    assert {name: len(rows) for name, rows in splits.items()} == {"train": 4, "dev": 3, "test": 12}
    assert statistics["splits"]["train"]["selected_by_task"] == {"ae": 3, "continuation": 1}
    assert statistics["splits"]["dev"]["selected_by_task"] == {"ae": 2, "continuation": 1}
    assert statistics["splits"]["train"]["kept"] == 12
    assert statistics["splits"]["train"]["selected"] == 4
    assert statistics["splits"]["train"]["selected_input_tokens"] == 8
    assert statistics["source_data"] == full_statistics["source_data"]
    assert splits["test"] == full["test"]
    assert load_splits(limited, tokenizer, 64) == (splits, statistics)
    other_method = replace(limited, objective=replace(limited.objective, method="information_loss"))
    assert load_splits(other_method, tokenizer, 64) == (splits, statistics)
    changed_seed = replace(limited, training=replace(limited.training, seed=999))
    assert load_splits(changed_seed, tokenizer, 64)[0]["train"] != splits["train"]
    run = make_run(limited, splits, statistics, "cpu", 1)
    assert run["source_data"]["train"] == dataset_identity(full["train"])
    assert run["data"]["train"] == dataset_identity(splits["train"])
    assert run["source_data"]["train"] != run["data"]["train"]


@pytest.mark.parametrize("name", ["max_train_samples", "max_dev_samples"])
def test_pretraining_sample_limit_can_keep_one_source_without_forcing_both_objectives(
    tmp_path, name
):
    tokenizer = pretraining_corpus(tmp_path)
    config = make_config(tmp_path / "output")
    config = replace(
        config,
        training=replace(config.training, dataset_dir=str(tmp_path), **{name: 1}),
    )
    splits, statistics = load_splits(config, tokenizer, 64)
    split = "train" if name == "max_train_samples" else "dev"
    assert len(splits[split]) == statistics["splits"][split]["selected"] == 1
    assert sum(statistics["splits"][split]["selected_by_task"].values()) == 1
    assert load_splits(config, tokenizer, 64) == (splits, statistics)


def test_selection_follows_full_canonical_validation(tmp_path):
    tokenizer = pretraining_corpus(tmp_path)
    path = tmp_path / "train.jsonl"
    with path.open("a") as stream:
        stream.write(path.read_text().splitlines()[0] + "\n")
    config = make_config(tmp_path / "output")
    config = replace(
        config,
        training=replace(config.training, dataset_dir=str(tmp_path), max_train_samples=2),
    )
    with pytest.raises(ValueError, match="duplicate pretraining trajectory_id"):
        load_splits(config, tokenizer, 64)


def test_qa_selection_keeps_complete_trajectories_and_original_test_split(tmp_path, monkeypatch):
    original = {
        split: tuple(
            replace(
                trajectory(n=3),
                trajectory_id=f"{split}-{index}",
                document_id=f"{split}-doc-{index}",
                split=split,
            )
            for index in range(8)
        )
        for split in ("train", "dev", "test")
    }
    monkeypatch.setattr(runtime, "load_factqa", lambda path, tokenizer: dict(original))
    config = make_config(tmp_path / "output", "policy", "parent.pt")
    config = replace(
        config, training=replace(config.training, max_train_samples=3, max_dev_samples=2)
    )
    splits, statistics = load_splits(config, None, 64)
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 3, "dev": 2, "test": 8}
    assert splits["test"] is original["test"]
    for split in ("train", "dev"):
        for row in splits[split]:
            assert any(row is original_row for original_row in original[split])
        assert statistics["splits"][split]["selected_questions"] == sum(
            len(row.qas) for row in splits[split]
        )
        assert statistics["source_data"][split] == dataset_identity(original[split])
    changed = replace(config, objective=replace(config.objective, method="information_loss"))
    assert load_splits(changed, None, 64)[0] == splits


@pytest.mark.parametrize(
    "method,stage",
    [
        ("icae_single", "qa"),
        ("icae_multi", "qa"),
        ("memory_change", "warmup"),
        ("memory_change", "policy"),
        ("information_loss", "warmup"),
        ("information_loss", "policy"),
    ],
)
def test_factqa_train_length_filter_precedes_selection_and_preserves_dev_test_source(
    tmp_path, monkeypatch, method, stage
):
    original = {
        split: tuple(
            replace(
                trajectory(split=split),
                trajectory_id=f"{split}-{index}",
                document_id=f"{split}-doc-{index}",
                full_input_ids=tuple(range(length)),
            )
            for index, length in enumerate((6, 9, 10, 12) * 2)
        )
        for split in ("train", "dev", "test")
    }
    monkeypatch.setattr(runtime, "load_factqa", lambda path, tokenizer: original)
    config = make_config(tmp_path / "output")
    config = replace(
        config,
        objective=replace(
            config.objective, method=method, stage=stage, icae_min_segments=2, icae_max_segments=2
        ),
        training=replace(
            config.training,
            max_input_tokens=1,
            max_qa_input_tokens=9,
            max_train_samples=3,
            max_dev_samples=2,
        ),
    )
    splits, statistics = load_splits(config, None, 64)
    assert statistics["kind"] == "factqa"
    assert statistics["upper_limit_split"] == "train"
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 3, "dev": 2, "test": 8}
    assert splits["test"] is original["test"]
    for split, rows in original.items():
        filtered = (
            tuple(row for row in rows if len(row.full_input_ids) <= 9) if split == "train" else rows
        )
        assert statistics["source_data"][split] == dataset_identity(filtered)
        counts = statistics["splits"][split]
        assert counts["read"] == 8
        assert counts["kept"] == counts["trajectories"] == len(filtered)
        assert counts["filtered_too_long"] == (4 if split == "train" else 0)
        assert counts["source_tokens"] == sum(len(row.full_input_ids) for row in filtered)
        assert counts["questions"] == sum(len(row.qas) for row in filtered)
        assert counts["selected"] == len(splits[split])
        assert counts["selected_source_tokens"] == sum(
            len(row.full_input_ids) for row in splits[split]
        )
        for selected in splits[split]:
            assert any(selected is row for row in filtered)
            assert len(selected.segments) == 3
            assert len(selected.qas) == 6
            assert len(selected.usage) == 3
    changed = replace(config, training=replace(config.training, max_input_tokens=64))
    assert load_splits(changed, None, 64) == (splits, statistics)
    run = make_run(config, splits, statistics, "cpu", 1)
    assert run["source_data"]["train"] != run["data"]["train"]
    assert run["data_statistics"]["splits"]["train"]["filtered_too_long"] == 4
    assert run["source_data"]["dev"] == dataset_identity(original["dev"])
    assert run["source_data"]["test"] == run["data"]["test"] == dataset_identity(original["test"])


@pytest.mark.parametrize("max_qa_input_tokens", [None, 8, 12])
def test_factqa_train_length_filter_uses_real_model_window(
    tmp_path, monkeypatch, max_qa_input_tokens
):
    original = {
        split: (trajectory(split=split, n=2), trajectory(split=split, n=3))
        for split in ("train", "dev", "test")
    }
    monkeypatch.setattr(runtime, "load_factqa", lambda path, tokenizer: original)
    config = make_config(tmp_path / "output", "policy")
    config = replace(
        config,
        training=replace(config.training, max_qa_input_tokens=max_qa_input_tokens),
    )
    splits, statistics = load_splits(config, None, 8)
    assert splits["train"] == (original["train"][0],)
    assert statistics["splits"]["train"]["filtered_too_long"] == 1
    for split in ("dev", "test"):
        assert splits[split] is original[split]
        assert len(splits[split][-1].full_input_ids) > 8
        assert statistics["splits"][split]["filtered_too_long"] == 0
        assert statistics["source_data"][split] == dataset_identity(original[split])


def experiment_config(root, stage, identifier="trial", method=None, init=None):
    config = make_config(root / stage)
    return replace(
        config,
        objective=replace(
            config.objective,
            method=method or ("dynamic" if stage == "pretrain" else "memory_change"),
            stage=stage,
            icae_min_segments=2,
            icae_max_segments=2,
            ac_num_segments=2,
        ),
        training=replace(
            config.training,
            experiment_dir=str(root),
            experiment_id=identifier,
            init_checkpoint=init,
        ),
    )


@pytest.mark.parametrize("explicit_name", [False, True])
def test_dynamic_stages_preserve_shared_source_and_accumulate_only_method_steps(
    tmp_path, explicit_name
):
    source_root = tmp_path / ("dynamic-pretrain" if explicit_name else "dynamic-pretrain_trial")
    source_config = experiment_config(source_root, "pretrain")
    if explicit_name:
        source_config = replace(
            source_config,
            training=replace(source_config.training, experiment_name="dynamic-pretrain_trial"),
        )
    source_engine = make_engine(source_config)
    pretraining = make_splits()
    source_result = train_loop(
        source_config,
        source_engine,
        pretraining,
        _run(source_config, source_engine, pretraining),
        stop_after_steps=2,
    )
    assert source_result["global_step"] == 2
    qa = make_splits("qa")
    sources = []
    for method in ("memory_change", "information_loss"):
        root = tmp_path / f"{method}_trial"
        config = experiment_config(root, "warmup", method=method, init=source_result["checkpoint"])
        engine = make_engine(config)
        initialization = load_initialization(source_result["checkpoint"], engine.model, config)
        sources.append(initialization["pretraining"])
        run = _run(config, engine, qa, initialization)
        assert run["step_offset"] == 0
        result = train_loop(config, engine, qa, run, stop_after_steps=2)
        assert result["global_step"] == result["completed_steps"] == 2
        config = experiment_config(root, "policy", method=method, init=result["checkpoint"])
        engine = make_engine(config)
        initialization = load_initialization(result["checkpoint"], engine.model, config)
        assert initialization["stage"] == "warmup"
        assert initialization["pretraining"] == sources[-1]
        run = _run(config, engine, qa, initialization)
        assert run["step_offset"] == 2
        result = train_loop(config, engine, qa, run, stop_after_steps=1)
        assert result["global_step"] == 3 and result["completed_steps"] == 1
        checkpoint = read_checkpoint(result["checkpoint"])
        assert checkpoint["run"]["pretraining"] == sources[-1]
        assert checkpoint["run"]["step_offset"] == 2
        restored = make_engine(config)
        result = train_loop(
            config,
            restored,
            qa,
            _run(config, restored, qa, initialization),
            resume=result["checkpoint"],
            stop_after_steps=2,
        )
        assert result["global_step"] == 4 and result["completed_steps"] == 2
        records = [
            json.loads(line) for line in (root / "policy/metrics.jsonl").read_text().splitlines()
        ]
        assert [row["global_step"] for row in records] == [3, 4]
        assert [row["step"] for row in records] == [1, 2]
        assert all(row["stage"] == "policy" for row in records)
        manifest = json.loads((root / "experiment.json").read_text())
        assert list(manifest["stages"]) == ["warmup", "policy"]
        assert manifest["pretraining"] == sources[-1]
        assert "pretraining_sources" not in json.dumps(manifest)
    assert sources[0] == sources[1]
    assert sources[0]["experiment_id"] == "trial"
    assert sources[0]["run_name"] == "dynamic-pretrain_trial"
    assert sources[0]["run_dir"] == str(source_root.resolve())
    assert sources[0]["step"] == 2
    assert sources[0]["run_id"] is sources[0]["run_url"] is None


def test_saved_pretraining_without_explicit_name_uses_its_directory_name(tmp_path):
    root = tmp_path / "shared-pretrain-k64_trial"
    source = experiment_config(root, "pretrain", method="memory_change")
    source_run = _run(source, make_engine(source), make_splits())
    source_run["config"]["training"].pop("experiment_name", None)
    target = experiment_config(tmp_path / "memory-change-k64", "warmup")
    initialization = runtime.initialization_record(
        root / "pretrain/checkpoints/global_step_2", source_run, 2, target
    )
    assert initialization["pretraining"]["run_name"] == root.name
    assert initialization["pretraining"]["run_dir"] == str(root.resolve())


def test_icae_qa_offset_includes_pretraining_in_same_method_run(tmp_path):
    root = tmp_path / "icae-single_trial"
    pretrain = experiment_config(root, "pretrain", method="icae_single")
    engine = make_engine(pretrain)
    data = make_splits()
    result = train_loop(pretrain, engine, data, _run(pretrain, engine, data), stop_after_steps=2)
    qa = experiment_config(root, "qa", method="icae_single", init=result["checkpoint"])
    next_engine = make_engine(qa)
    initialization = load_initialization(result["checkpoint"], next_engine.model, qa)
    data = make_splits("qa")
    run = _run(qa, next_engine, data, initialization)
    assert run["step_offset"] == 2
    result = train_loop(qa, next_engine, data, run, stop_after_steps=1)
    assert result["global_step"] == 3
    manifest = json.loads((root / "experiment.json").read_text())
    assert list(manifest["stages"]) == ["pretrain", "qa"]
    assert manifest["method"] == "icae_single"


@pytest.mark.parametrize("source_id", [None, "other-trial"])
def test_dynamic_source_experiment_identity_must_match_before_loading_weights(tmp_path, source_id):
    source = (
        make_config(tmp_path / "legacy-source")
        if source_id is None
        else experiment_config(tmp_path / "source", "pretrain", identifier=source_id)
    )
    engine = make_engine(source)
    data = make_splits()
    result = train_loop(source, engine, data, _run(source, engine, data), stop_after_steps=1)
    target = experiment_config(tmp_path / "method_trial", "warmup", init=result["checkpoint"])
    model = TinyTask()
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="pretraining experiment_id|original pretraining source"):
        load_initialization(result["checkpoint"], model, target)
    torch.testing.assert_close(model.weight, before)
    assert not (tmp_path / "method_trial").exists()
