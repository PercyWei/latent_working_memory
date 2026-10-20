from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    set_seed,
)

from latent_working_memory.runtime import capture_rng, initialize_device
from latent_working_memory.v4.checkpoint import MetaMemoryCheckpointHandler, read_checkpoint
from latent_working_memory.v4.config import ModelConfig, TrainingConfig
from latent_working_memory.v4.data import TokenizedEpisode
from latent_working_memory.v4.engine import MetaLearningEngine
from latent_working_memory.v4.model import StreamingMemoryLM
from latent_working_memory.v4.train import run_training
from latent_working_memory.v4.training_data import make_data_loader


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    directory = tmp_path_factory.mktemp("v4-training-base")
    set_seed(17)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
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
    model.save_pretrained(directory)
    tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "red", "blue", "sky", "water"]
    tokenizer = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    fast.chat_template = (
        "{% for message in messages %}{% if message['role'] != 'system' %}"
        "{{ message['content'].replace('Document:\\n', '').replace('\\n\\nQuestion:\\n', ' ') }} "
        "{% endif %}{% endfor %}"
    )
    fast.save_pretrained(directory)
    return directory


def model_config(base, query_mode="frozen_query_gen"):
    return ModelConfig(
        str(base),
        pending_size=2,
        recent_size=2,
        num_slots=2,
        memory_dim=4,
        query_dim=4,
        value_dim=4,
        num_probes=2,
        query_mode=query_mode,
        inner_steps=2,
        inner_lr=0.2,
    )


def training_config(output, dataset_dir):
    return TrainingConfig(
        dataset_dir=str(dataset_dir),
        output_dir=str(output),
        epochs=2,
        micro_batch_size_per_gpu=1,
        gradient_accumulation_steps=3,
        learning_rate=0.005,
        weight_decay=0.01,
        max_train_data_tokens=16,
        dev_every=1,
        save_every=1,
        save_total_limit=4,
    )


def episodes():
    return (
        TokenizedEpisode("a", (4, 6, 4, 7, 5, 6), (4,)),
        TokenizedEpisode("b", (4, 6, 7, 5, 4, 6), (6, 7, 5)),
        TokenizedEpisode("c", (7, 5, 4, 6, 7, 6), (5, 4, 6, 7)),
        TokenizedEpisode("d", (5, 6, 4, 7, 5, 6), (4, 7)),
    )


def make_engine(base, output, query_mode="frozen_query_gen"):
    set_seed(42)
    model = StreamingMemoryLM(model_config(base, query_mode))
    engine = MetaLearningEngine(model, training_config(output, output), "cpu")
    engine.initialize()
    return engine


def assert_nested_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("query_mode", ["fixed", "frozen_query_gen"])
def test_engine_matches_global_token_weighted_meta_gradient(tiny_base, tmp_path, query_mode):
    torch.set_num_threads(1)
    engine = make_engine(tiny_base, tmp_path, query_mode)
    reference = make_engine(tiny_base, tmp_path, query_mode)
    batch = episodes()[:3]
    base_before = {
        name: value.clone() for name, value in engine.model.backbone.state_dict().items()
    }
    initial_memory = engine.model.trainable_state_dict()
    outputs = [
        reference.model(
            torch.tensor(episode.prompt_ids + episode.target_ids),
            target_starts=[len(episode.prompt_ids)],
        )
        for episode in batch
    ]
    count = sum(output["target_tokens"] for output in outputs)
    loss = sum(output["loss"] * output["target_tokens"] for output in outputs) / count
    reference.optimizer_zero_grad()
    loss.backward()
    reference.optimizer_step()
    metrics = engine.step(batch)
    assert metrics["target_tokens"] == 8
    assert metrics["samples"] == 3
    assert metrics["source_tokens"] == 26
    assert metrics["loss"] == pytest.approx(float(loss.detach()), rel=1e-6)
    for name, state in engine.model.trainable_state_dict().items():
        torch.testing.assert_close(state, reference.model.trainable_state_dict()[name])
    assert any(
        not torch.equal(value, initial_memory[name])
        for name, value in engine.model.trainable_state_dict().items()
    )
    assert all(parameter.grad is not None for parameter in engine.parameters)
    for name, value in engine.model.backbone.state_dict().items():
        torch.testing.assert_close(value, base_before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in engine.model.backbone.parameters())


def test_eval_runs_inner_update_without_mutating_outer_weights_or_gradients(tiny_base, tmp_path):
    engine = make_engine(tiny_base, tmp_path)
    before = engine.model.trainable_state_dict()
    metrics = engine.eval_batch(episodes())
    assert metrics["target_tokens"] == 10
    assert metrics["samples"] == 4
    assert metrics["write_events"] > 0
    assert np.isfinite(metrics["write_loss"])
    assert all(parameter.grad is None for parameter in engine.parameters)
    assert_nested_equal(before, engine.model.trainable_state_dict())
    assert engine.model.training
    assert not engine.model.backbone.training


def document_record(words):
    return {
        "text": " ".join(words[:5]),
        "qas": [{"question": words[5], "answer": " ".join(words[6:])}],
    }


def write_experiment(directory, base, output):
    directory.mkdir(parents=True, exist_ok=True)
    words = {4: "red", 5: "blue", 6: "sky", 7: "water"}
    for name, split_episodes in (("train", episodes()), ("dev", episodes()[:2])):
        rows = []
        for episode in split_episodes:
            tokens = episode.prompt_ids + episode.target_ids
            if name == "dev":
                tokens = tokens[:4][::-1] + tokens[4:]
            rows.append(document_record([words[token] for token in tokens]))
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    path = directory / "experiment.json"
    training = asdict(training_config(directory / output, directory))
    seed = training.pop("seed")
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "model": asdict(model_config(base)),
                "training": training,
            }
        )
    )
    return path


def run(config, resume=None):
    preset = json.loads(config.read_text())
    common_path, train_path = config.parent / "config.json", config.parent / "train.json"
    common_path.write_text(json.dumps({"seed": preset["seed"], "model": preset["model"]}))
    train_path.write_text(
        json.dumps(
            {
                "execution": {
                    "train_only": False,
                    "test_data": str(Path(preset["training"]["dataset_dir"]).resolve()),
                    "max_test_samples": None,
                },
                "training": preset["training"],
            }
        )
    )
    return run_training(
        Namespace(config=common_path, train_config=train_path, device="cpu", resume=resume)
    )


def test_training_resume_is_exact_and_checkpoint_excludes_frozen_backbone(tiny_base, tmp_path):
    full_path = write_experiment(tmp_path / "full", tiny_base, "output")
    resumed_path = write_experiment(tmp_path / "resumed", tiny_base, "output")
    complete = run(full_path)
    run(resumed_path)
    checkpoint = resumed_path.parent / "output" / "checkpoints" / "global_step_1"
    resumed = run(resumed_path, resume=checkpoint)
    assert resumed["complete"] is True
    assert resumed["completed_steps"] == 4
    assert resumed["sample_visits"] == 8
    assert resumed["target_tokens"] == 20
    full = read_checkpoint(complete["checkpoint"])
    restored = read_checkpoint(resumed["checkpoint"])
    assert set(restored) == {"run", "memory", "optimizer", "cursor", "rng"}
    assert not any("backbone" in name for name in restored["memory"])
    for key in ("memory", "optimizer", "cursor", "rng"):
        assert_nested_equal(full[key], restored[key])
    full_metrics = [
        {key: value for key, value in json.loads(line).items() if not key.startswith("resources/")}
        for line in (full_path.parent / "output" / "metrics.jsonl").read_text().splitlines()
    ]
    resumed_metrics = [
        {key: value for key, value in json.loads(line).items() if not key.startswith("resources/")}
        for line in (resumed_path.parent / "output" / "metrics.jsonl").read_text().splitlines()
    ]
    assert full_metrics == resumed_metrics
    assert not (resumed_path.parent / "output" / "tokenizer").exists()


def test_resume_rejects_changed_tokenized_data(tiny_base, tmp_path):
    config = write_experiment(tmp_path, tiny_base, "output")
    result = run(config)
    path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["qas"][0]["answer"] = "blue"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="resume configuration or tokenized data"):
        run(config, resume=result["checkpoint"])


def test_backbone_position_limit_is_checked_before_training(tiny_base, tmp_path):
    config = write_experiment(tmp_path, tiny_base, "output")
    raw = json.loads(config.read_text())
    raw["training"]["max_train_data_tokens"] = 100
    config.write_text(json.dumps(raw))
    (tmp_path / "train.jsonl").write_text(json.dumps(document_record(["red"] * 65)))
    with pytest.raises(ValueError, match="no episodes remain"):
        run(config)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("visible", [None, "", "4,4", "-1", "all"])
def test_cuda_requires_explicit_distinct_physical_gpu_indices(monkeypatch, visible):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(RuntimeError, match="distinct non-negative physical GPU indices"):
        initialize_device("cuda")


def distributed_engine_worker(rank, base, output, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        engine = make_engine(base, output)
        loader = make_data_loader(episodes(), 3, 42, shuffle=False)
        iterator = iter(loader)
        first = engine.step(next(iterator))
        # 各 rank 故意留下不同 RNG 状态，验证恢复的是自己的状态。
        torch.rand(rank + 1)
        directory = Path(output) / "checkpoints"
        checkpoint = directory / "global_step_1"
        run_config = {"model": asdict(engine.model.config), "training": asdict(engine.config)}
        cursor = {"epoch": 0, "sample_offset": 3, "step": 1, "sample_visits": 3, "target_tokens": 8}
        handler = MetaMemoryCheckpointHandler(engine, loader, directory, run_config, cursor)
        rng = capture_rng(engine.device)
        handler.save_checkpoint(1)
        second = engine.step(next(iterator))
        evaluation = engine.eval_batch(episodes()[3:])
        restored = make_engine(base, output)
        restored_loader = make_data_loader(episodes(), 3, 42, shuffle=False)
        restored_cursor = {}
        restored_handler = MetaMemoryCheckpointHandler(
            restored, restored_loader, directory, run_config, restored_cursor, checkpoint
        )
        assert restored_handler.load_checkpoint() == 1
        assert restored_cursor == cursor
        assert_nested_equal(capture_rng(restored.device), rng)
        assert_nested_equal(restored.step(next(iter(restored_loader))), second)
        assert_nested_equal(
            restored.model.trainable_state_dict(), engine.model.trainable_state_dict()
        )
        assert_nested_equal(restored.optimizer.state_dict(), engine.optimizer.state_dict())
        torch.save(
            {
                "memory": engine.model.trainable_state_dict(),
                "metrics": [first, second],
                "eval": evaluation,
            },
            Path(output) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_ddp_unequal_token_counts_and_empty_tail_rank_match_single_rank(tiny_base, tmp_path):
    torch.set_num_threads(1)
    reference = make_engine(tiny_base, tmp_path)
    metrics = [reference.step(episodes()[:3]), reference.step(episodes()[3:])]
    evaluation = reference.eval_batch(episodes()[3:])
    mp.spawn(
        distributed_engine_worker,
        args=(str(tiny_base), str(tmp_path), str(tmp_path / "rendezvous")),
        nprocs=2,
        join=True,
    )
    ranks = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=False) for rank in range(2)]
    assert_nested_equal(ranks[0], ranks[1])
    for name, value in reference.model.trainable_state_dict().items():
        torch.testing.assert_close(ranks[0]["memory"][name], value, rtol=1e-5, atol=1e-7)
    for actual, expected in zip(ranks[0]["metrics"], metrics):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
    assert ranks[0]["eval"] == pytest.approx(evaluation, rel=1e-6, abs=1e-7)
