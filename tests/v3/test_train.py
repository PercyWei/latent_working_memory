"""用本地小模型验证连续阶段的参数、优化器、记录和 DDP 生命周期。"""

from dataclasses import replace
from datetime import timedelta
import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from latent_working_memory.v3 import train
from latent_working_memory.v3.pretrain_data import PretrainExample
from latent_working_memory.v3.runtime import dataset_identity, read_checkpoint
from .test_model import build_model
from .test_objective import TinyTokenizer, trajectory
from .test_runtime import experiment_config


def stage_data(config, tokenizer):
    splits = {}
    for split in ("train", "dev", "test"):
        if config.objective.stage == "pretrain":
            rows = tuple(
                PretrainExample(
                    f"{split}-{index}",
                    f"pre-{split}-{index}",
                    f"pre-cluster-{split}-{index}",
                    "ae" if index % 2 == 0 else "continuation",
                    (3, 4, 5),
                    (6, 7),
                )
                for index in range(4)
            )
        else:
            rows = tuple(
                replace(
                    trajectory(n=2),
                    split=split,
                    trajectory_id=f"{split}-{index}",
                    document_id=f"qa-{split}-{index}",
                    dedup_cluster=f"qa-cluster-{split}-{index}",
                )
                for index in range(4)
            )
        splits[split] = rows
    return splits, {"source_data": {name: dataset_identity(rows) for name, rows in splits.items()}}


def stage_configs(tmp_path, method="icae_single", source=None):
    root = tmp_path / method
    stages = ("pretrain", "qa") if method.startswith("icae") else ("warmup", "policy")
    paths = []
    for index, stage in enumerate(stages):
        config = experiment_config(root, stage, method=method, init=source if index == 0 else None)
        config = replace(
            config,
            training=replace(
                config.training,
                epochs=1,
                gradient_accumulation_steps=1,
                micro_batch_size_per_gpu=index + 1,
                learning_rate=0.01 * (index + 1),
            ),
        )
        path = tmp_path / f"{method}-{stage}.json"
        path.write_text(json.dumps(config.to_dict()))
        paths.append(path)
    return paths


def install_small_training(monkeypatch):
    loads, records = [], []

    def load_codec(config, device):
        codec = build_model(memory_slots=config.memory_slots)
        loads.append(codec)
        return codec, TinyTokenizer()

    original_loop = train.train_loop

    def observe(config, engine, *args, **kwargs):
        record = {
            "stage": config.objective.stage,
            "model": engine.model,
            "module": engine.module,
            "optimizer": engine.optimizer,
            "initial": engine.model.trainable_state_dict(),
            "optimizer_initially_empty": not engine.optimizer.state,
            "gradients_initially_clear": all(p.grad is None for p in engine.parameters),
            "global_batch": engine.global_batch_size,
        }
        result = original_loop(config, engine, *args, **kwargs)
        record["final"] = engine.model.trainable_state_dict()
        record["result"] = result
        records.append(record)
        return result

    monkeypatch.setattr(train, "load_codec", load_codec)
    monkeypatch.setattr(train, "load_splits", stage_data)
    monkeypatch.setattr(train, "train_loop", observe)
    return loads, records


def arguments(paths, *options):
    return train.parse_args(
        ["--config", *map(str, paths), "--device", "cpu", "--stop-after-steps", "1", *options]
    )


@pytest.mark.parametrize(
    "method", ["icae_single", "icae_multi", "memory_change", "information_loss"]
)
def test_method_loads_once_and_inherits_weights_with_fresh_stage_optimizer(
    tmp_path, monkeypatch, method
):
    loads, records = install_small_training(monkeypatch)
    source = None
    if method in {"memory_change", "information_loss"}:
        config = experiment_config(tmp_path / "shared", "pretrain")
        path = tmp_path / "source.json"
        path.write_text(json.dumps(config.to_dict()))
        source = train.run_training(arguments([path]))["checkpoint"]
        loads.clear()
        records.clear()
    paths = stage_configs(tmp_path, method, source)
    result = train.run_training(arguments(paths))
    assert len(loads) == 1
    first, second = records
    assert first["model"] is second["model"]
    assert first["module"] is second["module"]
    assert first["optimizer"] is not second["optimizer"]
    torch.testing.assert_close(first["final"], second["initial"], rtol=0, atol=0)
    assert second["optimizer_initially_empty"] and second["gradients_initially_clear"]
    assert [row["global_batch"] for row in records] == [1, 2]
    assert second["optimizer"].param_groups[0]["lr"] == 0.02
    assert all(state["step"] == 1 for state in second["optimizer"].state.values())
    assert result["global_step"] == 2 and result["completed_steps"] == 1
    resolved = json.loads((tmp_path / method / second["stage"] / "config.json").read_text())
    assert resolved["training"]["init_checkpoint"] == first["result"]["checkpoint"]
    assert json.loads(paths[1].read_text())["training"]["init_checkpoint"] is None
    checkpoint = read_checkpoint(result["checkpoint"])
    assert checkpoint["run"]["step_offset"] == 1
    assert checkpoint["run"]["initialization"]["checkpoint"] == first["result"]["checkpoint"]
    if source is not None:
        assert checkpoint["run"]["pretraining"]["checkpoint"] == source
    manifest = json.loads((tmp_path / method / "experiment.json").read_text())
    assert list(manifest["stages"]) == [row["stage"] for row in records]


def test_second_stage_can_resume_from_its_saved_checkpoint(tmp_path, monkeypatch):
    install_small_training(monkeypatch)
    paths = stage_configs(tmp_path)
    partial = train.run_training(arguments(paths))
    resumed = train.run_training(
        arguments(
            [tmp_path / "icae_single/qa/config.json"],
            "--resume",
            partial["checkpoint"],
            "--stop-after-steps",
            "2",
        )
    )
    assert resumed["complete"]
    assert resumed["global_step"] == 3 and resumed["completed_steps"] == 2
    records = [
        json.loads(row)
        for row in (tmp_path / "icae_single/qa/metrics.jsonl").read_text().splitlines()
    ]
    assert [row["global_step"] for row in records] == [2, 3]


@pytest.mark.parametrize("change", ["method", "model", "order", "source", "identity"])
def test_invalid_sequence_fails_before_loading_any_model(tmp_path, monkeypatch, change):
    loads, _ = install_small_training(monkeypatch)
    paths = stage_configs(tmp_path)
    config = json.loads(paths[1].read_text())
    if change == "method":
        config["objective"]["method"] = "icae_multi"
    elif change == "model":
        config["model"]["memory_slots"] += 1
    elif change == "order":
        paths.reverse()
        config = json.loads(paths[1].read_text())
    elif change == "source":
        config["training"]["init_checkpoint"] = "unrelated.pt"
    else:
        config["training"]["experiment_id"] = "unrelated"
    paths[1].write_text(json.dumps(config))
    with pytest.raises(ValueError):
        train.run_training(arguments(paths))
    assert not loads


def _distributed_stages(rank, rendezvous, paths):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            loads, records = install_small_training(monkeypatch)
            result = train.run_training(arguments(paths))
            assert len(loads) == 1
            assert records[0]["module"] is records[1]["module"]
            assert isinstance(records[1]["module"], torch.nn.parallel.DistributedDataParallel)
            assert [record["global_batch"] for record in records] == [2, 4]
            assert result["global_step"] == 2
            torch.testing.assert_close(records[0]["final"], records[1]["initial"], rtol=0, atol=0)
            weights = records[1]["model"].codec.memory_embeddings.detach()
            replicas = [torch.empty_like(weights) for _ in range(2)]
            dist.all_gather(replicas, weights)
            torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_rank_training_reuses_ddp_across_stages(tmp_path):
    paths = stage_configs(tmp_path)
    mp.spawn(_distributed_stages, args=(str(tmp_path / "rendezvous"), paths), nprocs=2, join=True)
