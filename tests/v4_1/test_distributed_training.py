"""CPU Gloo 验证全局样本均值、空 rank 与按 rank 恢复，无需加载基座。"""

from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from latent_working_memory.v4_1.config import ExperimentConfig, TrainingConfig
from latent_working_memory.v4_1.training import checkpoint
from latent_working_memory.v4_1.training.engine import (
    SampleCursor, TrainingEngine, evaluate_dev, training_steps,
)
from latent_working_memory.v4_1.training.train import _restore_metrics, _training_output_dir


class ToyMemory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Parameter(torch.tensor([[0.4, -0.3], [0.2, 0.5]]))
        self.memory = torch.nn.Parameter(torch.tensor([0.1, -0.2]))
        self.decoder_unused = torch.nn.Parameter(torch.ones(2, 2))

    def forward(self, episode, teacher, lambda_am, generator, differentiable):
        features = torch.tensor([episode + 1, episode % 3 - 1], dtype=torch.float32) / 8
        qa = ((self.decoder @ features).sum() - (episode % 4) / 3).square()
        # 偶数样本才使用 memory，分片后某 rank 的该参数没有梯度。
        am = self.memory.square().sum() if episode % 2 == 0 else qa.new_zeros(())
        return {
            "loss": qa + lambda_am * am, "qa_loss": qa, "outer_am": am,
            "inner_before": am + 1, "inner_after": am,
            "target_tokens": episode % 4 + 1, "write_events": episode % 3,
        }

    def forward_batch(self, episodes, teachers, lambda_am, generator, differentiable):
        # 仅作为分布式样本均值的数值 oracle；真实批量计算由 Qwen3 对照覆盖。
        return [self.forward(episode, teacher, lambda_am, generator, differentiable)
                for episode, teacher in zip(episodes, teachers)]

    def trainable_state_dict(self):
        return {name: value.detach().clone() for name, value in self.named_parameters()}

    def load_trainable_state_dict(self, state):
        self.load_state_dict(state)


class EmptyTeacher:
    def episode(self, episode, generator=None):
        return nullcontext()


def training_config():
    return TrainingConfig(
        max_steps=4, warmup_steps=0, learning_rate=0.03, weight_decay=0.1,
        gradient_clip=0.7, gradient_accumulation_steps=2, tracking="disabled",
    )


def snapshot(model, metrics):
    return {
        "metrics": metrics,
        "parameters": model.trainable_state_dict(),
        "gradients": {name: None if value.grad is None else value.grad.clone()
                      for name, value in model.named_parameters()},
    }


def init_group(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )


def update_worker(rank, rendezvous, expected, expected_dev, micro_batch_size):
    init_group(rank, rendezvous)
    try:
        model, teacher, generator = ToyMemory(), EmptyTeacher(), torch.Generator().manual_seed(1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(rank)
        engine = TrainingEngine(model, replace(
            training_config(), micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=2 // micro_batch_size,
        ))
        for episodes, reference in zip(([0, 1, 2, 3], [4, 5, 6], [7]), expected):
            metrics = engine.train_batch(episodes[rank::2], teacher, generator)
            torch.testing.assert_close(snapshot(model, metrics), reference, rtol=2e-6, atol=2e-7)
            assert model.decoder_unused.grad is None
        for episodes, reference in zip(([0, 2, 5], [7]), expected_dev):
            metrics = evaluate_dev(model, episodes[rank::2], teacher, 1.0, generator)
            torch.testing.assert_close(metrics, reference, rtol=2e-6, atol=2e-7)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("micro_batch_size", [1, 2])
def test_two_ranks_match_single_global_batch(tmp_path, micro_batch_size):
    model, teacher, generator = ToyMemory(), EmptyTeacher(), torch.Generator().manual_seed(1)
    # 单卡与双卡均使用有效全局 batch 4；尾批分别含 3 和 1 个实际样本。
    engine = TrainingEngine(model, replace(training_config(), gradient_accumulation_steps=4))
    expected = [snapshot(model, engine.train_batch(episodes, teacher, generator))
                for episodes in ([0, 1, 2, 3], [4, 5, 6], [7])]
    expected_dev = [evaluate_dev(model, episodes, teacher, 1.0, generator)
                    for episodes in ([0, 2, 5], [7])]
    mp.spawn(update_worker, args=(str(tmp_path / "update-rendezvous"), expected, expected_dev,
                                  micro_batch_size), nprocs=2, join=True)


def draw_rng(query_generator):
    return (random.random(), float(np.random.random()), torch.rand(3),
            torch.rand(3, generator=query_generator))


def checkpoint_worker(rank, rendezvous, output_dir):
    init_group(rank, rendezvous)
    try:
        random.seed(31 + rank)
        np.random.seed(41 + rank)
        torch.manual_seed(51 + rank)
        generator = torch.Generator().manual_seed(61 + rank)
        model, teacher = ToyMemory(), EmptyTeacher()
        config = ExperimentConfig(training=replace(
            training_config(), output_root=str(Path(output_dir).parent.parent), output_dir=str(output_dir),
        ))
        engine, cursor = TrainingEngine(model, config.training), SampleCursor(7, 71)
        engine.train_batch(cursor.next_batch(4)[rank::2], teacher, generator)
        original_save, writes = torch.save, []

        def record_save(state, destination, *args, **kwargs):
            # gather_object 也会借助 torch.save 序列化 BytesIO，仅计入文件写入。
            if isinstance(destination, (str, Path)):
                writes.append(destination)
            return original_save(state, destination, *args, **kwargs)

        with patch.object(checkpoint.torch, "save", side_effect=record_save):
            path = checkpoint.save_checkpoint(
                output_dir, 1, config, model, engine, cursor, generator, {}, torch.device("cpu"),
            )
            assert len(writes) == int(rank == 0)
        assert path == Path(output_dir) / "checkpoints/global_step_1/state.pt"
        state = checkpoint.read_checkpoint(path)
        assert Path(state["config"]["training"]["output_dir"]) == Path(output_dir)
        assert _training_output_dir(config, state) == Path(output_dir).resolve()
        assert len(state["rng"]) == len(state["query_rng"]) == 2
        assert not torch.equal(state["query_rng"][0], state["query_rng"][1])
        expected_rng = draw_rng(generator)
        next_episodes = cursor.next_batch(4)
        assert len(next_episodes) == 3 and cursor.epoch_end and cursor.epoch == 1
        expected = snapshot(model, engine.train_batch(next_episodes[rank::2], teacher, generator))

        restored = ToyMemory()
        restored_engine = TrainingEngine(restored, config.training)
        restored_cursor, restored_generator = SampleCursor(7, 999), torch.Generator().manual_seed(999)
        step = checkpoint.restore_training(
            state, restored, restored_engine, restored_cursor, restored_generator, torch.device("cpu"),
        )
        assert step == 1
        torch.testing.assert_close(draw_rng(restored_generator), expected_rng, rtol=0, atol=0)
        assert restored_cursor.next_batch(4) == next_episodes
        actual = restored_engine.train_batch(next_episodes[rank::2], teacher, restored_generator)
        torch.testing.assert_close(snapshot(restored, actual), expected, rtol=0, atol=0)

        # 主 rank 写盘失败须传达给所有 rank，不能使其余进程停在保存同步点。
        def fail_write(state, destination, *args, **kwargs):
            if isinstance(destination, (str, Path)):
                raise OSError("disk unavailable")
            return original_save(state, destination, *args, **kwargs)

        with patch.object(checkpoint.torch, "save", side_effect=fail_write):
            with pytest.raises(OSError if rank == 0 else RuntimeError, match="disk unavailable"):
                checkpoint.save_checkpoint(
                    output_dir, 2, config, restored, restored_engine, restored_cursor,
                    restored_generator, {}, torch.device("cpu"),
                )
    finally:
        dist.destroy_process_group()


def test_checkpoint_restores_each_rank_rng_and_rejects_changed_world_size(tmp_path):
    output_dir = tmp_path / "run" / "train"
    mp.spawn(checkpoint_worker, args=(str(tmp_path / "checkpoint-rendezvous"), str(output_dir)),
             nprocs=2, join=True)
    assert list(output_dir.glob("checkpoints/*/state.pt")) == [
        output_dir / "checkpoints/global_step_1/state.pt",
    ]
    state = checkpoint.read_checkpoint(output_dir / "checkpoints/global_step_1")
    model = ToyMemory()
    before = model.trainable_state_dict()
    with pytest.raises(ValueError, match="world_size=1"):
        checkpoint.restore_training(
            state, model, TrainingEngine(model, training_config()), SampleCursor(7, 71),
            torch.Generator(), torch.device("cpu"),
        )
    torch.testing.assert_close(model.trainable_state_dict(), before, rtol=0, atol=0)


def test_training_budget_cursor_and_committed_metrics(tmp_path):
    experiment = ExperimentConfig(training=replace(
        training_config(), output_root=str(tmp_path), run_id="01_20261029",
    ))
    directory = _training_output_dir(experiment, None)
    assert directory == tmp_path / "am-recent_m256-d512_p1024-r256_inner3_01_20261029/train"
    other = replace(experiment, training=replace(experiment.training, run_id="02_20261029"))
    assert _training_output_dir(other, None).parent != directory.parent
    assert other.training.teacher_cache_dir == experiment.training.teacher_cache_dir
    assert Path(experiment.training.teacher_cache_dir) == tmp_path / "teacher/p1024-r256"

    config = replace(training_config(), epochs=3, max_steps=None)
    assert training_steps(config, 7, 4) == 6
    assert training_steps(replace(config, max_steps=5), 7, 4) == 5
    assert training_steps(training_config(), 7, 4) == 4

    cursor = SampleCursor(7, 71)
    assert len(cursor.next_batch(4)) == 4 and not cursor.epoch_end
    boundary = cursor.state_dict()
    # 旧格式在最后一批后仍保存 offset == size，新读取须保持下轮排列不变。
    boundary["offset"] = 7
    assert len(cursor.next_batch(4)) == 3 and cursor.epoch_end
    assert cursor.epoch == 1 and cursor.offset == 0
    restored = SampleCursor(7, 999)
    restored.load_state_dict(boundary, 2, 4)
    torch.testing.assert_close(restored.state_dict(), cursor.state_dict(), rtol=0, atol=0)
    current = cursor.state_dict()
    restored.load_state_dict(current, 2, 4)
    assert restored.next_batch(4) == cursor.next_batch(4)
    for state, step in ((current, 1), ({**current, "offset": 1}, 3),
                        ({**current, "order": [0] * 7}, 2)):
        with pytest.raises(ValueError, match="checkpoint"):
            restored.load_state_dict(state, step, 4)

    metrics = tmp_path / "metrics.jsonl"
    rows = [json.dumps({"step": step, "train/loss": 1.0}) + "\n" for step in (1, 2, 3)]
    metrics.write_text("".join(rows) + '{"step":4', encoding="utf-8")
    _restore_metrics(metrics, 2)
    assert metrics.read_text(encoding="utf-8") == "".join(rows[:2])
    for content in (rows[1], rows[0] * 2, rows[0] + '{"step":2', ""):
        metrics.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="training metrics"):
            _restore_metrics(metrics, 2)
