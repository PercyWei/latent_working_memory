"""以旧列表训练流程为参照，验证数据管线和 checkpoint 重构不改变训练。"""

from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch
from transformers import set_seed

from latent_working_memory.v3 import runtime
from .test_runtime import _run, make_config, make_engine, make_splits


def rng_snapshot():
    numpy_state = np.random.get_state()
    return (
        random.getstate(),
        (numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]),
        torch.get_rng_state().numpy().tobytes(),
    )


def record_execution(engine):
    """记录真实样本顺序和 task 的 RNG 消费，不额外生成随机数。"""
    batches, forwards = [], []
    original_step, original_forward = engine.step, engine.model.forward

    def step(batch, epoch=0):
        batches.append((epoch, tuple(row.document_id for row in batch)))
        return original_step(batch, epoch=epoch)

    def forward(example, epoch=0, differentiable=True, batched=False):
        rows = example if batched else [example]
        before = rng_snapshot()
        output = original_forward(
            example, epoch=epoch, differentiable=differentiable, batched=batched
        )
        forwards.append(
            (
                epoch,
                differentiable,
                tuple(row.document_id for row in rows),
                before,
                rng_snapshot(),
            )
        )
        return output

    engine.step, engine.model.forward = step, forward
    return batches, forwards


def equivalence_splits():
    splits = make_splits()
    # 三条 dev 数据形成 2+1 的尾批，验证必须按实际样本数加权。
    splits["dev"] += (
        replace(
            splits["dev"][-1],
            document_id="pretrain-dev-extra",
            dedup_cluster="pretrain-cluster-dev-extra",
            values=(9.0, -0.5),
            target=0.7,
        ),
    )
    return splits


def config_with_batch(output, microbatch, accumulation):
    config = make_config(output)
    return replace(
        config,
        training=replace(
            config.training,
            micro_batch_size_per_gpu=microbatch,
            gradient_accumulation_steps=accumulation,
        ),
    )


def original_list_development(engine, rows, epoch):
    totals, samples = {}, 0
    for start in range(0, len(rows), engine.global_batch_size):
        metrics = engine.eval_batch(rows[start : start + engine.global_batch_size], epoch=epoch)
        weight = metrics["samples"]
        for name, value in metrics.items():
            if name != "samples" and value is not None:
                totals[name] = totals.get(name, 0.0) + value * weight
        samples += weight
    return {"samples": samples, **{name: value / samples for name, value in totals.items()}}


def original_list_training(config, engine, splits):
    """使用替换前的 Python shuffle、全局列表切片与 dev 加权统计。"""
    records, visits = [], 0
    for epoch in range(config.training.epochs):
        indices = list(range(len(splits["train"])))
        random.Random(f"{config.training.seed}:v3:{epoch}").shuffle(indices)
        ordered = [splits["train"][index] for index in indices]
        for start in range(0, len(ordered), engine.global_batch_size):
            batch = ordered[start : start + engine.global_batch_size]
            metrics = engine.step(batch, epoch=epoch)
            visits += len(batch)
            step = len(records) + 1
            development = original_list_development(engine, splits["dev"], epoch)
            records.append(
                {
                    "step": step,
                    "global_step": step,
                    "stage": config.objective.stage,
                    "epoch": epoch + 1,
                    **{
                        f"train/{key}": value for key, value in metrics.items() if value is not None
                    },
                    **{f"dev/{key}": value for key, value in development.items()},
                }
            )
    return records, visits


def train(config, engine, splits, run, resume=None, stop_after_steps=None):
    stage = runtime.prepare_training(
        config, engine, splits, run, resume=resume, stop_after_steps=stop_after_steps
    )
    return runtime.train_loop(config, engine, splits, run, stage, stop_after_steps=stop_after_steps)


def recorded_metrics(output):
    return [
        {key: value for key, value in json.loads(line).items() if not key.startswith("resources/")}
        for line in (output / "metrics.jsonl").read_text().splitlines()
    ]


def assert_identical_training(actual, expected):
    torch.testing.assert_close(
        actual.model.trainable_state_dict(), expected.model.trainable_state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.optimizer.state_dict(), expected.optimizer.state_dict(), rtol=0, atol=0
    )


@pytest.mark.parametrize("microbatch,accumulation", [(1, 2), (2, 1)])
def test_verl_data_pipeline_matches_original_list_schedule_and_task_rng(
    tmp_path, microbatch, accumulation
):
    splits = equivalence_splits()
    reference_config = config_with_batch(tmp_path / "reference", microbatch, accumulation)
    set_seed(11)
    reference = make_engine(reference_config)
    reference_batches, reference_forwards = record_execution(reference)
    reference_metrics, expected_visits = original_list_training(reference_config, reference, splits)
    reference_rng = rng_snapshot()

    config = config_with_batch(tmp_path / "actual", microbatch, accumulation)
    set_seed(11)
    actual = make_engine(config)
    actual_batches, actual_forwards = record_execution(actual)
    result = train(config, actual, splits, _run(config, actual, splits))

    assert result["complete"]
    assert result["completed_steps"] == len(reference_metrics) == 6
    assert result["sample_visits"] == expected_visits == 10
    assert actual_batches == reference_batches
    assert actual_forwards == reference_forwards
    assert rng_snapshot() == reference_rng
    assert_identical_training(actual, reference)
    assert recorded_metrics(tmp_path / "actual") == reference_metrics


@pytest.mark.parametrize("microbatch,accumulation", [(1, 2), (2, 1)])
@pytest.mark.parametrize("stop_step", [2, 3])
def test_verl_checkpoint_resume_matches_full_pipeline_at_mid_epoch_and_epoch_boundary(
    tmp_path, microbatch, accumulation, stop_step
):
    splits = equivalence_splits()
    full_config = config_with_batch(tmp_path / "full", microbatch, accumulation)
    set_seed(11)
    full = make_engine(full_config)
    full_batches, full_forwards = record_execution(full)
    expected = train(full_config, full, splits, _run(full_config, full, splits))
    full_rng = rng_snapshot()

    config = config_with_batch(tmp_path / "resumed", microbatch, accumulation)
    set_seed(11)
    partial = make_engine(config)
    partial_batches, partial_forwards = record_execution(partial)
    run = _run(config, partial, splits)
    stopped = train(config, partial, splits, run, stop_after_steps=stop_step)
    assert not stopped["complete"]
    checkpoint = runtime.read_checkpoint(stopped["checkpoint"])
    assert checkpoint["cursor"] == (
        {"epoch": 0, "sample_offset": 4, "step": 2, "sample_visits": 4}
        if stop_step == 2
        else {"epoch": 1, "sample_offset": 0, "step": 3, "sample_visits": 5}
    )

    # 丢弃任务 RNG，再用 checkpoint 恢复；不能偶然依赖中断进程的残留状态。
    set_seed(999)
    restored = make_engine(config)
    restored_batches, restored_forwards = record_execution(restored)
    result = train(config, restored, splits, run, resume=stopped["checkpoint"])

    assert result["complete"]
    assert result["completed_steps"] == expected["completed_steps"] == 6
    assert result["sample_visits"] == expected["sample_visits"] == 10
    assert partial_batches + restored_batches == full_batches
    assert partial_forwards + restored_forwards == full_forwards
    assert rng_snapshot() == full_rng
    assert_identical_training(restored, full)
    assert recorded_metrics(tmp_path / "resumed") == recorded_metrics(tmp_path / "full")
