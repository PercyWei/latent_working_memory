from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from latent_working_memory.v3.config import ObjectiveConfig, TrainingConfig
from latent_working_memory.v3.engine import TokenMemoryEngine
from latent_working_memory.v3.objective import TokenMemoryTask, example_rng
from .test_objective import TinyTokenizer, trajectory


class WindowCodec(torch.nn.Module):
    """不同写入轮次使用不同参数，覆盖最终窗口未活跃参数的梯度同步。"""

    def __init__(self):
        super().__init__()
        self.memory_embeddings = torch.nn.Parameter(torch.tensor([[0.7]], dtype=torch.float64))
        self.second = torch.nn.Parameter(torch.tensor([[-0.2]], dtype=torch.float64))
        self.histories = []

    def compress_batch(self, inputs, histories=None, output_slots=None):
        if histories is None:
            histories = [[] for _ in inputs]
        if output_slots is None:
            output_slots = [1 for _ in inputs]
        self.histories.extend([block.requires_grad for block in blocks] for blocks in histories)
        results = []
        for tokens, blocks, slots in zip(inputs, histories, output_slots, strict=True):
            parameter = self.memory_embeddings if int(tokens[0]) % 2 else self.second
            value = parameter + float(tokens[0]) / 10
            if blocks:
                value = value + sum(block.sum() for block in blocks) * 0.3
            results.append(value.expand(slots, 1))
        return results

    def answer_nll(self, memories, prompts, answers):
        return torch.stack(
            [
                (memory.sum() - answer.double().mean() / 10).square() + 0.1
                for memory, answer in zip(memories, answers, strict=True)
            ]
        )


def make_model(stage="warmup", bptt_steps=2):
    return TokenMemoryTask(
        WindowCodec(),
        TinyTokenizer(),
        ObjectiveConfig(
            method="memory_change",
            stage=stage,
            append_slots=1,
            append_probability=0.5,
            threshold_i=1e8,
            bptt_steps=bptt_steps,
        ),
    )


def training_config(microbatch):
    return TrainingConfig(
        dataset_dir="unused",
        output_dir="unused",
        micro_batch_size_per_gpu=microbatch,
        gradient_accumulation_steps=4,
        learning_rate=0.05,
        weight_decay=0.0,
        gradient_clip=100.0,
    )


def records():
    return [
        replace(trajectory(n=n), trajectory_id=f"row-{i}-{n}")
        for i, n in enumerate((2, 3, 5, 2, 4))
    ]


def first_parameter_only_record():
    record = trajectory(n=2)
    return replace(
        record,
        trajectory_id="first-parameter-only",
        segments=tuple(replace(segment, input_ids=(3, 4, 5)) for segment in record.segments),
    )


def reference_step(model, optimizer, batch, epoch):
    """逐轨迹独立展开并截断，手动按完整轨迹及全局样本数加权。"""
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for record in batch:
        blocks, losses = [], []
        rng = example_rng(model.cfg.seed, epoch, record.trajectory_id)
        for step, segment in enumerate(record.segments):
            append = (
                step > 0
                and model.cfg.stage == "warmup"
                and rng.random() < model.cfg.append_probability
            )
            candidate = model.codec.compress_batch(
                [model.ids(segment.input_ids)],
                [[] if step == 0 or append else [blocks[-1]]],
                [1],
            )[0]
            blocks = blocks + [candidate] if step == 0 or append else blocks[:-1] + [candidate]
            usage = record.usage[step]
            losses.append(
                model.qa_losses(blocks, record, usage.new_qa_ids + usage.old_qa_ids).mean()
            )
            if (step + 1) % model.cfg.bptt_steps == 0 or step + 1 == len(record.segments):
                loss = torch.stack(losses).sum() / len(record.segments) / len(batch)
                total_loss += float(loss.detach())
                loss.backward()
                blocks = [block.detach() for block in blocks]
                losses = []
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    optimizer.step()
    return total_loss, float(norm)


@pytest.mark.parametrize("stage", ["warmup", "policy"])
@pytest.mark.parametrize("microbatch", [1, 2, 8])
def test_window_engine_matches_independent_truncation_and_updates_once(stage, microbatch):
    model = make_model(stage)
    reference = deepcopy(model)
    engine = TokenMemoryEngine(model, training_config(microbatch), "cpu")
    engine.initialize()
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.05, weight_decay=0.0)
    batch = records()
    expected_loss, expected_norm = reference_step(reference, optimizer, batch, epoch=3)
    actual = engine.step(batch, epoch=3)
    assert actual["loss"] == pytest.approx(expected_loss)
    assert actual["grad_norm"] == pytest.approx(expected_norm)
    assert actual["samples"] == len(batch)
    for observed, wanted in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(observed, wanted, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(observed.grad, wanted.grad, rtol=1e-12, atol=1e-12)
    assert all(state["step"].item() == 1 for state in engine.optimizer.state.values())


def test_window_engine_backwards_before_next_forward_and_detaches_all_kept_blocks():
    model = make_model()
    model.cfg = replace(model.cfg, append_probability=1.0)
    engine = TokenMemoryEngine(model, training_config(8), "cpu")
    engine.initialize()
    boundaries = []
    original = model.forward

    def traced(*args, **kwargs):
        state = kwargs["qa_state"]
        if state is not None:
            assert all(
                not block.requires_grad and block.grad_fn is None
                for blocks in state.blocks
                for block in blocks
            )
            assert model.codec.memory_embeddings.grad is not None
            assert model.codec.second.grad is not None
            boundaries.append(state.step)
        return original(*args, **kwargs)

    model.forward = traced
    engine.step([trajectory(n=5)])
    assert boundaries == [2, 4]


def _distributed_window_worker(rank, rendezvous, world_size, microbatch):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        for stage in ("warmup", "policy"):
            model = make_model(stage)
            reference = deepcopy(model)
            engine = TokenMemoryEngine(model, training_config(microbatch), "cpu")
            engine.initialize()
            optimizer = torch.optim.AdamW(reference.parameters(), lr=0.05, weight_decay=0.0)
            rows = records()
            for epoch, batch in enumerate(
                (rows + rows[:3], rows, [first_parameter_only_record()], rows[2:3])
            ):
                expected_loss, expected_norm = reference_step(reference, optimizer, batch, epoch)
                actual = engine.step(batch, epoch=epoch)
                assert actual["loss"] == pytest.approx(expected_loss)
                assert actual["grad_norm"] == pytest.approx(expected_norm)
                assert actual["samples"] == len(batch)
                for observed, wanted in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    torch.testing.assert_close(observed, wanted, rtol=1e-12, atol=1e-12)
                    if wanted.grad is None:
                        assert observed.grad is None
                    else:
                        torch.testing.assert_close(
                            observed.grad, wanted.grad, rtol=1e-12, atol=1e-12
                        )
                    assert engine.optimizer.state[observed]["step"].item() == (
                        optimizer.state[wanted]["step"].item()
                    )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size,microbatch", [(2, 1), (2, 2), (4, 2)])
def test_distributed_window_schedule_handles_lengths_branches_and_empty_ranks(
    tmp_path,
    world_size,
    microbatch,
):
    mp.spawn(
        _distributed_window_worker,
        args=(str(tmp_path / "rendezvous"), world_size, microbatch),
        nprocs=world_size,
        join=True,
    )
