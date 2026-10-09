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
from latent_working_memory.v3 import objective
from latent_working_memory.v3.pretrain_data import PretrainExample
from latent_working_memory.v3.segmentation import even_token_chunks
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


class ACWindowCodec(torch.nn.Module):
    """每个正文段使用独立参数，检测 BPTT 边界是否丢掉末次写入的监督。"""

    def __init__(self, num_segments, memory_slots):
        super().__init__()
        self.memory_slots = memory_slots
        self.max_positions = 128
        self.memory_embeddings = torch.nn.Parameter(torch.tensor([[0.7]], dtype=torch.float64))
        self.writes = torch.nn.Parameter(torch.arange(num_segments, dtype=torch.float64) / 10 + 0.1)
        self.histories, self.outputs = [], []

    def compress_batch(self, inputs, histories, output_slots):
        results = []
        for tokens, blocks, slots in zip(inputs, histories, output_slots, strict=True):
            self.histories.append([block.requires_grad for block in blocks])
            index = (int(tokens[0]) - 3) // 3
            value = self.writes[index] + self.memory_embeddings.sum() * 0.2
            value = value + sum(block.sum() for block in blocks) * 0.3
            result = value.expand(slots, 1)
            if result.requires_grad:
                result.retain_grad()
            self.outputs.append(result)
            results.append(result)
        return results

    def answer_nll(self, memories, prompts, answers):
        return torch.stack(
            [
                (memory.sum() - answer.double().mean() / 10).square() + 0.1
                for memory, answer in zip(memories, answers, strict=True)
            ]
        )


def ac_reference_loss(model, example):
    """独立逐段展开，按所有实际预测 token 加权，并在后继监督后截断。"""
    segments = even_token_chunks(example.input_ids, model.cfg.ac_num_segments)
    count, remainder = divmod(model.codec.memory_slots, len(segments))
    blocks, losses, target_count = [], [], 0
    for step, segment in enumerate(segments):
        answer = segment[1:] + (segments[step + 1][:1] if step + 1 < len(segments) else ())
        memory = torch.cat(blocks) if blocks else model.codec.memory_embeddings[:0]
        value = model.codec.answer_nll([memory], [model.ids(segment[:1])], [model.ids(answer)])[0]
        losses.append(value * len(answer))
        target_count += len(answer)
        if model.cfg.bptt_steps is not None and step and step % model.cfg.bptt_steps == 0:
            blocks = [block.detach() for block in blocks]
        blocks.append(
            model.codec.compress_batch(
                [model.ids(segment)], [blocks], [count + (step < remainder)]
            )[0]
        )
    answer = example.target_ids + (model.tokenizer.eos_token_id,)
    prompt = model.ids(model.tokenizer.encode(model.cfg.lm_prompt))
    value = model.codec.answer_nll([torch.cat(blocks)], [prompt], [model.ids(answer)])[0]
    losses.append(value * len(answer))
    return torch.stack(losses).sum() / (target_count + len(answer))


@pytest.mark.parametrize(
    "num_segments,bptt_steps",
    [(4, None), (5, None), (4, 1), (4, 2), (4, 3), (5, 2), (4, 4), (4, 6)],
)
def test_ac_full_and_partial_bptt_windows_match_independent_gradients(
    monkeypatch, num_segments, bptt_steps
):
    class MiddleCuts:
        def randint(self, low, high):
            return (low + high) // 2

    monkeypatch.setattr(objective, "example_rng", lambda *args: MiddleCuts())
    config = ObjectiveConfig(
        method="autocompressors",
        stage="lm",
        ac_num_segments=num_segments,
        bptt_steps=bptt_steps,
    )
    model = TokenMemoryTask(ACWindowCodec(num_segments, 9), TinyTokenizer(), config)
    reference = deepcopy(model)
    example = PretrainExample(
        "ac",
        "document",
        "cluster",
        "continuation",
        tuple(range(3, 3 + 3 * num_segments)),
        (30, 31),
    )
    expected = ac_reference_loss(reference, example)
    actual = model(example)
    torch.testing.assert_close(actual["loss"], expected, rtol=1e-12, atol=1e-12)
    actual["loss"].backward()
    expected.backward()
    torch.testing.assert_close(
        model.codec.writes.grad, reference.codec.writes.grad, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        model.codec.memory_embeddings.grad,
        reference.codec.memory_embeddings.grad,
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.all(model.codec.writes.grad != 0)
    assert all(
        output.grad is not None and output.grad.abs().sum() > 0 for output in model.codec.outputs
    )
    for step, history in enumerate(model.codec.histories):
        latest_boundary = 0 if bptt_steps is None else step - step % bptt_steps
        assert history == [index >= latest_boundary for index in range(step)]
    assert actual["metrics"]["segments"] == num_segments
    assert actual["metrics"]["write_calls"] == num_segments
    assert actual["metrics"]["slots_final"] == 9
    assert actual["metrics"]["input_tokens"] == len(example.input_ids)
    assert actual["metrics"]["target_tokens"] == len(example.input_ids) + len(example.target_ids)
