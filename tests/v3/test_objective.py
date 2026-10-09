from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from latent_working_memory.v3 import objective
from latent_working_memory.v3.config import ObjectiveConfig
from latent_working_memory.v3.data import FactQATrajectory, QA, Segment, StepUsage
from latent_working_memory.v3.objective import TokenMemoryTask, damage_action, memory_change_score
from latent_working_memory.v3.pretrain_data import PretrainExample
from latent_working_memory.v3.segmentation import (
    ac_plan,
    even_token_chunks,
    example_rng,
    icae_multi_plan,
)
from .test_model import build_model


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(character) % 35 for character in text[:5]]


def trajectory(split="train", n=3):
    segments = tuple(
        Segment(f"seg{i}", (3 * i, 3 * i + 3), (3 + i, 4 + i, 5 + i)) for i in range(n)
    )
    qas = {}
    for i in range(n):
        for role in ("gate", "train" if split == "train" else "evaluation"):
            qid = f"{role}{i}"
            qas[qid] = QA(
                qid,
                f"seg{i}",
                qid,
                qid,
                qid,
                qid,
                role,
                (3 * i, 3 * i + 2),
                (3 * i, 3 * i + 1),
                (7,),
                tuple(range(8, 9 + i)),
            )
    usages = tuple(
        StepUsage(
            f"seg{i}",
            (f"train{i}",),
            tuple(f"train{j}" for j in range(i)),
            (f"gate{i - 1}",) if i else (),
        )
        for i in range(n)
    )
    return FactQATrajectory(
        "t",
        "d",
        "cluster",
        split,
        "abc" * n,
        {},
        (0, 3 * n),
        segments,
        qas,
        usages,
        tuple(range(3, 3 + 3 * n)),
    )


def task(method, stage, memory_slots=3, **options):
    if method == "icae_multi":
        options = {"icae_min_segments": 3, "icae_max_segments": 3, **options}
    elif method == "autocompressors":
        options = {"ac_num_segments": 3, "bptt_steps": 2, **options}
    config = ObjectiveConfig(method=method, stage=stage, **{"append_slots": 3, **options})
    return TokenMemoryTask(
        build_model(
            memory_slots=memory_slots,
            write_slots=(memory_slots + config.icae_min_segments - 1) // config.icae_min_segments
            if method == "icae_multi"
            else (memory_slots + config.ac_num_segments - 1) // config.ac_num_segments
            if method == "autocompressors"
            else None,
        ),
        TinyTokenizer(),
        config,
    )


def trace_writes(model):
    calls, original = [], model.codec.compress_batch

    def traced(ids, memory_blocks=None, output_slots=None):
        outputs = original(ids, memory_blocks, output_slots)
        histories = memory_blocks if memory_blocks is not None else [[] for _ in ids]
        for tokens, history, output in zip(ids, histories, outputs, strict=True):
            if output.requires_grad:
                output.retain_grad()
            calls.append((tokens.detach().clone(), list(history), output))
        return outputs

    model.codec.compress_batch = traced
    return calls


def test_change_score_uses_slot_rms_without_mutating_memory():
    old = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    other = torch.tensor([[4.0, 3.0], [1.0, 1.0]])
    copy = old.clone()
    eps = 1e-6
    normalized = old / torch.sqrt(torch.mean(old**2, dim=1, keepdim=True) + eps)
    expected = torch.norm(
        other / torch.sqrt(torch.mean(other**2, dim=1, keepdim=True) + eps) - normalized
    ) / (torch.norm(normalized) + eps)
    torch.testing.assert_close(memory_change_score(old, other, eps), expected)
    torch.testing.assert_close(old, copy)
    assert torch.isfinite(memory_change_score(torch.zeros_like(old), other, eps))


@pytest.mark.parametrize(
    "values,expected",
    [
        ((1.0, 1.3, 1.1), True),  # gain alone
        ((1.0, 1.3, 1.25), True),  # damage + small positive gain
        ((1.0, 1.3, 1.4), False),  # damage without append benefit
        ((1.0, 1.0, 1.0), False),
        ((1.0, 1.1, 1.0), False),  # strict > at exact representable boundary below
    ],
)
def test_damage_rule(values, expected):
    threshold = 0.125 if values == (1.0, 1.1, 1.0) else 0.1
    assert damage_action(*values, threshold, threshold, 0.01) is expected


@pytest.mark.parametrize(
    "method,expected_histories,expected_slots",
    [
        ("icae_single", [0], 3),
        ("icae_multi", [0, 0, 0], 3),
        ("autocompressors", [0, 1, 2], 3),
    ],
)
def test_baselines_write_scopes(method, expected_histories, expected_slots):
    model = task(method, "lm" if method == "autocompressors" else "qa")
    calls = trace_writes(model)
    blocks, events = model.build_memory(trajectory())
    assert [len(call[1]) for call in calls] == expected_histories
    assert sum(map(len, blocks)) == expected_slots
    if method == "icae_single":
        assert calls[0][0].tolist() == list(trajectory().full_input_ids)
        assert len(events) == 1
    else:
        assert [event["action"] for event in events] == ["initial", "append", "append"]


@pytest.mark.parametrize("append", [False, True])
def test_memory_change_selects_only_local_graph(append):
    model = task("memory_change", "policy", threshold_i=0.0 if append else 1e8)
    calls = trace_writes(model)
    output = model(trajectory())
    output["loss"].backward()
    assert [len(call[1]) for call in calls] == ([0, 1, 0, 1, 0] if append else [0, 1, 1])
    assert output["metrics"]["slots_final"] == (9 if append else 3)
    assert output["metrics"]["gate_qa_reads"] == 0
    for index, (_, _, memory) in enumerate(calls):
        if append and index in (1, 3):
            assert memory.grad is None  # discarded rewrite has no auxiliary loss
        else:
            assert memory.grad is not None and memory.grad.abs().sum() > 0


@pytest.mark.parametrize("append", [False, True])
def test_damage_gates_are_no_grad_and_excluded_from_task_loss(append):
    model = task("information_loss", "policy")
    calls = trace_writes(model)
    original = model._qa_losses_batch
    gate_calls = []

    def controlled(requests, events=None, timing_key="read_seconds"):
        blocks, record, ids = requests[0]
        if ids[0].startswith("gate"):
            assert not torch.is_grad_enabled()
            gate_calls.append(tuple(ids))
            triple = (1.0, 2.0, 1.0) if append else (1.0, 1.0, 2.0)
            return [torch.tensor([triple[(len(gate_calls) - 1) % 3]])]
        assert all(not qid.startswith("gate") for qid in ids)
        return original(requests, events, timing_key)

    model._qa_losses_batch = controlled
    result = model(trajectory())
    result["loss"].backward()
    assert len(gate_calls) == 6
    assert result["metrics"]["gate_qa_reads"] == 6
    assert result["metrics"]["task_qa_reads"] == 6
    assert result["metrics"]["slots_final"] == (9 if append else 3)
    for index in (1, 3) if append else (2, 4):
        assert calls[index][2].grad is None
    assert model.codec.memory_embeddings.grad.abs().sum() > 0


def test_dynamic_qa_is_actual_question_mean_then_step_mean():
    model = task("memory_change", "warmup", append_probability=0.0)
    record = trajectory()
    per_id = {f"train{i}": float(i + 1) for i in range(3)}
    model._qa_losses_batch = lambda requests, events=None, timing_key="read_seconds": [
        torch.stack([blocks[-1].sum() * 0 + per_id[qid] for qid in ids])
        for blocks, record, ids in requests
    ]
    output = model(record)
    # step means 1, (2+1)/2, (3+1+2)/3; fixed half old/new would differ at step 3.
    torch.testing.assert_close(output["loss"], torch.tensor(1.5))
    assert output["metrics"]["qa_new_nll"] == 2
    assert output["metrics"]["qa_old_nll"] == pytest.approx(4 / 3)


def test_warmup_actions_are_matched_between_methods_and_eval_uses_policy():
    record = trajectory(n=5)
    left = task("memory_change", "warmup", threshold_i=1e8)
    right = task("information_loss", "warmup")
    a = left.build_memory(record, epoch=3, force_policy=False)[1]
    b = right.build_memory(record, epoch=3, force_policy=False)[1]
    assert [e["action"] for e in a] == [e["action"] for e in b]
    assert all(e["write_calls"] == 1 and e["gate_qa_reads"] == 0 for e in a + b)
    policy = left.build_memory(record, force_policy=True)[1]
    assert all(e["action"] == "overwrite" for e in policy[1:])


@pytest.mark.parametrize("method", ["memory_change", "information_loss"])
@pytest.mark.parametrize("stage", ["warmup", "policy"])
def test_dynamic_initial_64_append_8_and_overwrite_preserves_last_block_size(
    monkeypatch, method, stage
):
    model = TokenMemoryTask(
        build_model(memory_slots=64, max_positions=256),
        TinyTokenizer(),
        ObjectiveConfig(method=method, stage=stage, append_slots=8, threshold_i=0.5),
    )
    choices = iter([False, True, False, True])
    compared_sizes = []
    if stage == "warmup":

        class FixedActions:
            def random(self):
                return 0.0 if next(choices) else 1.0

        monkeypatch.setattr(objective, "example_rng", lambda *args: FixedActions())
    elif method == "memory_change":

        def controlled_score(old, rewritten, epsilon):
            assert old.shape == rewritten.shape
            compared_sizes.append(len(old))
            return torch.tensor(float(next(choices)))

        monkeypatch.setattr(objective, "memory_change_score", controlled_score)
    else:
        monkeypatch.setattr(objective, "damage_action", lambda *args: next(choices))
    with torch.no_grad():
        states = list(model._states(trajectory(n=5), force_policy=stage == "policy"))
    assert [event["action"] for _, event in states] == [
        "initial",
        "overwrite",
        "append",
        "overwrite",
        "append",
    ]
    assert [event["slots"] for _, event in states] == [64, 64, 72, 72, 80]
    assert [list(map(len, blocks)) for blocks, _ in states] == [
        [64],
        [64],
        [64, 8],
        [64, 8],
        [64, 8, 8],
    ]
    # 追加保留所有旧块；覆盖只替换末块。
    assert states[2][0][0] is states[1][0][0]
    assert states[3][0][0] is states[2][0][0]
    assert states[3][0][-1] is not states[2][0][-1]
    assert states[4][0][0] is states[3][0][0]
    assert states[4][0][1] is states[3][0][1]
    if compared_sizes:
        assert compared_sizes == [64, 64, 8, 8]


@pytest.mark.parametrize(
    "method", ["icae_single", "icae_multi", "memory_change", "information_loss"]
)
@pytest.mark.parametrize("kind", ["ae", "continuation"])
def test_pretrain_ae_and_lm_backpropagate_from_frozen_reader(method, kind):
    model = task(method, "pretrain", icae_min_segments=2, icae_max_segments=2)
    calls = trace_writes(model)
    example = PretrainExample("s", "d", "c", kind, (3, 4, 5, 6, 7, 8), (9, 10))
    if kind == "ae":
        example = replace(example, target_ids=example.input_ids)
    result = model(example)
    result["loss"].backward()
    assert model.codec.memory_embeddings.grad.abs().sum() > 0
    assert all(call[2].grad.abs().sum() > 0 for call in calls)
    assert all(len(history) == 0 for _, history, _ in calls)
    assert len(calls) == (2 if method == "icae_multi" else 1)
    if method != "icae_multi":
        assert calls[0][0].tolist() == list(example.input_ids)
    assert result["metrics"]["slots_final"] == model.codec.memory_slots
    assert result["metrics"]["target_tokens"] == len(example.target_ids) + 1


@pytest.mark.parametrize("total_slots,num_segments", [(4, 2), (5, 4), (6, 2), (8, 3)])
def test_icae_multi_pretrain_uses_uniform_chunks_with_fixed_total_budget(total_slots, num_segments):
    write_slots = (total_slots + num_segments - 1) // num_segments
    model = TokenMemoryTask(
        build_model(memory_slots=total_slots, write_slots=write_slots),
        TinyTokenizer(),
        ObjectiveConfig(
            method="icae_multi", icae_min_segments=num_segments, icae_max_segments=num_segments
        ),
    )
    writes = trace_writes(model)
    reads, original = [], model.codec.answer_nll

    def traced(memories, prompts, answers):
        reads.append(memories)
        return original(memories, prompts, answers)

    model.codec.answer_nll = traced
    tokens = tuple(range(3, 17))
    example = PretrainExample("ae", "d", "c", "ae", tokens, tokens)
    result = model(example)
    result["loss"].backward()
    expected = even_token_chunks(tokens, num_segments)
    assert [tuple(tokens.tolist()) for tokens, _, _ in writes] == expected
    assert len(writes) == num_segments
    assert max(map(len, expected)) - min(map(len, expected)) <= 1
    assert tuple(token for chunk in expected for token in chunk) == tokens
    assert all(not history for _, history, _ in writes)
    expected_slots, remainder = divmod(total_slots, num_segments)
    assert [len(memory) for _, _, memory in writes] == [
        expected_slots + (index < remainder) for index in range(num_segments)
    ]
    assert all(memory.grad is not None and memory.grad.abs().sum() > 0 for _, _, memory in writes)
    assert len(reads) == 1 and len(reads[0]) == 1
    torch.testing.assert_close(reads[0][0], torch.cat([memory for _, _, memory in writes]))
    assert result["metrics"]["slots_final"] == total_slots
    assert model.codec.memory_embeddings.shape[0] == write_slots
    assert torch.all(model.codec.memory_embeddings.grad.abs().sum(dim=1) > 0)


@pytest.mark.parametrize("num_segments", [2, 4])
def test_icae_multi_qa_rechunks_full_text_and_keeps_original_question_scopes(num_segments):
    model = task(
        "icae_multi",
        "qa",
        memory_slots=8,
        icae_min_segments=num_segments,
        icae_max_segments=num_segments,
    )
    record = trajectory()
    segment_tokens = ((3, 4), tuple(range(5, 12)), (12, 13, 14, 15))
    record = replace(
        record,
        segments=tuple(
            replace(segment, input_ids=tokens)
            for segment, tokens in zip(record.segments, segment_tokens, strict=True)
        ),
        full_input_ids=tuple(token for tokens in segment_tokens for token in tokens),
    )
    writes = trace_writes(model)
    requests, original = [], model._qa_losses_batch

    def traced(values, events=None, timing_key="read_seconds"):
        requests.extend(values)
        return original(values, events, timing_key)

    model._qa_losses_batch = traced
    result = model(record)
    result["loss"].backward()
    assert [tuple(tokens.tolist()) for tokens, _, _ in writes] == even_token_chunks(
        record.full_input_ids, num_segments
    )
    assert all(not history for _, history, _ in writes)
    assert result["metrics"]["slots_final"] == model.codec.memory_slots
    assert len(requests) == 1
    assert requests[0][2] == ("train2", "train0", "train1")
    assert result["metrics"]["qa_new_count"] == 1
    assert result["metrics"]["qa_old_count"] == 2
    assert result["metrics"]["write_calls"] == num_segments


@pytest.mark.parametrize("stage", ["pretrain", "qa"])
def test_icae_multi_rejects_text_shorter_than_fixed_chunk_count(stage):
    model = task("icae_multi", stage, memory_slots=4, icae_min_segments=4, icae_max_segments=4)
    example = (
        PretrainExample("short", "d", "c", "ae", (3, 4, 5), (3, 4, 5))
        if stage == "pretrain"
        else trajectory(n=1)
    )
    with pytest.raises(ValueError, match="as many tokens as sampled ICAE chunks"):
        model(example)


def test_icae_multi_sampled_count_and_capacity_are_reproducible_per_example():
    tokens = tuple(range(3, 23))
    plans = [icae_multi_plan(tokens, 512, 3, 6, 42, f"s{i}") for i in range(32)]
    assert {len(chunks) for chunks, _ in plans} == {3, 4, 5, 6}
    for index, (chunks, slots) in enumerate(plans):
        assert (chunks, slots) == icae_multi_plan(tokens, 512, 3, 6, 42, f"s{index}")
        assert sum(slots) == 512
        assert max(slots) - min(slots) <= 1
        assert max(map(len, chunks)) - min(map(len, chunks)) <= 1
        assert tuple(token for chunk in chunks for token in chunk) == tokens
    assert plans[0][1] == [86, 86, 85, 85, 85, 85]
    assert plans[2][1] == [171, 171, 170]


def test_icae_multi_keeps_sampled_chunks_across_epochs():
    model = task(
        "icae_multi", "qa", memory_slots=8, icae_min_segments=3, icae_max_segments=6, seed=42
    )
    record = replace(trajectory(n=3), trajectory_id="s0")
    with torch.no_grad():
        first, first_events = model.build_memory(record, epoch=0)
        later, later_events = model.build_memory(record, epoch=10)
    assert len(first) == len(later) == 6
    assert sum(map(len, first)) == 8
    torch.testing.assert_close(first, later, rtol=0, atol=0)
    assert [event["step"] for event in first_events] == list(range(6))
    assert all(event["segment_id"] is None for event in first_events + later_events)


def test_ac_truncates_memory_at_two_segments_but_retains_writer_learning(monkeypatch):
    class MiddleCuts:
        def randint(self, low, high):
            return (low + high) // 2

    monkeypatch.setattr(objective, "example_rng", lambda *args: MiddleCuts())
    model = task("autocompressors", "lm", memory_slots=9, ac_num_segments=4)
    calls = trace_writes(model)
    source = tuple(range(3, 15))
    example = PretrainExample("s", "d", "c", "continuation", source, (18, 19))
    reads, original = [], model.codec.answer_nll

    def traced(memories, prompts, answers):
        reads.extend(
            (len(memory), prompt.tolist(), answer.tolist())
            for memory, prompt, answer in zip(memories, prompts, answers, strict=True)
        )
        return original(memories, prompts, answers)

    model.codec.answer_nll = traced
    result = model(example)
    result["loss"].backward()
    assert result["metrics"]["segments"] == 4
    assert reads[:4] == [
        (0, [3], [4, 5, 6]),
        (3, [6], [7, 8, 9]),
        (5, [9], [10, 11, 12]),
        (7, [12], [13, 14]),
    ]
    assert reads[4] == (9, model.tokenizer.encode(model.cfg.lm_prompt), [18, 19, 2])
    assert result["metrics"]["input_tokens"] == 12
    assert result["metrics"]["target_tokens"] == 14  # 11正文 targets + 3续文 targets
    assert result["metrics"]["slots_final"] == 9
    assert result["metrics"]["write_calls"] == 4
    assert [len(history) for _, history, _ in calls] == [0, 1, 2, 3]
    assert [len(memory) for _, _, memory in calls] == [3, 2, 2, 2]
    assert torch.cat([ids for ids, _, _ in calls]).tolist() == list(source)
    assert not any(block.requires_grad for block in calls[2][1])
    assert calls[0][2].grad is not None and calls[0][2].grad.abs().sum() > 0
    assert calls[1][2].grad is not None and calls[1][2].grad.abs().sum() > 0
    assert calls[2][2].grad is not None and calls[2][2].grad.abs().sum() > 0
    assert calls[3][2].grad is not None and calls[3][2].grad.abs().sum() > 0
    assert model.codec.memory_embeddings.grad.abs().sum() > 0


def test_eval_forward_never_creates_gradient_graph():
    model = task("memory_change", "policy")
    result = model(trajectory(), differentiable=False)
    assert not result["loss"].requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())


def test_ac_rejects_source_too_short_for_fixed_segment_count():
    model = task("autocompressors", "lm", ac_num_segments=2)
    example = PretrainExample("s", "d", "c", "continuation", (3, 4, 5), (6, 7))
    with pytest.raises(ValueError, match="at least two tokens per segment"):
        model(example)


@pytest.mark.parametrize("total_tokens", [4, 5, 6])
def test_ac_short_sources_retain_writer_and_memory_gradients(total_tokens):
    model = task("autocompressors", "lm", ac_num_segments=2)
    calls = trace_writes(model)
    example = PretrainExample(
        "short", "document", "cluster", "continuation", tuple(range(3, 3 + total_tokens)), (20, 21)
    )
    result = model(example)
    result["loss"].backward()
    assert result["metrics"]["segments"] == 2
    assert result["metrics"]["input_tokens"] == total_tokens
    assert len(calls) == 2
    assert all(call[2].grad.abs().sum() > 0 for call in calls)
    assert model.codec.memory_embeddings.grad.abs().sum() > 0
    adapter_gradients = [
        parameter.grad
        for name, parameter in model.codec.language_model.named_parameters()
        if "lora_" in name
    ]
    assert all(
        gradient is not None and torch.isfinite(gradient).all() for gradient in adapter_gradients
    )
    assert sum(gradient.abs().sum() for gradient in adapter_gradients) > 0


def test_ac_random_segments_match_paper_pairs_and_reproduce_per_example_epoch():
    source = tuple(range(6144))

    def plan(epoch, identifier="sample"):
        return ac_plan(source, 512, 4, 2, 8192, example_rng(23, epoch, identifier))

    chunks, slots = plan(0)
    assert slots == [128] * 4
    assert len(chunks) == 4
    assert tuple(token for chunk in chunks for token in chunk) == source
    assert all(1024 <= len(chunk) <= 2048 for chunk in chunks)
    assert [sum(len(chunk) for chunk in chunks[start : start + 2]) for start in (0, 2)] == [
        3072,
        3072,
    ]
    assert plan(0) == plan(0)
    assert plan(0) != plan(1)
    assert plan(0) != plan(0, "other")


def test_ac_random_segments_keep_partial_window_and_respect_cumulative_memory_budget():
    source = tuple(range(81))
    chunks, slots = ac_plan(source, 13, 5, 2, 64, example_rng(1, 3, "tail"))
    assert slots == [3, 3, 3, 2, 2]
    assert [sum(map(len, chunks[start : start + 2])) for start in (0, 2, 4)] == [33, 32, 16]
    assert len(chunks[-1]) == 16
    assert tuple(token for chunk in chunks for token in chunk) == source
    for seed in range(10):
        chunks, slots = ac_plan(tuple(range(20)), 8, 4, 2, 13, example_rng(seed, 0, "budget"))
        assert all(len(chunk) + sum(slots[: index + 1]) <= 13 for index, chunk in enumerate(chunks))
    with pytest.raises(ValueError, match="exceed model window"):
        ac_plan(tuple(range(8)), 8, 4, 2, 9, example_rng(1, 0, "too-long"))


@pytest.mark.parametrize("num_segments", [4, 5])
def test_ac_full_bptt_random_segments_match_one_window(num_segments):
    source = tuple(range(81))

    def plan(bptt_steps):
        return ac_plan(source, 13, num_segments, bptt_steps, 64, example_rng(1, 3, "full-bptt"))

    assert plan(None) == plan(num_segments) == plan(num_segments + 2)


@pytest.mark.parametrize("bptt_steps", [4, 6])
def test_ac_full_bptt_matches_large_window_losses_and_gradients(bptt_steps):
    model = task("autocompressors", "lm", memory_slots=9, ac_num_segments=4, bptt_steps=None)
    reference = deepcopy(model)
    reference.cfg = replace(reference.cfg, bptt_steps=bptt_steps)
    calls = trace_writes(model)
    example = PretrainExample(
        "full-bptt", "document", "cluster", "continuation", tuple(range(3, 24)), (26, 27)
    )
    actual = model(example, epoch=3)
    expected = reference(example, epoch=3)
    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=0, atol=0)
    actual["loss"].backward()
    expected["loss"].backward()
    for parameter, other in zip(model.parameters(), reference.parameters(), strict=True):
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
    assert all(block.requires_grad for _, history, _ in calls for block in history)
    assert all(output.grad is not None and output.grad.abs().sum() > 0 for _, _, output in calls)


@pytest.mark.parametrize("bptt_steps", [None, 1, 2, 6])
def test_ac_evaluation_evenly_splits_full_body_independent_of_factqa_segments(bptt_steps):
    model = task("autocompressors", "lm", memory_slots=8, ac_num_segments=4, bptt_steps=bptt_steps)
    calls = trace_writes(model)
    record = trajectory(n=5)
    blocks, events = model.build_memory(record, epoch=7)
    assert [ids.tolist() for ids, _, _ in calls] == [
        list(chunk) for chunk in even_token_chunks(record.full_input_ids, 4)
    ]
    assert [len(history) for _, history, _ in calls] == [0, 1, 2, 3]
    assert sum(map(len, blocks)) == 8
    assert [event["slots"] for event in events] == [2, 4, 6, 8]
    assert [event["step"] for event in events] == [0, 1, 2, 3]
    assert all(event["segment_id"] is None for event in events)


@pytest.mark.parametrize(
    "method,stage,options",
    [
        ("icae_single", "pretrain", {}),
        (
            "icae_multi",
            "pretrain",
            {"memory_slots": 8, "icae_min_segments": 3, "icae_max_segments": 6},
        ),
        ("memory_change", "pretrain", {}),
        ("information_loss", "pretrain", {}),
        ("icae_single", "qa", {}),
        (
            "icae_multi",
            "qa",
            {"memory_slots": 8, "icae_min_segments": 3, "icae_max_segments": 6},
        ),
        ("memory_change", "warmup", {"append_probability": 0.5}),
        ("information_loss", "warmup", {"append_probability": 0.5}),
        ("memory_change", "policy", {"threshold_i": 0.0}),
        ("memory_change", "policy", {"threshold_i": 1e8}),
        ("information_loss", "policy", {"threshold_g": 1e-8, "eta": 0.0}),
        ("information_loss", "policy", {"threshold_d": 1e8, "threshold_g": 1e8}),
        ("autocompressors", "lm", {"ac_num_segments": 3}),
    ],
)
def test_batch_matches_individual_losses_gradients_and_metrics(method, stage, options):
    model = task(method, stage, qa_batch_size=2, **options)
    reference = deepcopy(model)
    if stage in {"pretrain", "lm"}:
        text = tuple(range(3, 14))
        examples = [
            PretrainExample("ae", "d", "c", "ae", text, text),
            PretrainExample("lm", "d2", "c2", "continuation", text[:8], (17, 18, 19)),
        ]
        if stage == "lm":
            examples[0] = replace(examples[0], task="continuation", target_ids=(20, 21))
    else:
        examples = [trajectory(n=2), replace(trajectory(n=4), trajectory_id="other")]
    expected = [reference(row, epoch=3) for row in examples]
    expected_loss = torch.stack([result["loss"] for result in expected]).mean()
    actual = model(examples, epoch=3, batched=True)
    torch.testing.assert_close(actual["loss"], expected_loss, rtol=1e-5, atol=1e-6)
    expected_loss.backward()
    actual["loss"].backward()
    for name, parameter in model.named_parameters():
        other = dict(reference.named_parameters())[name]
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-4, atol=1e-6)
    for name, value in actual["metrics"].items():
        if not name.endswith("_seconds"):
            assert value == pytest.approx(
                sum(result["metrics"][name] for result in expected) / len(expected),
                rel=1e-5,
                abs=1e-6,
            )


def test_batch_qa_preserves_question_then_update_then_trajectory_weights():
    model = task("memory_change", "warmup", append_probability=0.0)
    records = [trajectory(n=2), replace(trajectory(n=4), trajectory_id="other")]

    def controlled(requests, events=None, timing_key="read_seconds"):
        return [
            torch.stack(
                [blocks[-1].sum() * 0 + float(qid.removeprefix("train")) + 1 for qid in ids]
            )
            for blocks, _, ids in requests
        ]

    model._qa_losses_batch = controlled
    result = model(records, batched=True)
    # 两条轨迹分别为 mean(1, 1.5)=1.25 和 mean(1, 1.5, 2, 2.5)=1.75。
    torch.testing.assert_close(result["loss"], torch.tensor(1.5))
    assert result["metrics"]["task_qa_reads"] == 6.5
    assert result["metrics"]["qa_new_count"] == 3
    result["loss"].backward()


def test_batch_shared_timing_is_charged_once_per_model_call(monkeypatch):
    # 每次 measured 上下文固定耗时 1 秒，避开机器负载对测试的影响。
    ticks = iter(range(1000))
    monkeypatch.setattr(objective, "perf_counter", lambda: float(next(ticks)))
    model = task("memory_change", "warmup", append_probability=0.0, qa_batch_size=2)
    records = [trajectory(n=2), replace(trajectory(n=3), trajectory_id="other")]
    result = model(records, batched=True)
    # writer共3次；reader在更新点0/1/2分别1/1/2次。
    assert result["metrics"]["write_seconds"] * len(records) == 3
    assert result["metrics"]["read_seconds"] * len(records) == 4


def test_batch_warmup_rng_and_actions_are_independent_of_batch_order():
    model = task("memory_change", "warmup", append_probability=0.5)
    records = [replace(trajectory(n=4), trajectory_id=f"row-{i}") for i in range(4)]

    def actions(rows):
        result = [[] for _ in rows]
        with torch.no_grad():
            for states in model._states_batch(rows, epoch=7):
                for i, _, event in states:
                    result[i].append(event["action"])
        return result

    expected = [actions([row])[0] for row in records]
    assert len({tuple(values) for values in expected}) > 1
    writes, original = [], model.codec.compress_batch

    def traced(ids, histories=None, output_slots=None):
        writes.append([len(history) for history in histories])
        return original(ids, histories, output_slots)

    model.codec.compress_batch = traced
    assert actions(records) == expected
    assert len(writes) == 4
    assert all(len(histories) == len(records) for histories in writes)
    assert any(set(histories) == {0, 1} for histories in writes)
    assert actions(records[::-1]) == expected[::-1]


def test_empty_microbatch_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        task("memory_change", "pretrain")([], batched=True)


def test_missing_sample_metric_is_not_replaced_with_zero(monkeypatch):
    model = task("memory_change", "pretrain")
    text = (3, 4, 5)
    examples = [PretrainExample(str(i), "d", "c", "ae", text, text) for i in range(2)]
    objective = model._pretrain_objective

    def omit_metric(rows):
        losses, metrics = objective(rows)
        del metrics[1]["input_tokens"]
        return losses, metrics

    monkeypatch.setattr(model, "_pretrain_objective", omit_metric)
    with pytest.raises(KeyError, match="input_tokens"):
        model(examples, batched=True)


@pytest.mark.parametrize("method", ["memory_change", "information_loss"])
@pytest.mark.parametrize("stage", ["warmup", "policy"])
def test_one_bptt_window_matches_full_loss_gradients_and_detaches_memory(method, stage):
    model = task(method, stage, append_probability=0.0, threshold_i=1e8)
    reference = deepcopy(model)
    rows = [trajectory(n=2), replace(trajectory(n=5), trajectory_id="long")]
    expected = reference(rows, epoch=3, batched=True)
    actual = model(rows, epoch=3, batched=True, window_steps=5)
    torch.testing.assert_close(actual["loss"], expected["loss"])
    expected["loss"].backward()
    actual["loss"].backward()
    for name, parameter in model.named_parameters():
        other = dict(reference.named_parameters())[name]
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad)
    for blocks in actual["qa_state"].blocks:
        assert all(not block.requires_grad and block.grad_fn is None for block in blocks)
    assert actual["qa_state"].step == 5


@pytest.mark.parametrize("append_probability", [0.0, 1.0])
def test_two_step_bptt_preserves_values_actions_and_full_trajectory_tail_weights(
    append_probability,
):
    model = task("memory_change", "warmup", append_probability=append_probability)
    reference = deepcopy(model)
    rows = [trajectory(n=2), replace(trajectory(n=5), trajectory_id="long")]
    expected = reference(rows, epoch=4, batched=True)
    expected["loss"].backward()
    state, total_loss = None, 0.0
    for start in (0, 2, 4):
        actual = model(rows, epoch=4, batched=True, window_steps=2, qa_state=state)
        total_loss += float(actual["loss"].detach())
        actual["loss"].backward()
        state = actual["qa_state"]
        assert all(
            not block.requires_grad and block.grad_fn is None
            for blocks in state.blocks
            for block in blocks
        )
        assert state.step == min(start + 2, 5)
    assert total_loss == pytest.approx(float(expected["loss"].detach()))
    for name, value in actual["metrics"].items():
        if not name.endswith("_seconds"):
            assert value == pytest.approx(expected["metrics"][name])
    wanted_action = "append" if append_probability else "overwrite"
    assert [[event["action"] for event in events] for events in state.events] == [
        ["initial", wanted_action],
        ["initial", *([wanted_action] * 4)],
    ]
    assert not torch.allclose(
        model.codec.memory_embeddings.grad, reference.codec.memory_embeddings.grad
    )


def test_batch_damage_gates_discard_each_unselected_candidate_without_gradient():
    model = task("information_loss", "policy", qa_batch_size=1)
    records = [trajectory(n=2), replace(trajectory(n=3), trajectory_id="overwrite")]
    calls = trace_writes(model)
    original = model._qa_losses_batch
    gate_calls = []

    def controlled(requests, events=None, timing_key="read_seconds"):
        if requests[0][2][0].startswith("gate"):
            assert not torch.is_grad_enabled()
            variant = len(gate_calls) % 3
            gate_calls.append([record.trajectory_id for _, record, _ in requests])
            return [
                torch.tensor(
                    [
                        (1.0, 1.0, 2.0)[variant]
                        if record.trajectory_id == "overwrite"
                        else (1.0, 2.0, 1.0)[variant]
                    ]
                )
                for _, record, _ in requests
            ]
        assert all(not qid.startswith("gate") for _, _, ids in requests for qid in ids)
        return original(requests, events, timing_key)

    model._qa_losses_batch = controlled
    result = model(records, batched=True)
    result["loss"].backward()
    assert gate_calls == [["t", "overwrite"]] * 3 + [["overwrite"]] * 3
    assert result["metrics"]["appends"] == 0.5
    assert result["metrics"]["overwrites"] == 1
    assert result["metrics"]["slots_final"] == 4.5
    assert result["metrics"]["task_qa_reads"] == result["metrics"]["gate_qa_reads"] == 4.5
    assert len(calls) == 8
    for index, (_, _, memory) in enumerate(calls):
        if index in (2, 5, 7):
            assert memory.grad is None
        else:
            assert memory.grad is not None and memory.grad.abs().sum() > 0


def test_qa_reader_batches_active_trajectories_with_per_trajectory_chunk_limit():
    model = task("information_loss", "policy", qa_batch_size=2)
    records = [trajectory(n=3), replace(trajectory(n=4), trajectory_id="other")]
    original = model.codec.answer_nll
    batch_sizes = []

    def traced(memories, prompts, answers):
        batch_sizes.append((torch.is_grad_enabled(), len(memories)))
        return original(memories, prompts, answers)

    model.codec.answer_nll = traced
    model(records, batched=True)
    assert max(size for _, size in batch_sizes) == 4
    assert all(size <= len(records) * model.cfg.qa_batch_size for _, size in batch_sizes)
    assert any(not differentiable and size == 2 for differentiable, size in batch_sizes)
