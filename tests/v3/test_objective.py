from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from latent_working_memory.v3 import objective
from latent_working_memory.v3.config import ObjectiveConfig
from latent_working_memory.v3.data import FactQATrajectory, QA, Segment, StepUsage
from latent_working_memory.v3.objective import TokenMemoryTask, damage_action, memory_change_score
from latent_working_memory.v3.pretrain_data import PretrainExample
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


def task(method, stage, **options):
    return TokenMemoryTask(
        build_model(), TinyTokenizer(), ObjectiveConfig(method=method, stage=stage, **options)
    )


def trace_writes(model):
    calls, original = [], model.codec.compress_batch

    def traced(ids, memory_blocks=None):
        outputs = original(ids, memory_blocks)
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
        ("icae_multi", [0, 0, 0], 9),
        ("autocompressors", [0, 1, 2], 9),
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


@pytest.mark.parametrize(
    "method", ["icae_single", "icae_multi", "memory_change", "information_loss"]
)
@pytest.mark.parametrize("kind", ["ae", "continuation"])
def test_pretrain_ae_and_lm_backpropagate_from_frozen_reader(method, kind):
    model = task(method, "pretrain", segment_tokens=3 if method == "icae_multi" else 9)
    calls = trace_writes(model)
    example = PretrainExample("s", "d", "c", kind, (3, 4, 5, 6, 7, 8), (9, 10))
    if kind == "ae":
        example = replace(example, target_ids=example.input_ids)
    result = model(example)
    result["loss"].backward()
    assert model.codec.memory_embeddings.grad.abs().sum() > 0
    assert all(call[2].grad.abs().sum() > 0 for call in calls)
    assert all(len(history) == 0 for _, history, _ in calls)
    assert result["metrics"]["target_tokens"] == len(example.target_ids) + 1


def test_ac_truncates_memory_at_two_segments_but_retains_writer_learning():
    model = task("autocompressors", "lm", ac_min_segment_tokens=3, ac_max_segment_tokens=3)
    calls = trace_writes(model)
    source = tuple(range(3, 15))
    example = PretrainExample("s", "d", "c", "ae", source, source)
    reads, original = [], model.codec.answer_nll

    def traced(memories, prompts, answers):
        reads.extend(
            (prompt.tolist(), answer.tolist())
            for prompt, answer in zip(prompts, answers, strict=True)
        )
        return original(memories, prompts, answers)

    model.codec.answer_nll = traced
    result = model(example)
    result["loss"].backward()
    assert result["metrics"]["segments"] == 4
    assert reads == [([3], [4, 5, 6]), ([6], [7, 8]), ([9], [10, 11, 12]), ([12], [13, 14])]
    assert result["metrics"]["target_tokens"] == 10  # exclude one leading token per BPTT group
    assert [len(history) for _, history, _ in calls] == [0, 1, 2]
    assert not any(block.requires_grad for block in calls[2][1])
    assert calls[0][2].grad is not None and calls[0][2].grad.abs().sum() > 0
    assert calls[1][2].grad is None
    assert calls[2][2].grad is not None and calls[2][2].grad.abs().sum() > 0
    assert model.codec.memory_embeddings.grad.abs().sum() > 0


def test_eval_forward_never_creates_gradient_graph():
    model = task("memory_change", "policy")
    result = model(trajectory(), differentiable=False)
    assert not result["loss"].requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())


def test_ac_rejects_sample_without_any_trainable_cross_segment_target():
    model = task("autocompressors", "lm", ac_min_segment_tokens=3, ac_max_segment_tokens=3)
    example = PretrainExample("s", "d", "c", "ae", (3, 4, 5, 6), (3, 4, 5, 6))
    with pytest.raises(ValueError, match="no trainable next-token target"):
        model(example)


@pytest.mark.parametrize(
    "method,stage,options",
    [
        ("icae_single", "pretrain", {}),
        ("icae_multi", "pretrain", {"segment_tokens": 3}),
        ("memory_change", "pretrain", {}),
        ("information_loss", "pretrain", {}),
        ("icae_single", "qa", {}),
        ("icae_multi", "qa", {}),
        ("memory_change", "warmup", {"append_probability": 0.5}),
        ("information_loss", "warmup", {"append_probability": 0.5}),
        ("memory_change", "policy", {"threshold_i": 0.0}),
        ("memory_change", "policy", {"threshold_i": 1e8}),
        ("information_loss", "policy", {"threshold_g": 1e-8, "eta": 0.0}),
        ("information_loss", "policy", {"threshold_d": 1e8, "threshold_g": 1e8}),
        ("autocompressors", "lm", {"ac_min_segment_tokens": 3, "ac_max_segment_tokens": 4}),
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

    def traced(ids, histories=None):
        writes.append([len(history) for history in histories])
        return original(ids, histories)

    model.codec.compress_batch = traced
    assert actions(records) == expected
    assert len(writes) == 4
    assert all(len(histories) == len(records) for histories in writes)
    assert any(set(histories) == {0, 1} for histories in writes)
    assert actions(records[::-1]) == expected[::-1]


def test_empty_microbatch_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        task("memory_change", "pretrain")([], batched=True)


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
