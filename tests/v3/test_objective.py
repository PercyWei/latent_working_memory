from dataclasses import replace

import pytest
import torch

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
    calls, original = [], model.codec.compress

    def traced(ids, memory_blocks=None):
        output = original(ids, memory_blocks)
        if output.requires_grad:
            output.retain_grad()
        calls.append((ids.detach().clone(), list(memory_blocks or []), output))
        return output

    model.codec.compress = traced
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
    original = model.qa_losses
    gate_calls = []

    def controlled(blocks, record, ids):
        if ids[0].startswith("gate"):
            assert not torch.is_grad_enabled()
            gate_calls.append(tuple(ids))
            triple = (1.0, 2.0, 1.0) if append else (1.0, 1.0, 2.0)
            return torch.tensor([triple[(len(gate_calls) - 1) % 3]])
        assert all(not qid.startswith("gate") for qid in ids)
        return original(blocks, record, ids)

    model.qa_losses = controlled
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
    model.qa_losses = lambda blocks, record, ids: torch.stack(
        [blocks[-1].sum() * 0 + per_id[qid] for qid in ids]
    )
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
    result = model(example)
    result["loss"].backward()
    assert result["metrics"]["segments"] == 4
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
