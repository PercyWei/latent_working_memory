from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
import json

import pytest
import torch

from latent_working_memory.v1.backbone import ReadTokens
from latent_working_memory.v1.data import (
    Episode,
    EpisodeIndex,
    Read,
    Reference,
    Source,
    write_episodes,
)
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.pretrain.evaluation import evaluate_pretraining
from latent_working_memory.v1.pretrain.evaluation_batching import (
    EvaluationBatching,
    ReadJob,
    GenerationJob,
    read_batches,
    generation_batches,
)
from evaluation_reference import evaluate_pretraining_reference
from test_reader_projection import make_backbone


def assert_results_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_results_equal(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_results_equal(x, y)
    elif isinstance(a, float):
        assert b == pytest.approx(a, rel=1e-5, abs=1e-5)
    else:
        assert a == b


def panel_index(path, tokenizer):
    episodes = []
    for i in range(24):
        n = (2, 4, 8, 16, 24, 32)[(i // 2) % 6]
        ids = tuple(4 + (j + i) % 6 for j in range(n))
        task = "ae" if i % 2 == 0 else "continuation"
        target = ids if task == "ae" else tuple(4 + (j + i + 3) % 6 for j in range(max(2, n // 2)))
        source = Source(
            str(i),
            str(i),
            0,
            n,
            {
                "boundary_method": "pysbd_conservative",
                "boundary_variant": "semantic",
                "dedup_cluster": str(i),
            },
        )
        read = Read(
            str(i),
            task,
            n,
            "Reconstruct the text:" if task == "ae" else "Continue the text:",
            (Reference(tokenizer.decode(target, skip_special_tokens=True), ()),),
        )
        episodes.append(Episode(str(i), ids, (n,), (source,), (read,)))
    write_episodes(episodes, path)
    return EpisodeIndex(path)


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
@pytest.mark.parametrize("generation", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_full_evaluation_matches_original_schedule(
    tmp_path, tokenizer, tiny_config, monkeypatch, architecture, generation, bf16
):
    torch.manual_seed(913)
    backbone = make_backbone(architecture)
    writer = JointMemoryWriter(8, 1, 2, 16, 32)
    config = replace(tiny_config, eval_examples=24, eval_generation_examples=6)
    index = panel_index(tmp_path / "data.jsonl", tokenizer)
    original_read = backbone.read_batch
    original_generate = backbone.greedy_students
    reads = []
    generations = []
    call_sizes = []

    def memory_key(memory):
        return tuple(memory.shape), memory.detach().float().numpy().tobytes()

    def traced_read(memories, tasks, text_contexts=None, use_reader_lora=True):
        contexts = text_contexts if text_contexts is not None else [()] * len(tasks)
        reads.extend(
            (memory_key(m), t, c, use_reader_lora)
            for m, t, c in zip(memories, tasks, contexts, strict=True)
        )
        call_sizes.append(len(tasks))
        return original_read(memories, tasks, text_contexts, use_reader_lora)

    def traced_generate(memories, prompts, limits, use_reader_lora=True):
        output = original_generate(memories, prompts, limits, use_reader_lora)
        generations.extend(
            (memory_key(m), p, limit, use_reader_lora, ids)
            for m, p, limit, ids in zip(memories, prompts, limits, output, strict=True)
        )
        return output

    monkeypatch.setattr(backbone, "read_batch", traced_read)
    monkeypatch.setattr(backbone, "greedy_students", traced_generate)
    results = []
    rows = []
    prefixes = []
    observed = []
    step = 0 if generation else 1
    for name, evaluate in [("old", evaluate_pretraining_reference), ("new", evaluate_pretraining)]:
        reads.clear()
        generations.clear()
        call_sizes.clear()
        backbone.train()
        writer.eval()
        rng = torch.get_rng_state().clone()
        with torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext():
            results.append(
                evaluate(
                    config,
                    tokenizer,
                    backbone,
                    writer,
                    index,
                    tmp_path / name,
                    step,
                    123,
                    "dev",
                    (1,),
                )
            )
        assert backbone.training and not writer.training
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        rows.append(
            [
                json.loads(line)
                for line in (tmp_path / name / f"dev-step-{step:06d}.jsonl")
                .read_text()
                .splitlines()
            ]
        )
        p = tmp_path / name / f"dev-step-{step:06d}-prefix.jsonl"
        prefixes.append(
            [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
        )
        observed.append((Counter(reads), Counter(generations), len(call_sizes), max(call_sizes)))
    assert_results_equal(results[0], results[1])
    assert_results_equal(rows[0], rows[1])
    assert_results_equal(prefixes[0], prefixes[1])
    assert observed[0][0] == observed[1][0]
    assert observed[0][1] == observed[1][1]  # Exact generated token sequences, including EOS.
    assert observed[1][2] < observed[0][2]
    assert observed[1][3] > observed[0][3]
    assert bool(observed[1][1]) == generation


def test_read_batches_respect_adapter_and_both_token_budgets():
    jobs = [
        ReadJob([{"i": i}], torch.zeros(k, 8), ReadTokens((11,), (4,) * t + (2,)), (5,) * c, lora)
        for i, (k, t, c, lora) in enumerate(
            [
                (2, 4, 0, True),
                (4, 9, 0, True),
                (0, 3, 3, False),
                (0, 4, 4, True),
                (1, 2, 0, True),
                (3, 6, 0, False),
            ]
        )
    ]
    limits = EvaluationBatching(read_batch_size=3, read_context_tokens=40, read_target_tokens=16)
    batches = list(read_batches(jobs, limits))
    assert Counter(id(j) for b in batches for j in b) == Counter(map(id, jobs))
    for batch in batches:
        assert len({j.use_lora for j in batch}) == 1
        assert len(batch) <= 3
        assert len(batch) * max(j.input_length for j in batch) <= 40
        assert sum(j.target_length for j in batch) <= 16
    with pytest.raises(ValueError, match="batch token budget"):
        list(read_batches(jobs, replace(limits, read_target_tokens=1)))


def test_two_dimensional_generation_grouping_reduces_padding_and_decode_budget():
    # All requests use the same adapter; A/B and C/D have almost equal P+G.
    jobs = [
        GenerationJob(
            [{"i": i}],
            torch.empty(0, 8),
            (4,) * (p - 1),
            ReadTokens((11,), (4,) * (g - 1) + (2,)),
            True,
        )
        for i, (p, g) in enumerate([(136, 1025), (584, 577), (144, 1089), (616, 609)])
    ]
    old = sorted(jobs, key=lambda j: j.input_length + j.target_length)
    old_batches = [old[i : i + 2] for i in range(0, len(old), 2)]
    limits = EvaluationBatching(generation_batch_size=2, generation_context_tokens=4096)
    batches = list(generation_batches(jobs, limits))
    assert Counter(id(j) for b in batches for j in b) == Counter(map(id, jobs))

    def cost(batches):
        return (
            sum(len(b) * max(j.input_length for j in b) for b in batches),
            sum(len(b) * max(j.target_length for j in b) for b in batches),
        )

    assert cost(batches)[0] < cost(old_batches)[0]
    assert cost(batches)[1] < cost(old_batches)[1]
    for batch in batches:
        assert len({(j.target_length - 1).bit_length() for j in batch}) == 1
        assert (
            len(batch) * (max(j.input_length for j in batch) + max(j.target_length for j in batch))
            <= 4096
        )
    disabled = replace(jobs[0], use_lora=False)
    for batch in generation_batches(jobs + [disabled], limits):
        assert len({j.use_lora for j in batch}) == 1


def test_evaluation_restores_modes_on_batch_budget_failure(
    tmp_path, tokenizer, tiny_config, components
):
    backbone, writer = components
    index = panel_index(tmp_path / "data.jsonl", tokenizer)
    backbone.train()
    writer.eval()
    with pytest.raises(ValueError, match="batch token budget"):
        evaluate_pretraining(
            tiny_config,
            tokenizer,
            backbone,
            writer,
            index,
            tmp_path / "out",
            0,
            0,
            batching=EvaluationBatching(read_target_tokens=1),
        )
    assert backbone.training and not writer.training
    assert not (tmp_path / "out").exists()
