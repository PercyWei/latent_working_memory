from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import json
import sys

import pytest
import torch
import torch.distributed as dist
from transformers import LlamaConfig, LlamaForCausalLM

from latent_working_memory.data_preparation.squad import prepare_squad
from latent_working_memory.data_preparation.dynamic import prepare_dynamic
from latent_working_memory.v1 import dynamic_reporting
from latent_working_memory.v1.backbone import load_backbone
from latent_working_memory.v1.checkpoint import (
    capture_rng_state,
    load_model_checkpoint,
    save_model_checkpoint,
)
from latent_working_memory.v1.data import Episode, Read, Reference, Source
from latent_working_memory.v1.dynamic import run_dynamic, main
from latent_working_memory.v1.dynamic_config import DynamicConfig
from latent_working_memory.v1.dynamic_training import DynamicTrainer, read_schedule
from latent_working_memory.v1.dynamic_evaluation import answer_scores, evaluate_qa, aggregate_qa
from latent_working_memory.v1.dynamic_reporting import (
    qa_media,
    log_qa,
    main as publish_qa,
    training_curves,
    publish_training_history,
)
from latent_working_memory.v1.dynamic_data import (
    DynamicTextSampler,
    allocate_counts,
    text_bounds,
    write_boundaries,
)
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.v1.squad import QA_PROMPT, SquadDataset
from latent_working_memory.v1.training import trainable_model_state


def example(tokenizer, name="doc", paragraphs=6):
    ids, ends, sources, reads = [], [], [], []
    for i in range(paragraphs):
        context = f"First p{i}."
        start = len(ids)
        ids.extend(tokenizer.encode(context, add_special_tokens=False))
        end = len(ids)
        ends.append(end)
        sources.append(Source(f"{name}:p{i}", name, start, end, {"context": context}))
        reads.append(
            Read(
                f"{name}:q{i}",
                "qa",
                end,
                QA_PROMPT.format(question="What?"),
                (Reference("First", ((start, end),)),),
            )
        )
    return Episode(name, tuple(ids), tuple(ends), tuple(sources), tuple(reads))


def small_recipe(**kwargs):
    return DynamicConfig(
        capacities=(4, 8),
        global_batch_size=1,
        generation_tokens=2,
        qa_activation_checkpointing=False,
        **kwargs,
    )


@pytest.mark.parametrize("unit,span", [("tokens", 0), ("tokens", 1), ("updates", 1)])
def test_initial_compression_receives_gradient_and_base_is_frozen(
    components, tiny_config, tokenizer, unit, span
):
    backbone, writer = components
    e = example(tokenizer)
    recipe = small_recipe(new_count=0, history_count=1, bptt_unit=unit, bptt_span=span)
    features, starts, sizes = [], [], []
    original = backbone.text_features

    def trace(units, offsets):
        result = original(units, offsets)
        result[0].retain_grad()
        features.append(result[0])
        starts.extend(offsets)
        sizes.extend(map(len, units))
        return result

    backbone.text_features = trace
    base = {
        n: p.detach().clone()
        for n, p in backbone.language_model.named_parameters()
        if not p.requires_grad
    }
    trainer = DynamicTrainer(backbone, writer, tiny_config, recipe, torch.device("cpu"))
    result = trainer.step([e], tokenizer, [42], 4)
    assert sizes == [9, 3, 3, 3]  # Exactly 1.5K=6 is buffered until the next paragraph.
    assert starts == [0, 9, 12, 15]
    assert result["writes"] == 4 and result["updates"] == 3
    assert result["reads"] == 3
    assert features[0].grad.abs().sum() > 0
    assert all(f.grad is not None for f in features)
    assert backbone.memory_projection.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in backbone.language_model.named_parameters()
        if "lora_" in n
    )
    for n, p in backbone.language_model.named_parameters():
        if n in base:
            assert torch.equal(base[n], p) and p.grad is None


@pytest.mark.parametrize(
    "unit,span,expected",
    [
        ("tokens", 0, [(18, 4)]),
        ("tokens", 1, [(12, 2), (3, 1), (3, 1)]),
        ("tokens", 13, [(15, 3), (3, 1)]),
        ("updates", 2, [(12, 2), (6, 2)]),
    ],
)
def test_tbptt_boundaries_and_one_optimizer_step(
    components, tiny_config, tokenizer, monkeypatch, unit, span, expected
):
    b, w = components
    trainer = DynamicTrainer(
        b, w, tiny_config, small_recipe(bptt_unit=unit, bptt_span=span), torch.device("cpu")
    )
    calls = []
    original = trainer.optimizer.step

    def step():
        calls.append(1)
        return original()

    monkeypatch.setattr(trainer.optimizer, "step", step)
    result = trainer.step([example(tokenizer)], tokenizer, [42], 4)
    assert [
        (s["tokens"], s["updates"]) for s in result["sample_metrics"][0]["bptt_segments"]
    ] == expected
    assert result["truncations"] == len(expected) - 1 and calls == [1]


def test_schedule_initial_history_and_context_budgets(tokenizer, tiny_config):
    e = example(tokenizer)
    recipe = small_recipe(new_count=0, history_count=10, max_visits=1)
    schedule = read_schedule(e, tokenizer, recipe, tiny_config, 42, 4)
    assert not schedule[9]
    assert {r.read_id for r, _ in schedule[12]} == {"doc:q0", "doc:q1", "doc:q2"}
    assert all(r.prefix_end == end for end, jobs in schedule.items() for r, _ in jobs)
    with pytest.raises(ValueError, match="write context budget"):
        read_schedule(
            e,
            tokenizer,
            recipe,
            replace(tiny_config, write_context_tokens=9, max_input_tokens=8),
            42,
            4,
        )
    with pytest.raises(ValueError, match="QA read exceeds"):
        read_schedule(e, tokenizer, recipe, replace(tiny_config, read_context_tokens=8), 42, 4)
    with pytest.raises(ValueError, match="subsequent update"):
        write_boundaries(example(tokenizer, paragraphs=3), 4)


@pytest.mark.parametrize("span", [0, 1])
def test_reader_recomputation_matches_gradients(components, tiny_config, tokenizer, span):
    b, w = components
    other_b, other_w = deepcopy(components)
    recipe = small_recipe(bptt_span=span)
    a = DynamicTrainer(b, w, tiny_config, recipe, torch.device("cpu"))
    c = DynamicTrainer(
        other_b,
        other_w,
        tiny_config,
        replace(recipe, qa_activation_checkpointing=True),
        torch.device("cpu"),
    )
    episodes = [example(tokenizer)]
    assert a.step(episodes, tokenizer, [42], 4) == c.step(episodes, tokenizer, [42], 4)
    for x, y in zip(a.parameters, c.parameters, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize("unit,span", [("tokens", 0), ("tokens", 1), ("updates", 2)])
def test_equal_sample_weight_and_memory_reset(
    components, tiny_config, tokenizer, monkeypatch, unit, span
):
    b, w = components
    recipe = small_recipe(bptt_unit=unit, bptt_span=span, gradient_clip=1e9, max_visits=5)
    episodes = [example(tokenizer, "short", 4), example(tokenizer, "long", 7)]
    grads, losses = [], []
    for e in episodes:
        x, y = deepcopy(components)
        trainer = DynamicTrainer(x, y, tiny_config, recipe, torch.device("cpu"))
        losses.append(trainer.step([e], tokenizer, [42], 4)["loss"])
        grads.append(
            [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in trainer.parameters]
        )
    trainer = DynamicTrainer(
        b, w, tiny_config, replace(recipe, global_batch_size=2), torch.device("cpu")
    )
    calls = []
    original = w.initialize_state

    def initialize(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(w, "initialize_state", initialize)
    result = trainer.step(episodes, tokenizer, [42, 42], 4)
    assert calls == [1, 1] and result["samples"] == 2
    assert result["loss"] == pytest.approx(sum(losses) / 2)
    for p, g, h in zip(trainer.parameters, *grads, strict=True):
        actual = torch.zeros_like(p) if p.grad is None else p.grad
        torch.testing.assert_close(actual, (g + h) / 2, atol=1e-6, rtol=1e-4)
    with pytest.raises(ValueError, match="global_batch_size samples"):
        trainer.step(episodes[:1], tokenizer, [42], 4)


def test_paired_evaluation_uses_initialization_and_original_evidence(
    components, tiny_config, tokenizer, monkeypatch
):
    b, w = components
    recipe = small_recipe(new_count=0, history_count=1, max_visits=1)
    episodes = [example(tokenizer, "a", 4), example(tokenizer, "b", 4)]
    saved = {k: v.clone() for k, v in w.state_dict().items()}
    calls = []
    original = b.read_batch

    def read(memories, tokens, **kwargs):
        calls.append(
            (len(memories[0]), tokenizer.decode(tokens[0].prompt_ids), kwargs["use_reader_lora"])
        )
        return original(memories, tokens, **kwargs)

    monkeypatch.setattr(b, "read_batch", read)
    metrics, rows = evaluate_qa(
        b, w, tokenizer, tiny_config, recipe, episodes, torch.device("cpu"), 4
    )
    assert len(rows) == 10 and all(row["prefix_end"] == 12 for row in rows)
    assert all(
        row["delay_writes"] == 1 for row in rows
    )  # Initial paragraphs enter memory together.
    assert all(row["capacity"] == 4 and row["final_compression_ratio"] == 3 for row in rows)
    assert (metrics, rows) == evaluate_qa(
        b, w, tokenizer, tiny_config, recipe, episodes, torch.device("cpu"), 4
    )
    assert all(torch.equal(v, w.state_dict()[k]) for k, v in saved.items())
    assert any(n == 4 for n, _, _ in calls)
    assert any(n == 0 and "First" in prompt and not lora for n, prompt, lora in calls)
    assert all(
        not module.disable_adapters
        for module in b.language_model.modules()
        if hasattr(module, "disable_adapters") and not callable(module.disable_adapters)
    )
    with pytest.raises(ValueError, match="two independent"):
        evaluate_qa(b, w, tokenizer, tiny_config, recipe, episodes[:1], torch.device("cpu"), 4)


def test_squad_scoring():
    assert answer_scores("The Observer.", ["observer"]) == (1, 1)
    assert answer_scores("red blue", ["red green"]) == (0, 0.5)
    assert answer_scores("a", ["the"]) == (1, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"global_batch_size": 0},
        {"global_batch_size": True},
        {"samples_per_micro_epoch": 1},
        {"micro_epochs_per_capacity": 0},
        {"capacities": (4, 4)},
        {"ratios": ()},
        {"stage_ends": (0.7, 0.3, 1)},
        {"ratio_weights": ((0.5, 0.5),) * 3},
        {"ratio_weights": ((0.5, 0.5, 0.5),) * 3},
        {"bptt_span": -1},
        {"bptt_unit": "steps"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        DynamicConfig(**kwargs)


def test_capacity_order_curriculum_and_integer_quotas():
    recipe = DynamicConfig(micro_epochs_per_capacity=3)
    order = recipe.capacity_order(0)
    assert len(order) == 15 and all(order.count(k) == 3 for k in recipe.capacities)
    assert order == recipe.capacity_order(0) and order != recipe.capacity_order(1)
    for e, weights in [
        (0, (0.6, 0.3, 0.1)),
        (2999, (0.6, 0.3, 0.1)),
        (3000, (0.3, 0.4, 0.3)),
        (6999, (0.3, 0.4, 0.3)),
        (7000, (0.1, 0.3, 0.6)),
        (9999, (0.1, 0.3, 0.6)),
    ]:
        assert recipe.weights(e, 10000) == weights
    assert [recipe.weights(e, 3) for e in range(3)] == list(recipe.ratio_weights)
    assert allocate_counts(7, (0.6, 0.3, 0.1)) == [4, 2, 1]
    assert text_bounds(1024, 8) == (7373, 12288)


@pytest.fixture
def training_data(tmp_path, tokenizer):
    tokdir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokdir)
    tokenizer.name_or_path = str(tokdir)
    paths = {}
    for split, indices in [("train", range(30)), ("dev", range(40, 44))]:
        rows = []
        for i in indices:
            paragraphs = [
                {
                    "context": f"First d{i}p{j}.",
                    "qas": [
                        {
                            "id": f"q{i}-{j}",
                            "question": "What?",
                            "answers": [{"text": "First", "answer_start": 0}],
                        }
                    ],
                }
                for j in range(40)
            ]
            rows.append({"title": str(i), "paragraphs": paragraphs})
        paths[split] = tmp_path / f"{split}.json"
        paths[split].write_text(json.dumps({"version": "1.1", "data": rows}))
    index = tmp_path / "tokenizer_index.json"
    prepare_squad(paths["train"], paths["dev"], tokenizer, index)
    return index, SquadDataset(index)


def test_text_construction_no_overlap_offsets_quotas_drop_last(training_data):
    _, data = training_data
    recipe = DynamicConfig(capacities=(4, 8), samples_per_micro_epoch=7, global_batch_size=2)
    sampler = DynamicTextSampler(data, recipe, 256)
    k, texts, report = sampler.micro_epoch(0, 0, 3)
    assert len(texts) == 6 and report["dropped_samples"] == 1
    assert sum(report["requested_counts"].values()) == 7
    assert (k, texts, report) == sampler.micro_epoch(0, 0, 3)
    for text in texts:
        e = text.episode(data)
        assert len(e.input_ids) == text.input_tokens
        assert text_bounds(k, text.ratio)[0] <= len(e.input_ids) <= text_bounds(k, text.ratio)[1]
        assert write_boundaries(e, k)[0] == text.initial_tokens
        assert [s.provenance["paragraph_index"] for s in e.sources] == list(
            range(text.paragraph_start, text.paragraph_end)
        )
        assert data.records[text.document_id]["input_tokens"] > text_bounds(k, text.ratio)[1]
        assert data.records[text.document_id]["split"] == "train"
        for other in texts:
            if text != other and text.document_id == other.document_id:
                assert (
                    text.paragraph_end <= other.paragraph_start
                    or other.paragraph_end <= text.paragraph_start
                )
    panel = sampler.evaluation_texts("test", 1)
    assert panel == sampler.evaluation_texts("test", 1)
    assert all(data.records[t.document_id]["split"] == "test" for ts in panel.values() for t in ts)
    with pytest.raises(ValueError, match="insufficient non-overlapping"):
        sampler.select("train", 4, [100000, 0, 0], 42)


def test_oversize_paragraph_and_initial_only_texts_are_filtered(training_data):
    _, data = training_data
    doc = next(doc for doc, r in data.records.items() if r["split"] == "train")
    data.records = {
        doc: dict(data.records[doc], paragraph_tokens=[3, 3, 100, 3, 3, 3, 3], input_tokens=118)
    }
    data.articles[doc]["paragraphs"] = data.articles[doc]["paragraphs"][:7]
    sampler = DynamicTextSampler(data, DynamicConfig(capacities=(4,)), 256)
    pool = sampler.pool("train", 4, 2)
    assert [(t.paragraph_start, t.paragraph_end) for t in pool] == [(3, 7)]
    assert not DynamicTextSampler(data, DynamicConfig(capacities=(4,)), 9).pool("train", 4, 2)


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_equal(x, y)
    else:
        assert a == b


def test_run_resume_across_micro_epochs_and_final_test(
    tmp_path, tiny_config, tokenizer, training_data, monkeypatch
):
    index, data = training_data
    model_dir = tmp_path / "model"
    torch.manual_seed(7)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    ).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    config = replace(tiny_config, model_name_or_path=str(model_dir))
    _, backbone = load_backbone(config, torch.device("cpu"), torch.float32)
    writer = JointMemoryWriter(
        config.d_mem, config.num_layers, config.num_heads, config.ffn_dim, config.k_limit
    )
    value = GrowthValueNetwork(config.d_mem)
    initial = tmp_path / "initial.pt"
    save_model_checkpoint(
        initial,
        "pretrain",
        config,
        trainable_model_state(backbone, writer, value),
        {},
        {},
        capture_rng_state(),
    )
    recipe = DynamicConfig(
        capacities=(4, 8),
        samples_per_micro_epoch=5,
        global_batch_size=2,
        generation_tokens=1,
        qa_activation_checkpointing=False,
        bptt_unit="updates",
        bptt_span=2,
    )
    recipe = replace(recipe, epochs=2, save_every=1, eval_every=100, eval_texts_per_ratio=1)
    prepare_dynamic(index, recipe, config, tmp_path / "plan")
    opts = dict(evaluation_plan=tmp_path / "plan/evaluation-plan.json")
    full = run_dynamic(initial, index, tmp_path / "full", recipe, torch.device("cpu"), **opts)
    first = run_dynamic(
        initial, index, tmp_path / "resume", recipe, torch.device("cpu"), steps=3, **opts
    )
    resumed = run_dynamic(
        first, index, tmp_path / "resume", recipe, torch.device("cpu"), resume=True, **opts
    )
    runtime = json.loads(next((tmp_path / "full").glob("runtime-from-*.json")).read_text())[
        "runtime"
    ]
    assert runtime["python_executable"] == sys.executable
    assert runtime["environment"] == sys.prefix
    assert runtime["source_file"].endswith("src/latent_working_memory/v1/dynamic.py")
    assert len(runtime["git_commit"]) == 40
    a, b = load_model_checkpoint(full), load_model_checkpoint(resumed)
    assert_equal(a.model_state, b.model_state)
    assert_equal(a.optimizer_state, b.optimizer_state)
    assert_equal(a.progress, b.progress)
    assert a.progress["next_step"] == 8 and a.progress["samples_seen"] == 16
    assert (a.progress["epoch"], a.progress["micro_epoch"], a.progress["batch_in_micro_epoch"]) == (
        2,
        0,
        0,
    )
    for split in ("dev",):
        assert json.loads(
            (tmp_path / f"full/dev/{split}-step-000008.json").read_text()
        ) == json.loads((tmp_path / f"resume/dev/{split}-step-000008.json").read_text())
    rows = []
    for path in (tmp_path / "full").glob("train-*.jsonl"):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    assert all(row["samples"] == 2 for row in rows)
    assert len({row["capacity"] for row in rows}) == 2
    assert not list(tmp_path.rglob("swanlab.json"))
    for plan in (tmp_path / "full/data_plans").glob("micro-*.json"):
        record = json.loads(plan.read_text())
        assert record["dropped_samples"] == 1
        assert record["training_totals"]["samples"] == 4
        assert record == json.loads((tmp_path / "resume/data_plans" / plan.name).read_text())
    resumed_rows = sorted(
        (
            json.loads(line)
            for path in (tmp_path / "resume").glob("train-*.jsonl")
            for line in path.read_text().splitlines()
        ),
        key=lambda row: row["step"],
    )
    for full_row, resumed_row in zip(rows, resumed_rows, strict=True):
        assert full_row["sample_metrics"] == resumed_row["sample_metrics"]
    with pytest.raises(ValueError, match="configuration differs"):
        run_dynamic(
            first,
            index,
            tmp_path / "resume",
            replace(recipe, seed=43),
            torch.device("cpu"),
            resume=True,
            **opts,
        )
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(asdict(recipe)))
    monkeypatch.setattr(
        "sys.argv",
        [
            "dynamic",
            "evaluate",
            "--checkpoint",
            str(full),
            "--index",
            str(index),
            "--config",
            str(recipe_path),
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--evaluation-plan",
            str(tmp_path / "plan/evaluation-plan.json"),
            "--device",
            "cpu",
            "--split",
            "test",
        ],
    )
    main()
    assert (tmp_path / "evaluation/test-step-000008.json").is_file()


def distributed_worker(rank, rendezvous, output, components, config, tokenizer):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        b, w = deepcopy(components)
        recipe = replace(small_recipe(), global_batch_size=2)
        trainer = DynamicTrainer(b, w, config, recipe, torch.device("cpu"))
        episodes = [example(tokenizer, "a", 4), example(tokenizer, "b", 6)]
        result = trainer.step(episodes, tokenizer, [42, 43], 4)
        metrics, rows = evaluate_qa(
            b, w, tokenizer, config, recipe, episodes, torch.device("cpu"), 4
        )
        torch.save(
            {
                "parameters": [p.detach() for p in trainer.parameters],
                "result": result,
                "metrics": metrics,
                "rows": rows,
            },
            output / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_distributed_matches_single_process(components, tiny_config, tokenizer, tmp_path):
    torch.multiprocessing.spawn(
        distributed_worker,
        args=((tmp_path / "gloo").as_uri(), tmp_path, deepcopy(components), tiny_config, tokenizer),
        nprocs=2,
        join=True,
    )
    b, w = components
    recipe = replace(small_recipe(), global_batch_size=2)
    trainer = DynamicTrainer(b, w, tiny_config, recipe, torch.device("cpu"))
    episodes = [example(tokenizer, "a", 4), example(tokenizer, "b", 6)]
    result = trainer.step(episodes, tokenizer, [42, 43], 4)
    metrics, rows = evaluate_qa(
        b, w, tokenizer, tiny_config, recipe, episodes, torch.device("cpu"), 4
    )
    a = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    c = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    for expected, x, y in zip(trainer.parameters, a["parameters"], c["parameters"], strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        torch.testing.assert_close(x, expected, rtol=1e-4, atol=1e-6)
    assert a["result"]["loss"] == pytest.approx(result["loss"])
    assert len(a["rows"]) == len(rows)
    for key, values in metrics.items():
        assert a["metrics"][key] == pytest.approx(values, rel=1e-5)


def test_evaluation_without_generation_matches_nll_and_caches_controls(
    components, tiny_config, tokenizer, monkeypatch, tmp_path
):
    b, w = components
    recipe = replace(small_recipe(max_visits=6), eval_reads_per_kind=20)
    episodes = [example(tokenizer, "a", 7), example(tokenizer, "b", 7)]
    _, generated = evaluate_qa(
        b, w, tokenizer, tiny_config, recipe, episodes, torch.device("cpu"), 4
    )
    calls = []
    original = b.read_batch

    def read(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def no_generation(*args, **kwargs):
        raise AssertionError("NLL-only evaluation invoked generation")

    monkeypatch.setattr(b, "read_batch", read)
    monkeypatch.setattr(b, "greedy_students", no_generation)
    metrics, rows = evaluate_qa(
        b,
        w,
        tokenizer,
        tiny_config,
        recipe,
        episodes,
        torch.device("cpu"),
        4,
        generate=False,
    )
    assert len(calls) < len(rows)
    for a, c in zip(generated, rows, strict=True):
        assert a["nll_sum"] == c["nll_sum"]
        assert a["target_tokens"] == c["target_tokens"]
        assert "prediction" not in c
    assert all("em" not in values and values["generations"] == 0 for values in metrics.values())
    for row in generated:
        row["target_ratio"] = 8
    report = aggregate_qa(generated)
    media = qa_media(report, generated)
    assert (
        "evaluation/test/f1" in media
        and "tables/test/paired" in media
        and media["examples/test/qa"]
    )
    assert report["paired/memory-minus-no_memory"]["reads"] == len(generated) // 5
    records = tmp_path / "test.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in generated))
    reports = tmp_path / "reports.json"
    reports.write_text(
        json.dumps(
            [
                {"name": "full", "report": "test.json"},
                {"name": "tokens", "report": "test.json"},
            ]
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "report",
            "--reports",
            str(reports),
            "--output-dir",
            str(tmp_path / "comparison"),
            "--swanlab-group",
            "test",
            "--swanlab-mode",
            "disabled",
        ],
    )
    publish_qa()
    comparison = json.loads((tmp_path / "comparison/comparison.json").read_text())
    assert comparison["full"] == comparison["tokens"] == report
    assert not list(tmp_path.rglob("swanlab.json"))


def test_training_curves_keep_real_steps_and_generation_schedule(tmp_path):
    history = tmp_path / "dev"
    history.mkdir()
    for step, nll, em in ((0, 4.0, 0.1), (100, 3.0, None), (250, 2.0, 0.4)):
        report = {}
        for condition in (
            "memory",
            "no_memory",
            "wrong_memory",
            "gold_paragraph",
            "gold_paragraph_base",
        ):
            values = {"nll": nll}
            if em is not None:
                values.update(em=em, f1=em + 0.1, hit_limit_rate=0.0)
            for kind, offset in (("all", 0.0), ("arrival", -0.5), ("delayed", 0.5)):
                report[f"overall/{condition}/{kind}"] = {**values, "nll": nll + offset}
            if condition == "memory":
                for capacity in (64, 1024):
                    for kind, offset in (("all", 0.0), ("arrival", -0.5), ("delayed", 0.5)):
                        report[f"k{capacity}/memory/{kind}"] = {**values, "nll": nll + offset}
            else:
                paired = {"nll_difference": -0.5}
                if em is not None:
                    paired.update(em_difference=0.1, f1_difference=0.2)
                report[f"paired/memory-minus-{condition}"] = paired
        (history / f"dev-step-{step:06d}.json").write_text(json.dumps(report))
    curves = training_curves(history, 250)
    assert len(curves) == 8
    assert all(key.startswith("evaluation/dev/") for key in curves)
    capacity_chart = curves["evaluation/dev/by-capacity/nll"].options
    assert capacity_chart["baseOption"]["timeline"]["data"] == ["all", "arrival", "delayed"]
    capacity = capacity_chart["options"][0]["series"]
    assert [series["name"] for series in capacity] == ["K=64", "K=1024"]
    em_chart = curves["evaluation/dev/em"].options
    assert em_chart["baseOption"]["timeline"]["data"] == ["all", "arrival", "delayed", "paired"]
    assert em_chart["baseOption"]["timeline"]["replaceMerge"] == ["series"]
    assert em_chart["baseOption"]["timeline"]["autoPlay"] is False
    paired = em_chart["options"][3]["series"]
    assert len(paired) == 4 and paired[0]["data"] == [[0, 0.1], [250, 0.1]]
    assert em_chart["options"][3]["yAxis"][0]["name"] == "em_difference"
    for chart in curves.values():
        option = json.loads(chart.dump_options())
        assert option["baseOption"]["timeline"]["currentIndex"] == 0
        assert option["series"] == option["options"][0]["series"]
        assert option["series"] and all(series["type"] == "line" for series in option["series"])

    class Recorder:
        def log(self, values, step):
            self.values, self.step = values, step

    run = Recorder()
    log_qa(run, report, [], 250, "dev", media=True, history_dir=history)
    assert run.step == 250
    assert set(run.values) == set(curves) | {
        "tables/dev/capacity-and-ratio",
        "tables/dev/paired",
        "examples/dev/qa",
    }
    nll_views = curves["evaluation/dev/nll"].options["options"]
    assert nll_views[1]["series"][0]["data"] == [[0, 3.5], [100, 2.5], [250, 1.5]]
    assert nll_views[2]["series"][0]["data"] == [[0, 4.5], [100, 3.5], [250, 2.5]]
    nll = nll_views[0]
    assert nll["xAxis"][0]["type"] == "value"
    assert nll["series"][0]["data"] == [[0, 4.0], [100, 3.0], [250, 2.0]]
    assert len(nll["series"]) == 5 and all(s["type"] == "line" for s in nll["series"])
    assert curves["evaluation/dev/em"].options["options"][0]["series"][0]["data"] == [
        [0, 0.1],
        [250, 0.4],
    ]
    assert training_curves(history, 100)["evaluation/dev/em"].options["options"][0]["series"][0][
        "data"
    ] == [[0, 0.1]]
    (tmp_path / "config.json").write_text(
        json.dumps({"eval_every": 100, "eval_generation_every": 250})
    )
    (tmp_path / "provenance.json").write_text(json.dumps({"target_steps": 250}))
    with pytest.raises(ValueError, match="completed training run"):
        publish_training_history(tmp_path, 251, "disabled")
    (tmp_path / "resources-from-000000-test.json").write_text(json.dumps({"completed_steps": 250}))
    (tmp_path / "swanlab.json").write_text(
        json.dumps({"project": "test", "group": "test", "tags": []})
    )
    with pytest.raises(ValueError, match="media step"):
        publish_training_history(tmp_path, 250, "disabled")
    publish_training_history(tmp_path, 251, "disabled")
    record = json.loads((tmp_path / "evaluation-history-000251.json").read_text())
    assert record["checkpoint_step"] == 250 and record["media_step"] == 251


def test_rebuild_training_run_replays_original_steps_without_old_charts(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    dev = source / "dev"
    dev.mkdir()
    identity = {"id": "original", "project": "test", "group": "series", "tags": []}
    (source / "swanlab.json").write_text(json.dumps(identity))
    (source / "provenance.json").write_text(json.dumps({"target_steps": 2}))
    (source / "config.json").write_text(json.dumps({"eval_every": 2, "eval_generation_every": 2}))
    (source / "resources-from-000000.json").write_text(json.dumps({"completed_steps": 2}))
    records = [
        {
            "step": step,
            "loss": 3.0 / step,
            "seconds": 12.0,
            "cumulative": {"input_tokens": 200 * step},
            "sample_metrics": [],
        }
        for step in (1, 2)
    ]
    (source / "train-from-000000.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    for step in (0, 2):
        (dev / f"dev-step-{step:06d}.json").write_text(
            json.dumps({"overall/memory/all": {"nll": 4.0 - step}})
        )
        (dev / f"dev-step-{step:06d}.jsonl").write_text("")
    calls = []

    class Recorder:
        def log(self, values, step):
            calls.append((step, values))

    @contextmanager
    def fake_run(output_dir, config, mode, project, **kwargs):
        assert config["report_source"]["swanlab_id"] == "original"
        assert project == "test" and kwargs["group"] == "series"
        yield Recorder()

    monkeypatch.setattr(dynamic_reporting, "swanlab_run", fake_run)
    output = tmp_path / "rebuilt"
    dynamic_reporting.rebuild_training_run(source, output, "disabled")
    assert [step for step, _ in calls] == [0, 1, 2, 2]
    assert calls[1][1] == {
        "train/step": 1,
        "train/loss": 3.0,
        "resources/seconds": 12.0,
        "train/cumulative_input_tokens": 200,
    }
    assert set(calls[0][1]) == set(calls[-1][1]) == {"evaluation/dev/nll"}
    assert json.loads((source / "swanlab.json").read_text()) == identity
    assert json.loads((output / "republication.json").read_text())["training_steps"] == 2
    (output / "swanlab.json").write_text(json.dumps({**identity, "id": "rebuilt"}))
    dynamic_reporting.publish_training_history(source, 3, "disabled", output)
    assert calls[-1][0] == 3 and set(calls[-1][1]) == {"evaluation/dev/nll"}
    assert (output / "evaluation-history-000003.json").exists()
    assert not (source / "evaluation-history-000003.json").exists()
    assert json.loads((source / "swanlab.json").read_text())["id"] == "original"
    with pytest.raises(FileExistsError):
        dynamic_reporting.rebuild_training_run(source, output, "disabled")
    with (source / "train-from-000000.jsonl").open("a") as stream:
        stream.write(json.dumps(records[-1]) + "\n")
    with pytest.raises(ValueError, match="exactly one training record"):
        dynamic_reporting.rebuild_training_run(source, tmp_path / "invalid", "disabled")
    assert not (tmp_path / "invalid").exists()
