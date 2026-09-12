from copy import deepcopy
from dataclasses import replace
from itertools import islice
import json

import pytest
import torch
import torch.distributed as dist
from transformers import LlamaConfig, LlamaForCausalLM

from latent_working_memory.data_preparation.squad import prepare_squad
from latent_working_memory.v1.backbone import load_backbone
from latent_working_memory.v1.checkpoint import (
    capture_rng_state,
    load_model_checkpoint,
    save_model_checkpoint,
)
from latent_working_memory.v1.data import Episode, Read, Reference, Source
from latent_working_memory.v1.dynamic import (
    DynamicConfig,
    DynamicTrainer,
    answer_scores,
    evaluate_qa,
    read_schedule,
    run_dynamic,
    shuffled_articles,
    main,
)
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.v1.squad import QA_PROMPT
from latent_working_memory.v1.training import trainable_model_state


def example(tokenizer, name="doc"):
    first = tuple(tokenizer.encode("First sentence.", add_special_tokens=False))
    second = tuple(tokenizer.encode("Second sentence.", add_special_tokens=False))
    return Episode(
        name,
        first + second,
        (len(first), len(first + second)),
        (
            Source(name + ":0", name, 0, len(first), {"context": "First sentence."}),
            Source(
                name + ":1", name, len(first), len(first + second), {"context": "Second sentence."}
            ),
        ),
        (
            Read(
                name + ":q",
                "qa",
                len(first),
                QA_PROMPT.format(question="What?"),
                (Reference("First", ((0, len(first)),)),),
            ),
        ),
    )


@pytest.mark.parametrize(
    "unit,span", [("tokens", 0), ("tokens", 1), ("updates", 1), ("updates", 2)]
)
def test_delayed_only_gradient_and_frozen_base(components, tiny_config, tokenizer, unit, span):
    truncate = span == 1
    backbone, writer = components
    e = example(tokenizer)
    recipe = DynamicConfig(
        4, 0, 128, new_count=0, history_count=1, max_visits=1, bptt_unit=unit, bptt_span=span
    )
    features = []
    original = backbone.text_features

    def trace(units, starts):
        values = original(units, starts)
        for value in values:
            value.retain_grad()
            features.append(value)
        return values

    backbone.text_features = trace
    base = {
        name: p.detach().clone()
        for name, p in backbone.language_model.named_parameters()
        if not p.requires_grad
    }
    trainer = DynamicTrainer(backbone, writer, tiny_config, recipe, torch.device("cpu"))
    metrics = trainer.step([e], tokenizer, [42])
    assert metrics["reads"] == 1
    assert metrics["truncations"] == int(truncate)
    if truncate:
        assert features[0].grad is None
    else:
        assert features[0].grad.abs().sum() > 0
    assert features[1].grad.abs().sum() > 0
    assert backbone.memory_projection.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in backbone.language_model.named_parameters()
        if "lora_" in n
    )
    for name, p in backbone.language_model.named_parameters():
        if name in base:
            assert torch.equal(base[name], p)
            assert p.grad is None


def test_squad_scoring():
    assert answer_scores("The Observer.", ["observer"]) == (1, 1)
    assert answer_scores("red blue", ["red green"]) == (0, 0.5)
    assert answer_scores("three", ["two", "Three!"]) == (1, 1)
    assert answer_scores("", ["one"]) == (0, 0)
    assert answer_scores("a", ["the"]) == (1, 0)  # Official v1 F1 has zero overlap.


def test_evaluation_is_paired_deterministic_and_read_only(components, tiny_config, tokenizer):
    backbone, writer = components
    recipe = DynamicConfig(4, 0, 128, generation_tokens=3)
    episodes = [example(tokenizer, "a"), example(tokenizer, "b")]
    saved = {k: v.clone() for k, v in writer.state_dict().items()}
    metrics, rows = evaluate_qa(
        backbone, writer, tokenizer, tiny_config, recipe, episodes, torch.device("cpu")
    )
    assert len(rows) == 20
    assert set(metrics) == {
        f"{c}/{k}"
        for c in ("memory", "no_memory", "wrong_memory", "gold_paragraph", "gold_paragraph_base")
        for k in ("all", "arrival", "delayed")
    }
    assert all(torch.equal(v, writer.state_dict()[k]) for k, v in saved.items())
    assert (metrics, rows) == evaluate_qa(
        backbone, writer, tokenizer, tiny_config, recipe, episodes, torch.device("cpu")
    )
    assert all(r["delay_tokens"] > 0 for r in rows if r["kind"] == "delayed")
    with pytest.raises(ValueError, match="two independent"):
        evaluate_qa(
            backbone, writer, tokenizer, tiny_config, recipe, episodes[:1], torch.device("cpu")
        )
    with pytest.raises(ValueError, match="context budget"):
        read_schedule(
            episodes[0], tokenizer, recipe, replace(tiny_config, read_context_tokens=2), 42
        )


@pytest.mark.parametrize("accumulation", [1, 2, 10])
def test_dynamic_run_resume_matches_uninterrupted(
    tmp_path, tiny_config, tokenizer, monkeypatch, accumulation
):
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
    tok, backbone = load_backbone(config, torch.device("cpu"), torch.float32)
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
    paths = {}
    for split, indices in [("train", range(20)), ("dev", range(30, 32))]:
        articles = []
        for i in indices:
            paragraphs = [
                {
                    "context": f"First sentence {i} {j}.",
                    "qas": [
                        {
                            "id": f"q{i}-{j}",
                            "question": "What?",
                            "answers": [{"text": "First", "answer_start": 0}],
                        }
                    ],
                }
                for j in range(2)
            ]
            articles.append({"title": str(i), "paragraphs": paragraphs})
        paths[split] = tmp_path / f"{split}.json"
        paths[split].write_text(json.dumps({"version": "1.1", "data": articles}))
    index = tmp_path / "test-tokenizer_index.json"
    prepare_squad(paths["train"], paths["dev"], tok, index)
    recipe = DynamicConfig(4, 0, 128, generation_tokens=2, gradient_accumulation_steps=accumulation)
    full = run_dynamic(
        initial,
        index,
        tmp_path / "full",
        recipe,
        torch.device("cpu"),
        2,
        save_every=1,
        eval_every=1,
        eval_articles=2,
    )
    first = run_dynamic(
        initial,
        index,
        tmp_path / "resume",
        recipe,
        torch.device("cpu"),
        1,
        save_every=1,
        eval_every=1,
        eval_articles=2,
        swanlab_mode="disabled",
        swanlab_group="dynamic-test",
    )
    resumed = run_dynamic(
        first,
        index,
        tmp_path / "resume",
        recipe,
        torch.device("cpu"),
        2,
        save_every=1,
        eval_every=1,
        eval_articles=2,
        swanlab_mode="disabled",
        swanlab_group="dynamic-test",
        resume=True,
    )

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b

    a, b = load_model_checkpoint(full), load_model_checkpoint(resumed)
    equal(a.model_state, b.model_state)
    equal(a.optimizer_state, b.optimizer_state)
    equal(a.progress, b.progress)
    assert json.loads((tmp_path / "full/dev-000002.json").read_text()) == json.loads(
        (tmp_path / "resume/dev-000002.json").read_text()
    )
    rows = [
        json.loads(line)
        for path in (tmp_path / "full").glob("train-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert [row["articles_seen"] for row in rows] == [accumulation, 2 * accumulation]
    assert all(row["articles"] == accumulation for row in rows)
    assert all(len(row["article_metrics"]) == accumulation for row in rows)
    train_docs = [
        row["document_id"]
        for row in json.loads(index.read_text())["articles"]
        if row["split"] == "train"
    ]
    consumed = [
        article["episode_id"].rsplit(":prefix:", 1)[0]
        for row in rows
        for article in row["article_metrics"]
    ]
    assert consumed == list(islice(shuffled_articles(train_docs, recipe.seed), 2 * accumulation))
    assert all(
        (row["epochs_completed"], row["articles_into_epoch"])
        == divmod(row["articles_seen"], len(train_docs))
        for row in rows
    )
    if accumulation == 10:
        assert len(set(consumed[: len(train_docs)])) == len(train_docs)
        assert rows[-1]["epochs_completed"] == 1
    resumed_rows = sorted(
        (
            json.loads(line)
            for path in (tmp_path / "resume").glob("train-*.jsonl")
            for line in path.read_text().splitlines()
        ),
        key=lambda row: row["step"],
    )
    for row, resumed_row in zip(rows, resumed_rows, strict=True):
        for key in ("seconds", "input_tokens_per_second"):
            row.pop(key)
            resumed_row.pop(key)
        assert row == resumed_row
    assert not (tmp_path / "resume/swanlab.json").exists()
    recipe_path = tmp_path / "recipe.json"
    from_dataclass = {name: getattr(recipe, name) for name in recipe.__dataclass_fields__}
    recipe_path.write_text(json.dumps(from_dataclass))
    monkeypatch.setattr(
        "sys.argv",
        [
            "dynamic",
            "evaluate",
            "--checkpoint",
            str(initial),
            "--index",
            str(index),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--eval-articles",
            "2",
            "--split",
            "test",
        ],
    )
    main()
    assert (tmp_path / "evaluation/metrics.json").is_file()
    assert json.loads((tmp_path / "evaluation/evaluation.json").read_text())["split"] == "test"
    if accumulation == 10:
        exact = run_dynamic(
            first,
            index,
            tmp_path / "resume",
            recipe,
            torch.device("cpu"),
            100,
            eval_articles=2,
            resume=True,
            epochs=1,
        )
        exact_checkpoint = load_model_checkpoint(exact)
        assert exact_checkpoint.progress["articles_seen"] == len(train_docs)
        assert exact_checkpoint.progress["next_step"] == 2
    with pytest.raises(ValueError, match="configuration differs"):
        run_dynamic(
            resumed,
            index,
            tmp_path / "resume",
            replace(recipe, capacity=8),
            torch.device("cpu"),
            3,
            resume=True,
        )


@pytest.mark.parametrize(
    "unit,span,expected",
    [
        ("tokens", 4, [(6, 2), (3, 1)]),
        ("tokens", 1, [(3, 1), (3, 1), (3, 1)]),
        ("updates", 2, [(6, 2), (3, 1)]),
        ("updates", 0, [(9, 3)]),
    ],
)
def test_segment_boundaries_and_single_optimizer_step(
    components, tiny_config, tokenizer, monkeypatch, unit, span, expected
):
    backbone, writer = components
    e = example(tokenizer)
    # All reads arrive after the first write; a token span shorter than one write
    # must still backpropagate that write's loss before detaching.
    e = replace(e, input_ids=e.input_ids + e.input_ids[:3], write_ends=(3, 6, 9))
    recipe = DynamicConfig(4, 0, 128, bptt_unit=unit, bptt_span=span)
    trainer = DynamicTrainer(backbone, writer, tiny_config, recipe, torch.device("cpu"))
    calls = []
    original_step = trainer.optimizer.step

    def step():
        calls.append(1)
        return original_step()

    monkeypatch.setattr(trainer.optimizer, "step", step)
    features = []
    original_features = backbone.text_features

    def trace(units, starts):
        result = original_features(units, starts)
        result[0].retain_grad()
        features.append(result[0])
        return result

    monkeypatch.setattr(backbone, "text_features", trace)
    result = trainer.step([e], tokenizer, [42])
    assert [
        (s["tokens"], s["updates"]) for s in result["article_metrics"][0]["bptt_segments"]
    ] == expected
    assert result["truncations"] == len(expected) - 1
    assert calls == [1]
    assert features[0].grad.abs().sum() > 0


def test_gold_paragraph_inputs_budget_and_adapter_restoration(
    components, tiny_config, tokenizer, monkeypatch
):
    backbone, writer = components
    recipe = DynamicConfig(4, 0, 128, generation_tokens=2)
    episode = example(tokenizer)
    calls = []
    read_batch = backbone.read_batch
    greedy = backbone.greedy_students

    def read(memories, tokens, **kwargs):
        calls.append(("nll", memories[0].shape[0], tokens[0].prompt_ids, kwargs["use_reader_lora"]))
        return read_batch(memories, tokens, **kwargs)

    def generate(memories, prompts, limits, **kwargs):
        calls.append(("generate", memories[0].shape[0], prompts[0], kwargs["use_reader_lora"]))
        return greedy(memories, prompts, limits, **kwargs)

    monkeypatch.setattr(backbone, "read_batch", read)
    monkeypatch.setattr(backbone, "greedy_students", generate)
    conditions = ("gold_paragraph", "gold_paragraph_base")
    _, rows = evaluate_qa(
        backbone,
        writer,
        tokenizer,
        tiny_config,
        recipe,
        [episode],
        torch.device("cpu"),
        conditions=conditions,
    )
    expected = tuple(
        tokenizer.encode(
            "Text:\nFirst sentence.\n\n"
            + QA_PROMPT.format(question="What?").replace(
                "information stored in memory", "provided text"
            ),
            add_special_tokens=False,
        )
    )
    assert len(calls) == 8
    assert all(n == 0 and prompt == expected for _, n, prompt, _ in calls)
    assert [enabled for _, _, _, enabled in calls] == [True, True, False, False] * 2
    assert len(rows) == 4 and rows[-1]["kind"] == "delayed"
    assert all(
        not module.disable_adapters
        for module in backbone.language_model.modules()
        if hasattr(module, "disable_adapters") and not callable(module.disable_adapters)
    )
    # Memory QA fits, but the original evidence paragraph does not.
    with pytest.raises(ValueError, match="gold_paragraph QA read exceeds context budget"):
        evaluate_qa(
            backbone,
            writer,
            tokenizer,
            replace(
                tiny_config,
                read_context_tokens=1
                + 1
                + len(tokenizer.encode(episode.reads[0].prompt, add_special_tokens=False))
                + 2,
            ),
            replace(recipe, capacity=1),
            [episode],
            torch.device("cpu"),
            conditions=conditions,
        )


@pytest.mark.parametrize("kwargs", [{"bptt_unit": "steps"}, {"bptt_span": -1}, {"bptt_span": 1.5}])
def test_invalid_truncation_config(kwargs):
    with pytest.raises(ValueError, match="bptt"):
        DynamicConfig(4, 0, 128, **kwargs)


@pytest.mark.parametrize("unit,span", [("tokens", 0), ("tokens", 1), ("updates", 2)])
def test_article_accumulation_averages_gradients_and_resets_memory(
    components, tiny_config, tokenizer, monkeypatch, unit, span
):
    backbone, writer = components
    original = example(tokenizer)
    short = replace(
        original,
        input_ids=original.input_ids[:3],
        write_ends=(3,),
        sources=original.sources[:1],
    )
    long = replace(
        original, input_ids=original.input_ids + original.input_ids[:3], write_ends=(3, 6, 9)
    )
    episodes = [short, long]
    recipe = DynamicConfig(
        4,
        0,
        128,
        max_visits=3,
        bptt_unit=unit,
        bptt_span=span,
        gradient_clip=1e9,
    )
    reference_grads, reference_metrics = [], []
    for episode in episodes:
        b, w = deepcopy((backbone, writer))
        trainer = DynamicTrainer(b, w, tiny_config, recipe, torch.device("cpu"))
        reference_metrics.append(trainer.step([episode], tokenizer, [42]))
        reference_grads.append(
            [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in trainer.parameters]
        )

    trainer = DynamicTrainer(
        backbone,
        writer,
        tiny_config,
        replace(recipe, gradient_accumulation_steps=2),
        torch.device("cpu"),
    )
    updates, initializations = [], []
    original_step = trainer.optimizer.step
    original_init = writer.initialize_state

    def step():
        updates.append(1)
        return original_step()

    def initialize(*args, **kwargs):
        initializations.append(1)
        return original_init(*args, **kwargs)

    monkeypatch.setattr(trainer.optimizer, "step", step)
    monkeypatch.setattr(writer, "initialize_state", initialize)
    result = trainer.step(episodes, tokenizer, [42, 42])
    assert updates == [1] and initializations == [1, 1]
    assert result["articles"] == 2
    assert [a["reads"] for a in result["article_metrics"]] == [1, 3]
    assert result["loss"] == pytest.approx(sum(m["loss"] for m in reference_metrics) / 2)
    for parameter, first, second in zip(trainer.parameters, *reference_grads, strict=True):
        actual = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
        torch.testing.assert_close(actual, (first + second) / 2, atol=1e-6, rtol=1e-4)
    assert result["target_nll"] == pytest.approx(
        sum(m["target_nll"] * m["target_tokens"] for m in reference_metrics)
        / sum(m["target_tokens"] for m in reference_metrics)
    )
    with pytest.raises(ValueError, match="articles and seeds"):
        trainer.step(episodes[:1], tokenizer, [42])


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_article_accumulation(value):
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        DynamicConfig(4, 0, 128, gradient_accumulation_steps=value)


def test_shuffled_article_epochs_cover_every_document_and_resume():
    docs = [f"article-{i}" for i in range(5)]
    sequence = list(islice(shuffled_articles(docs, 42), 20))
    assert docs == [f"article-{i}" for i in range(5)]
    for start in range(0, 20, 5):
        assert sorted(sequence[start : start + 5]) == sorted(docs)
    assert sequence[:5] != sequence[5:10]
    assert sequence != list(islice(shuffled_articles(docs, 43), 20))
    for offset in (0, 3, 5, 7, 10):
        assert list(islice(shuffled_articles(docs, 42, offset), 20 - offset)) == sequence[offset:]
    # B=8 crosses epoch boundaries, with no discarded tail or incomplete batch.
    stream = shuffled_articles(docs, 42)
    batches = [[next(stream) for _ in range(8)] for _ in range(2)]
    assert batches[0] + batches[1] == sequence[:16]
    assert list(islice(shuffled_articles(["only"], 42, 3), 4)) == ["only"] * 4


def test_partial_final_batch_uses_actual_article_weight(components, tiny_config, tokenizer):
    b, w = components
    other_b, other_w = deepcopy((b, w))
    episode = example(tokenizer)
    single = DynamicTrainer(b, w, tiny_config, DynamicConfig(4, 0, 128), torch.device("cpu"))
    partial = DynamicTrainer(
        other_b,
        other_w,
        tiny_config,
        DynamicConfig(4, 0, 128, gradient_accumulation_steps=2),
        torch.device("cpu"),
    )
    single_result = single.step([episode], tokenizer, [42])
    partial_result = partial.step([episode], tokenizer, [42], allow_partial=True)
    assert single_result == partial_result
    for a, b in zip(single.parameters, partial.parameters, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def _distributed_dynamic_worker(rank, rendezvous, output_dir, components, config, tokenizer):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        backbone, writer = deepcopy(components)
        recipe = DynamicConfig(4, 0, 128, gradient_accumulation_steps=2, generation_tokens=2)
        trainer = DynamicTrainer(backbone, writer, config, recipe, torch.device("cpu"))
        episodes = [example(tokenizer, "a"), example(tokenizer, "b")]
        full = trainer.step(episodes, tokenizer, [42, 43])
        # On the last partial batch rank 1 has no article, but must synchronize.
        partial = trainer.step(episodes[:1], tokenizer, [44], allow_partial=True)
        metrics, rows = evaluate_qa(
            backbone,
            writer,
            tokenizer,
            config,
            recipe,
            episodes,
            torch.device("cpu"),
        )
        torch.save(
            {
                "parameters": [p.detach() for p in trainer.parameters],
                "full": full,
                "partial": partial,
                "metrics": metrics,
                "rows": rows,
            },
            output_dir / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_distributed_dynamic_matches_single_process(components, tiny_config, tokenizer, tmp_path):
    torch.multiprocessing.spawn(
        _distributed_dynamic_worker,
        args=((tmp_path / "gloo").as_uri(), tmp_path, deepcopy(components), tiny_config, tokenizer),
        nprocs=2,
        join=True,
    )
    backbone, writer = components
    recipe = DynamicConfig(4, 0, 128, gradient_accumulation_steps=2, generation_tokens=2)
    trainer = DynamicTrainer(backbone, writer, tiny_config, recipe, torch.device("cpu"))
    episodes = [example(tokenizer, "a"), example(tokenizer, "b")]
    full = trainer.step(episodes, tokenizer, [42, 43])
    partial = trainer.step(episodes[:1], tokenizer, [44], allow_partial=True)
    metrics, rows = evaluate_qa(
        backbone,
        writer,
        tokenizer,
        tiny_config,
        recipe,
        episodes,
        torch.device("cpu"),
    )
    rank0 = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    for expected, a, b in zip(
        trainer.parameters, rank0["parameters"], rank1["parameters"], strict=True
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a, expected, rtol=1e-4, atol=1e-6)
    assert rank0["full"]["loss"] == pytest.approx(full["loss"])
    assert rank0["partial"]["loss"] == pytest.approx(partial["loss"])
    assert len(rank0["rows"]) == len(rows)
    for group, values in metrics.items():
        assert rank0["metrics"][group] == pytest.approx(values, rel=1e-5)
