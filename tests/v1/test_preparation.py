from __future__ import annotations

from dataclasses import replace
import json
import socket

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.data_preparation.dedup import cluster_documents
from latent_working_memory.data_preparation.fineweb import document_episodes
from latent_working_memory.data_preparation.pipeline import prepare_fineweb
from latent_working_memory.data_preparation.quality import (
    sentence_quality_flags,
    sentence_rejection_reason,
)
from latent_working_memory.data_preparation.scoring import (
    DocumentScorer,
    parse_review,
    review_blocks,
    score_token_windows,
)
from latent_working_memory.v1.data import EpisodeIndex, read_episodes
from latent_working_memory.v1.sampling import PretrainExample, PretrainSampler, read_tokens
from latent_working_memory.v1.training import PretrainTrainer, pretrain_forward, run_pretraining
from latent_working_memory.v1.evaluation import evaluate_pretraining
from latent_working_memory.v1.config import write_resolved_config
from latent_working_memory.data_preparation.__main__ import main as prepare_main


def test_ae_keeps_document_end_and_text_before_unusable_continuation(
    tiny_config, tokenizer, source_records
):
    config = replace(tiny_config, views_per_granularity=32)
    text = "First sentence ends.\nRead more.\nSecond sentence ends."
    episodes = document_episodes(dict(source_records[0], text=text), tokenizer, config)
    single = [e for e in episodes if e.sources[0].provenance["granularity"] == "sentence"]
    assert {e.reads[0].references[0].text for e in single} == {
        "First sentence ends.",
        "Second sentence ends.",
    }
    assert all(len(e.reads) == 1 for e in single)
    one_sentence = document_episodes(
        dict(source_records[0], text="One sentence ends."), tokenizer, config
    )
    assert one_sentence and all(read_tokens(e, tokenizer)[1] is None for e in one_sentence)


def test_continuation_can_exceed_eight_complete_sentences(tiny_config, tokenizer, source_records):
    record = dict(source_records[0], text=" ".join(f"Sentence number {i} ends." for i in range(20)))
    lengths = []
    for seed in range(12):
        config = replace(
            tiny_config, data_seed=seed, max_continuation_tokens=128, views_per_granularity=32
        )
        for episode in document_episodes(record, tokenizer, config):
            provenance = episode.sources[0].provenance
            if provenance["granularity"] == "sentence" and provenance["sentence_range"] == [0, 1]:
                start, end = provenance["continuation_sentence_range"]
                lengths.append(end - start)
    assert max(lengths) > 8


def test_narrative_mentions_and_internal_ellipsis_are_quality_flags():
    text = "The report criticizes the website's privacy policy."
    assert sentence_rejection_reason(text) is None
    assert sentence_quality_flags(text) == ["template_phrase"]
    assert sentence_rejection_reason("She paused... then continued speaking.") is None
    assert sentence_rejection_reason("Read more.") == "boilerplate"


def test_independent_nll_windows_do_not_condition_on_unrelated_excerpts(tokenizer):
    torch.manual_seed(9)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()
    tokens = [4, 5, 6, 7, 8, 9, 4, 5, 6]
    result = score_token_windows(model, tokenizer, tokens, 4, 3)
    changed = score_token_windows(model, tokenizer, [9, 9, 9, 9] + tokens[4:], 4, 1)
    assert [w["token_span"] for w in result["windows"]] == [[0, 4], [4, 8], [8, 9]]
    assert result["tokens"] == 9
    for original, replacement in zip(result["windows"][1:], changed["windows"][1:], strict=True):
        assert original["nll_sum"] == pytest.approx(replacement["nll_sum"], abs=1e-6)
    assert result["nll"] == pytest.approx(sum(w["nll_sum"] for w in result["windows"]) / 9)


def test_local_score_cache_reuses_scores_and_invalidates_changed_text(
    tmp_path, tokenizer, source_records, monkeypatch
):
    path = tmp_path / "local-model"
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
        )
    )
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    recipe = PreparationConfig(fluency_model_name_or_path=str(path), score_block_tokens=8)
    cache = tmp_path / "scores.jsonl"
    first = DocumentScorer(recipe, cache, torch.device("cpu"))
    expected = first.score(source_records[0])
    second = DocumentScorer(recipe, cache, torch.device("cpu"))

    def unexpected_forward(*args, **kwargs):
        raise RuntimeError("cache miss")

    monkeypatch.setattr(second.models[str(path)][1], "forward", unexpected_forward)
    assert second.score(source_records[0]) == expected
    with pytest.raises(RuntimeError, match="cache miss"):
        second.score(dict(source_records[0], text="A different sentence ends."))
    third = DocumentScorer(replace(recipe, score_block_tokens=7), cache, torch.device("cpu"))
    third.score(source_records[0])
    assert len(cache.read_text().splitlines()) == 2


def test_review_protocol_and_original_sentence_blocks(tokenizer):
    text = "First sentence ends.\nSecond sentence follows. Next paragraph ends."
    blocks = review_blocks(text, tokenizer, 8)
    assert blocks[0].start == 0 and blocks[-1].end == len(text)
    assert all(text[start:end].endswith(".") for block in blocks for start, end in block.sentences)
    assert parse_review('{"decision":"keep","reason":"local noise","rejected_sentences":[1]}', 2)[
        "rejected_sentences"
    ] == [1]
    for raw in (
        '{"decision":"keep","reason":"ok","rejected_sentences":[2]}',
        '{"decision":"keep","reason":"ok","rejected_sentences":[true]}',
        '{"decision":"keep","reason":"ok","rejected_sentences":[1,1]}',
        '{"decision":"keep","reason":"ok","rejected_sentences":[],"extra":1}',
    ):
        with pytest.raises(ValueError):
            parse_review(raw, 2)


def test_near_duplicates_cluster_before_split_with_transitive_membership():
    base = [f"word{i}" for i in range(100)]
    variants = [base, ["changed"] + base[1:], ["changed"] + base[1:-1] + ["different"]]
    records = [
        {"id": str(i), "url": f"https://example.org/{i}", "text": " ".join(words)}
        for i, words in enumerate(variants + [[f"distinct{i}" for i in range(100)]])
    ]
    clusters = cluster_documents(records, PreparationConfig())
    assert clusters[0] == clusters[1] == clusters[2]
    assert clusters[3] != clusters[0]


def test_pipeline_local_exclusion_keeps_other_spans_and_publishes_audit(
    tmp_path, tiny_config, tokenizer, source_records, monkeypatch
):
    text = "First sentence ends. Bad sentence follows.\nSecond sentence ends."
    begin, end = text.index("Bad"), text.index("\n")
    records = [
        dict(r, text=text + f" Source number {i} is here.")
        for i, r in enumerate(source_records[:12])
    ]
    recipe = PreparationConfig(max_documents=12)
    scorer = DocumentScorer(recipe, tmp_path / "cache.jsonl", torch.device("cpu"))
    monkeypatch.setattr(
        scorer,
        "score",
        lambda record: {
            "review": [
                {
                    "char_span": [0, len(record["text"])],
                    "decision": "keep",
                    "reason": "local defect",
                    "excluded_spans": [[begin, end]],
                    "raw_output": None,
                }
            ]
        },
    )
    output = tmp_path / "data"
    metadata = prepare_fineweb(records, tokenizer, tiny_config, output, recipe, scorer=scorer)
    assert metadata["audit"]["checks"]["excluded_spans_absent"]
    for split in ("train", "dev", "test"):
        episodes = read_episodes(output / f"{split}.jsonl")
        for episode in episodes:
            assert all("Bad sentence" not in read.references[0].text for read in episode.reads)
    assert (output / "audit-random-documents.jsonl").exists()
    assert (output / "preparation.json").exists()
    assert metadata["audit"]["quality_audit"]["status"] == "pending_review"
    with pytest.raises(ValueError, match="document budget"):
        prepare_fineweb(
            records[:1],
            tokenizer,
            tiny_config,
            tmp_path / "incomplete",
            replace(recipe, split_document_limits=(20, 20, 20)),
        )
    assert not (tmp_path / "incomplete/preparation.json").exists()
    with pytest.raises(ValueError, match="no usable"):
        prepare_fineweb(
            [dict(records[0], language="fr")],
            tokenizer,
            tiny_config,
            tmp_path / "empty",
            recipe,
        )
    assert not (tmp_path / "empty/preparation.json").exists()


def test_length_sampling_controls_exposure_and_resumes_exactly(
    tmp_path, tiny_config, tokenizer, source_records
):
    config = replace(
        tiny_config, input_length_bounds=(8, 32, 64), input_length_weights=(0.2, 0.3, 0.5)
    )
    output = tmp_path / "balanced"
    metadata = prepare_fineweb(
        source_records,
        tokenizer,
        config,
        output,
        PreparationConfig(max_documents=len(source_records)),
    )
    preview = metadata["audit"]["sampling_preview"]["length_groups"]
    assert preview["64"]["samples"] / 1000 == pytest.approx(0.5, abs=0.06)
    index = EpisodeIndex(output / "train.jsonl")
    first = PretrainSampler(index, tokenizer, config)
    for step in range(13):
        first.sample(step)
    saved = first.state_dict()
    expected = [first.sample(i) for i in range(50)]
    resumed = PretrainSampler(index, tokenizer, config)
    resumed.load_state_dict(saved)
    assert expected == [resumed.sample(i) for i in range(50)]


def test_mixed_ae_only_objective_and_microbatch_gradients(
    tmp_path, tiny_config, tokenizer, source_records, components
):
    paired = next(
        e for e in document_episodes(source_records[0], tokenizer, tiny_config) if len(e.reads) == 2
    )
    ae, lm = read_tokens(paired, tokenizer)
    only = replace(paired, reads=(paired.reads[0],))
    examples = [
        PretrainExample(paired, ae, lm, 4),
        PretrainExample(only, ae, None, 4),
        PretrainExample(only, ae, None, 4),
    ]
    backbone, writer = components
    full = pretrain_forward(tiny_config, backbone, writer, examples)
    assert full.lm[1:] == [None, None]
    expected = (
        tiny_config.ae_weight * torch.stack([r.mean_nll for r in full.ae]).mean()
        + tiny_config.lm_weight * full.lm[0].mean_nll
    )
    torch.testing.assert_close(full.loss, expected)
    full.loss.backward()
    expected_gradients = {
        name: p.grad.clone() for name, p in writer.named_parameters() if p.grad is not None
    }
    backbone.zero_grad(set_to_none=True)
    writer.zero_grad(set_to_none=True)
    for batch in (examples[:1], examples[1:]):
        pretrain_forward(tiny_config, backbone, writer, batch, (3, 1)).loss.backward()
    for name, parameter in writer.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(
                parameter.grad, expected_gradients[name], atol=1e-6, rtol=1e-4
            )
    trainer = PretrainTrainer(tiny_config, backbone, writer, torch.device("cpu"))
    result = trainer.step(examples[1:])
    assert all(
        row["lm_nll"] is None and row["continuation_tokens"] == 0 for row in result["samples"]
    )
    assert result["target_tokens"] == 2 * len(ae.target_ids)


def test_ae_only_evaluation_emits_only_available_tasks(
    tmp_path, tiny_config, tokenizer, source_records, components
):
    records = [
        dict(record, text=f"Source number {i} contains one sentence.")
        for i, record in enumerate(source_records)
    ]
    directory = tmp_path / "ae-only"
    prepare_fineweb(
        records, tokenizer, tiny_config, directory, PreparationConfig(max_documents=len(records))
    )
    backbone, writer = components
    metrics = evaluate_pretraining(
        tiny_config,
        tokenizer,
        backbone,
        writer,
        EpisodeIndex(directory / "dev.jsonl"),
        tmp_path / "eval",
        0,
        0,
    )
    assert "all/ae/memory" in metrics["groups"]
    assert "all/continuation/memory" not in metrics["groups"]
    assert metrics["groups"]["all/ae/memory"]["generated_reads"] > 0


def test_local_parquet_scoring_preparation_and_training_need_no_network(
    tmp_path, tiny_config, tokenizer, source_records, monkeypatch
):
    model_path = tmp_path / "tiny-local-model"
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
        )
    )
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    config = replace(tiny_config, model_name_or_path=str(model_path))
    config_path = tmp_path / "config.json"
    write_resolved_config(config, config_path)
    recipe = PreparationConfig(
        max_documents=len(source_records),
        fluency_model_name_or_path=str(model_path),
        score_block_tokens=8,
    )
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe.to_dict()))
    source = tmp_path / "HuggingFaceFW-fineweb/sample-10BT"
    source.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(source_records), source / "input.parquet")

    def reject_network(*args, **kwargs):
        raise AssertionError("the complete preparation and training path must remain local")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    directory = tmp_path / "prepared"
    prepare_main(
        [
            "--config",
            str(config_path),
            "--recipe",
            str(recipe_path),
            "--score-cache",
            str(tmp_path / "score-cache.jsonl"),
            "--dataset-dir",
            str(source.parent),
            "--output-dir",
            str(directory),
        ]
    )
    metadata = json.loads((directory / "preparation.json").read_text())
    assert metadata["scoring_protocols"]["fluency"]["method"] == "independent_full_coverage"
    assert (tmp_path / "score-cache.jsonl").exists()
    result = run_pretraining(
        config, directory, tmp_path / "training", torch.device("cpu"), max_steps=1
    )
    assert result.completed_steps == 1 and result.final_checkpoint.exists()
