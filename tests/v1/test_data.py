from __future__ import annotations

import json
import socket

import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import replace

import pytest

from latent_working_memory.v1.config import write_resolved_config
from latent_working_memory.data_preparation.pipeline import prepare_fineweb
from latent_working_memory.data_preparation.__main__ import main as prepare_main
from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.v1.data import Episode, EpisodeIndex, read_episodes
from latent_working_memory.data_preparation.fineweb import (
    document_episodes,
    document_split,
    sentence_spans,
    source_key,
    validate_topic_ranges,
)
from latent_working_memory.v1.sampling import PretrainSampler, capacity_weights, read_tokens
from latent_working_memory.data_preparation.quality import sentence_rejection_reason


def test_natural_views_preserve_exact_original_text(tiny_config, tokenizer, source_records):
    record = source_records[0]
    episodes = document_episodes(record, tokenizer, tiny_config)
    assert {"paragraph", "sentence", "sentence_group", "paragraph_group"} <= {
        e.sources[0].provenance["granularity"] for e in episodes
    }
    assert len({len(e.input_ids) for e in episodes}) > 2
    for episode in episodes:
        assert Episode.from_record(json.loads(json.dumps(episode.to_record()))) == episode
        provenance = episode.sources[0].provenance
        xs, xe = provenance["x_char_span"]
        assert record["text"][xs:xe] == episode.reads[0].references[0].text
        ae, lm = read_tokens(episode, tokenizer)
        assert tuple(ae.target_ids[:-1]) == episode.input_ids
        if lm is not None:
            ys, ye = provenance["y_char_span"]
            assert xe == ys
            assert record["text"][ys:ye] == episode.reads[1].references[0].text
            assert len(lm.target_ids) > 1
        else:
            assert provenance["y_char_span"] is None
        assert capacity_weights(tiny_config, len(episode.input_ids), ae, lm, 0)


def test_sentence_spans_handle_abbreviations_and_keep_offsets():
    text = "Dr. Smith stayed. Next sentence!\nThird paragraph ends."
    spans = sentence_spans(text)
    assert [text[s.start : s.end] for s in spans] == [
        "Dr. Smith stayed.",
        "Next sentence!",
        "Third paragraph ends.",
    ]
    assert spans[0].paragraph == spans[1].paragraph != spans[2].paragraph
    quoted = "“We left. He stayed.”"
    spans = sentence_spans(quoted)
    assert [quoted[s.start : s.end] for s in spans] == ["“We left.", "He stayed.”"]


def test_quality_keeps_short_sentences_and_contiguous_targets(
    tiny_config, tokenizer, source_records
):
    record = dict(
        source_records[0],
        text=(
            "Navigation\nWait! We left the station. The train arrived.\nRead more.\n"
            "The river overflowed. Residents moved uphill. They returned safely.\nFinal menu"
        ),
    )
    episodes = document_episodes(record, tokenizer, replace(tiny_config, views_per_granularity=16))
    assert any(e.reads[0].references[0].text == "Wait!" for e in episodes)
    for episode in episodes:
        source = episode.sources[0].provenance
        xs, xe = source["x_char_span"]
        ye = xe
        combined = episode.reads[0].references[0].text
        if len(episode.reads) == 2:
            ys, ye = source["y_char_span"]
            assert xe == ys
            combined += episode.reads[1].references[0].text
        assert combined == record["text"][xs:ye]
        assert not any(noise in combined for noise in ("Navigation", "Read more", "Final menu"))
    assert sentence_rejection_reason("Dr. Smith left.") is None
    assert sentence_rejection_reason("“Go!”") is None
    assert sentence_rejection_reason("The story continues...") == "ellipsis_fragment"
    assert sentence_rejection_reason("Click here to log in.") == "boilerplate"
    assert sentence_rejection_reason("<div>We left.</div>") == "markup_or_code"
    assert (
        sentence_rejection_reason("Content belongs to the respective copyright holders.")
        == "boilerplate"
    )
    assert sentence_rejection_reason("A broken quot;titlequot; appeared.") == "markup_or_code"
    assert not document_episodes(
        dict(record, text="Cheap jerseys are here. Cheap jerseys for sale. Buy cheap jerseys."),
        tokenizer,
        tiny_config,
    )
    assert document_episodes(
        dict(
            record, text="The casino opened yesterday. Guests arrived early. They watched the show."
        ),
        tokenizer,
        tiny_config,
    )
    assert not document_episodes(dict(record, language_score=0.5), tokenizer, tiny_config)


def test_source_split_and_dedup_are_applied_before_views(
    tmp_path, tiny_config, tokenizer, source_records
):
    records = source_records + [
        dict(source_records[0], id="duplicate", url="https://another.org/x")
    ]
    output = tmp_path / "data"
    metadata = prepare_fineweb(
        records, tokenizer, tiny_config, output, PreparationConfig(max_documents=len(records))
    )
    assert metadata["statistics"]["duplicates_removed"] == 1
    sources = []
    for split in ("train", "dev", "test"):
        episodes = read_episodes(output / f"{split}.jsonl")
        assert episodes
        sources.append({e.sources[0].source_id for e in episodes})
        for episode in episodes:
            assert (
                document_split(episode.sources[0].provenance["dedup_cluster"], tiny_config) == split
            )
    assert (
        not sources[0] & sources[1] and not sources[0] & sources[2] and not sources[1] & sources[2]
    )
    assert source_key("http://Example.org/x/#a") == source_key("https://example.org/x")


def test_topic_annotations_require_order_range_and_coverage(tiny_config, tokenizer, source_records):
    record = source_records[0]
    count = len(sentence_spans(record["text"]))
    annotation = {"ranges": [[0, 2], [2, count]], "model": "offline-fixture"}
    episodes = document_episodes(record, tokenizer, tiny_config, annotation)
    assert any(e.sources[0].provenance["granularity"] == "topic_group" for e in episodes)
    for ranges in ([[1, count]], [[0, 3], [2, count]], [[0, count + 1]], [[0, count - 1]]):
        with pytest.raises(ValueError, match="topic ranges"):
            validate_topic_ranges(ranges, count)


def test_capacity_dedup_budgets_curriculum_and_sampler_resume(
    tmp_path, tiny_config, tokenizer, source_records
):
    output = tmp_path / "data"
    prepare_fineweb(
        source_records,
        tokenizer,
        tiny_config,
        output,
        PreparationConfig(max_documents=len(source_records)),
    )
    index = EpisodeIndex(output / "train.jsonl")
    sampler = PretrainSampler(index, tokenizer, tiny_config)
    first_cycle = [sampler.sample(0).episode.sources[0].document_id for _ in sampler.documents]
    assert len(set(first_cycle)) == len(sampler.documents)
    state = sampler.state_dict()
    expected = [sampler.sample(1000) for _ in range(8)]
    resumed = PretrainSampler(index, tokenizer, tiny_config)
    resumed.load_state_dict(state)
    assert [resumed.sample(1000) for _ in range(8)] == expected
    example = expected[0]
    clamped = replace(tiny_config, pretrain_k_min=32)
    assert capacity_weights(clamped, example.input_length, example.ae, example.lm, 0) == {32: 1.0}
    constrained = replace(tiny_config, read_context_tokens=4)
    assert capacity_weights(constrained, example.input_length, example.ae, example.lm, 0) == {}
    assert capacity_weights(
        tiny_config, example.input_length, example.ae, example.lm, 0
    ) != capacity_weights(tiny_config, example.input_length, example.ae, example.lm, 1000)


def test_preparation_does_not_consume_beyond_document_budget(
    tmp_path,
    tiny_config,
    tokenizer,
    source_records,
):
    consumed = []

    def records():
        for record in source_records:
            consumed.append(record["id"])
            yield record

    prepare_fineweb(
        records(), tokenizer, tiny_config, tmp_path / "budgeted", PreparationConfig(max_documents=3)
    )
    assert len(consumed) == 3


def test_preparation_split_quotas_and_stratified_independent_panel(
    tmp_path, tiny_config, tokenizer, source_records
):
    output = tmp_path / "quota"
    metadata = prepare_fineweb(
        source_records,
        tokenizer,
        tiny_config,
        output,
        PreparationConfig(max_documents=len(source_records), split_document_limits=(8, 2, 1)),
    )
    for split, count in (("train", 8), ("dev", 2), ("test", 1)):
        assert metadata["statistics"][f"{split}_documents"] == count
    index = EpisodeIndex(output / "train.jsonl")
    panel = index.evaluation_panel(8, 42)
    assert panel == index.evaluation_panel(8, 42)
    assert len({index[i].sources[0].document_id for i in panel}) == 8
    assert len({index[i].sources[0].provenance["granularity"] for i in panel}) >= 3


def test_prepare_entry_reads_downloaded_parquet_without_network(
    tmp_path,
    tiny_config,
    tokenizer,
    source_records,
    monkeypatch,
):
    model_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(model_dir)
    config = replace(tiny_config, model_name_or_path=str(model_dir))
    config_path = tmp_path / "config.json"
    write_resolved_config(config, config_path)
    repository = tmp_path / "HuggingFaceFW-fineweb"
    shard_dir = repository / "sample-10BT"
    shard_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(source_records), shard_dir / "000.parquet")
    opened = []
    original_parquet_file = pq.ParquetFile

    def open_parquet(path):
        parquet = original_parquet_file(path)
        opened.append(parquet)
        return parquet

    def reject_network(*args, **kwargs):
        raise AssertionError("local preparation must not access the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(pq, "ParquetFile", open_parquet)
    output_dir = tmp_path / "prepared"
    prepare_main(
        [
            "--config",
            str(config_path),
            "--dataset-dir",
            str(repository),
            "--output-dir",
            str(output_dir),
            "--max-documents",
            "3",
        ]
    )
    metadata = json.loads((output_dir / "preparation.json").read_text())
    assert metadata["statistics"]["documents_read"] == 3
    assert metadata["contract"]["pretrain_dataset"] == "HuggingFaceFW/fineweb"
    assert opened and all(parquet.closed for parquet in opened)
