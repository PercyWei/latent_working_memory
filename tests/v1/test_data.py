from __future__ import annotations

import json
import pytest
from latent_working_memory.v1.data import Episode
from latent_working_memory.data_preparation.fineweb import validate_topic_ranges
from latent_working_memory.data_preparation.segmentation import sentence_spans
from latent_working_memory.v1.sampling import capacity_weights, read_tokens


def test_natural_views_preserve_exact_original_text(
    tiny_config, tokenizer, source_records, semantic_examples
):
    record = source_records[0]
    episodes = semantic_examples(record, tokenizer, tiny_config)
    assert {"paragraph", "sentence", "sentence_group"} <= {
        e.sources[0].provenance["granularity"] for e in episodes
    }
    assert len({len(e.input_ids) for e in episodes}) > 2
    for episode in episodes:
        assert Episode.from_record(json.loads(json.dumps(episode.to_record()))) == episode
        provenance = episode.sources[0].provenance
        xs, xe = provenance["x_char_span"]
        assert (
            tuple(tokenizer.encode(record["text"][xs:xe], add_special_tokens=False))
            == episode.input_ids
        )
        ae, lm = read_tokens(episode, tokenizer)
        if ae is not None:
            assert tuple(ae.target_ids[:-1]) == episode.input_ids
        if lm is not None:
            ys, ye = provenance["y_char_span"]
            assert xe == ys
            assert record["text"][ys:ye] == episode.reads[0].references[0].text
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
    assert [quoted[s.start : s.end] for s in spans] == ["“We left. He stayed.”"]


def test_topic_annotations_require_order_range_and_coverage(
    tiny_config, tokenizer, source_records, semantic_examples
):
    record = source_records[0]
    count = len(sentence_spans(record["text"]))
    annotation = {"ranges": [[0, 2], [2, count]], "model": "offline-fixture"}
    episodes = semantic_examples(record, tokenizer, tiny_config, annotation)
    topic_episodes = [
        e for e in episodes if e.sources[0].provenance["granularity"] == "topic_group"
    ]
    assert {e.reads[0].task for e in topic_episodes} == {"ae", "continuation"}
    assert all(
        e.sources[0].provenance["boundary_model"] == "offline-fixture" for e in topic_episodes
    )
    for ranges in ([[1, count]], [[0, 3], [2, count]], [[0, count + 1]], [[0, count - 1]]):
        with pytest.raises(ValueError, match="topic ranges"):
            validate_topic_ranges(ranges, count)
