from __future__ import annotations

import json
from latent_working_memory.v1.data import Episode
from latent_working_memory.data_preparation.segmentation import sentence_spans
from latent_working_memory.v1.sampling import capacity_weights, read_tokens


def test_natural_views_preserve_exact_original_text(
    tiny_config, tokenizer, source_records, semantic_examples
):
    record = source_records[0]
    episodes = semantic_examples(record, tokenizer, tiny_config)
    assert all("granularity" not in e.sources[0].provenance for e in episodes)
    assert any(
        e.sources[0].provenance["sentence_range"][1] - e.sources[0].provenance["sentence_range"][0]
        == 1
        for e in episodes
    )
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
    quoted = "“We left. He stayed.”"
    spans = sentence_spans(quoted)
    assert [quoted[s.start : s.end] for s in spans] == ["“We left. He stayed.”"]
