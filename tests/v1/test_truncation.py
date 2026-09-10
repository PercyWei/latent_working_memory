from __future__ import annotations

import random
from dataclasses import replace

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

from latent_working_memory.data_preparation.truncation import RandomSpans


def test_random_samples_obey_actual_length_interval_and_raw_continuity(
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
):
    record = preparation_records[0]
    sampler = RandomSpans(record, tokenizer, tiny_config, preparation_recipe)
    rng = random.Random(42)
    examples = [
        sampler.sample(task, 8, 32, rng) for task in ("ae", "continuation") for _ in range(60)
    ]
    retained = [e for e in examples if e is not None]
    assert retained
    assert any(
        not (
            e.sources[0].provenance["input_starts_at_sentence"]
            and e.sources[0].provenance["input_ends_at_sentence"]
        )
        for e in retained
    )
    for episode in retained:
        p = episode.sources[0].provenance
        start, end = p["x_char_span"]
        assert 8 <= len(episode.input_ids) <= 32
        assert (
            tuple(tokenizer.encode(record["text"][start:end], add_special_tokens=False))
            == episode.input_ids
        )
        if episode.reads[0].task == "continuation":
            assert p["y_char_span"][0] == end
            assert episode.reads[0].references[0].text == record["text"][slice(*p["y_char_span"])]
        else:
            assert episode.reads[0].references[0].text == record["text"][start:end]
        assert "pair_id" not in p


def test_random_byte_slices_handle_whitespace_and_subwords(
    tiny_config, preparation_records, preparation_recipe
):
    backend = Tokenizer(BPE(vocab={c: i for i, c in enumerate(ByteLevel.alphabet())}, merges=[]))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    record = dict(
        preparation_records[0],
        text="First sentence ends." + "\n" * 128 + "Second sentence follows.",
    )
    sampler = RandomSpans(record, tokenizer, tiny_config, preparation_recipe)
    rng = random.Random(7)
    retained = 0
    for _ in range(100):
        e = sampler.sample("continuation", 0, 32, rng)
        if e is not None:
            retained += 1
            assert e.reads[0].references[0].text.strip()
            p = e.sources[0].provenance
            assert record["text"][slice(*p["x_char_span"])].strip()
    assert retained > 0


def test_semantic_lm_writes_only_internal_prefix(
    tiny_config, tokenizer, preparation_records, semantic_examples
):
    for episode in semantic_examples(preparation_records[0], tokenizer, tiny_config):
        if episode.reads[0].task == "continuation":
            p = episode.sources[0].provenance
            assert p["x_char_span"][1] == p["y_char_span"][0] < p["y_char_span"][1]
            assert p["parent_char_span"] == [p["x_char_span"][0], p["y_char_span"][1]]


def test_continuation_has_no_eight_sentence_cap(
    tiny_config, tokenizer, preparation_records, preparation_recipe, semantic_examples
):
    record = dict(
        preparation_records[0], text=" ".join(f"Sentence number {i} ends." for i in range(20))
    )
    counts = []
    for seed in range(12):
        config = replace(tiny_config, data_seed=seed, max_continuation_tokens=128)
        for episode in semantic_examples(
            record,
            tokenizer,
            config,
            recipe=replace(
                preparation_recipe, max_sample_tokens=128, length_bounds=(8, 32, 64, 128)
            ),
        ):
            if episode.reads[0].task == "continuation":
                start, end = episode.sources[0].provenance["continuation_sentence_range"]
                counts.append(end - start)
    assert max(counts) > 8
