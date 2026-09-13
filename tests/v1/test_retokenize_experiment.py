import json
from dataclasses import replace

import pytest

from latent_working_memory.data_preparation.retokenize_experiment import prepare
from latent_working_memory.v1.data import Episode
from latent_working_memory.v1.sampling import read_tokens


def test_retokenization_uses_original_spans_and_preserves_split(tmp_path, tokenizer, tiny_config):
    source = tmp_path / 'old'
    source.mkdir()
    (source / 'preparation.json').write_text(json.dumps({'preparation_id': 'original', 'contract': {}, 'source_weights': {'semantic': 1}}))
    raw = 'A B C D'
    rows = []
    for task in ['ae', 'continuation']:
        rows.append({'episode_id': task, 'input_ids': [999], 'write_ends': [1],
                     'sources': [{'source_id': task, 'document_id': task, 'token_start': 0,
                                  'token_end': 1, 'provenance': {
                                      'x_char_span': [0, 3], 'y_char_span': [4, 7] if task == 'continuation' else None,
                                      'boundary_variant': 'semantic', 'tokenizer_name_or_path': 'old'}}],
                     'reads': [{'read_id': task, 'task': task, 'prefix_end': 1,
                                'prompt': tiny_config.ae_prompt if task == 'ae' else tiny_config.lm_prompt,
                                'references': [{'text': 'A B' if task == 'ae' else 'C D', 'evidence_spans': []}]}]})
    (source / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    pool = tmp_path / 'sources.jsonl'
    pool.write_text(''.join(json.dumps({'record': {'id': t, 'text': raw}, 'split': 'train',
                                      'status': 'eligible'}) + '\n' for t in ['ae', 'continuation']))
    spec = {'source_pool': str(pool), 'datasets': {'mixed': {'train': str(source / 'train.jsonl')}}}
    config = replace(tiny_config, max_input_tokens=2)
    output = tmp_path / 'new'
    report = prepare(spec, config, tokenizer, output)
    assert report['datasets']['mixed']['train']['retained'] == 2
    assert json.loads((output / 'mixed/preparation.json').read_text())['source_weights'] == {'semantic': 1}
    saved = [json.loads(l) for l in (output / 'mixed/train.jsonl').read_text().splitlines()]
    for row in saved:
        episode = Episode.from_record(row)
        assert episode.input_ids == tuple(tokenizer.encode('A B', add_special_tokens=False))
        assert episode.write_ends == (2,)
        assert episode.sources[0].token_end == episode.reads[0].prefix_end == 2
        read_tokens(episode, tokenizer)
    pool.write_text(pool.read_text().replace('"train"', '"test"'))
    with pytest.raises(ValueError, match='split differs'):
        prepare(spec, config, tokenizer, tmp_path / 'wrong-split')
