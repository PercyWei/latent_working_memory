from collections import Counter
from dataclasses import replace

import pytest

from latent_working_memory.v1.data import EpisodeIndex, write_episodes
from latent_working_memory.v1.sampling import BalancedPretrainSampler, task_weights_at


def test_balanced_task_source_batches_and_warmup_boundary(
    tmp_path, tiny_config, tokenizer, source_records, semantic_examples
):
    rows = []
    for source in ('semantic', 'random'):
        for number, record in enumerate(source_records[:4]):
            examples = semantic_examples(record, tokenizer, tiny_config)
            for task in ('ae', 'continuation'):
                episode = next(e for e in examples if e.reads[0].task == task)
                episode = replace(episode, episode_id=f'{source}:{number}:{task}')
                episode.sources[0].provenance.update(
                    boundary_variant=source, dedup_cluster=record['id'],
                )
                rows.append(episode)
    path = tmp_path / 'train.jsonl'
    write_episodes(rows, path)
    index = EpisodeIndex(path)
    config = replace(tiny_config, pretrain_balanced_batches=True,
                     ae_weight=.5, lm_weight=.5, pretrain_ae_warmup_steps=2)
    sampler = BalancedPretrainSampler(index, tokenizer, config)
    assert task_weights_at(config, 1) == (1., 0.)
    assert task_weights_at(config, 2) == (.5, .5)
    for step in range(4):
        batch = sampler.sample_batch(step, 8)
        counts = Counter((e.episode.reads[0].task,
                          e.episode.sources[0].provenance['boundary_variant']) for e in batch)
        tasks = ('ae',) if step < 2 else ('ae', 'continuation')
        assert counts == {(task, source): 8 // (2 * len(tasks))
                          for task in tasks for source in ('semantic', 'random')}
    assert sampler.visits == 32
    with pytest.raises(ValueError, match='divide evenly'):
        sampler.sample_batch(3, 6)


def test_warmup_configuration_requires_enabled_joint_objective(tiny_config):
    with pytest.raises(ValueError, match='AE warm-up'):
        replace(tiny_config, pretrain_ae_warmup_steps=5)
