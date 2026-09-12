import json
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from latent_working_memory.data_preparation.pipeline import prepare_fineweb
from latent_working_memory.v1.checkpoint import load_model_checkpoint
from latent_working_memory.v1.training import run_pretraining

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


def test_fork_inherits_optimizer_and_continues_with_joint_batch(
    tmp_path, tiny_config, tokenizer, preparation_records, preparation_recipe
):
    model_dir = tmp_path / 'model'
    LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    config = replace(tiny_config, model_name_or_path=str(model_dir),
                     pretrain_balanced_batches=True, ae_weight=1, lm_weight=0,
                     cache_text_features=True, eval_generation_examples=0, split_fractions=(.6, .2, .2))
    preparation_recipe = replace(preparation_recipe, samples_per_task=(16, 16, 16),
                                 candidates_per_document=16)
    root = tmp_path / 'data'
    prepare_fineweb(preparation_records, tokenizer, config, root, preparation_recipe)
    mixed = root / 'mixed'
    mixed.mkdir()
    (mixed / 'train.jsonl').write_text(
        (root / 'semantic/train.jsonl').read_text() + (root / 'random/train.jsonl').read_text())
    (mixed / 'preparation.json').write_text((root / 'semantic/preparation.json').read_text())
    evaluation_dirs = {name: root / name for name in ('semantic', 'random')}
    first = run_pretraining(config, mixed, tmp_path / 'ae', torch.device('cpu'),
                            max_steps=1, save_every=1, evaluation_dirs=evaluation_dirs)
    joint_config = replace(config, ae_weight=.5, lm_weight=.5, pretrain_ae_warmup_steps=1)
    second = run_pretraining(joint_config, mixed, tmp_path / 'warmup', torch.device('cpu'),
                             max_steps=2, save_every=1, evaluation_dirs=evaluation_dirs,
                             fork_from=first.final_checkpoint)
    checkpoint = load_model_checkpoint(second.final_checkpoint)
    assert checkpoint.progress['next_step'] == 2
    assert checkpoint.progress['sampler']['visits'] == 8
    assert all(state['step'].item() == 2 for state in checkpoint.optimizer_state['state'].values())
    log = next((tmp_path / 'warmup').glob('train-from-*.jsonl'))
    record = json.loads(log.read_text())
    assert record['step'] == 2 and record['task_weights'] == {'ae': .5, 'continuation': .5}
    assert sum(s['ae_nll'] is not None for s in record['samples']) == 2
    provenance = json.loads((tmp_path / 'warmup/provenance.json').read_text())
    assert provenance['inherited_steps'] == 1
