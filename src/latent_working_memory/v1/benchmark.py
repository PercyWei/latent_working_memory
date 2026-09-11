"""Benchmark the existing pretraining step without modifying experiment artifacts."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from latent_working_memory.v1.backbone import load_backbone
from latent_working_memory.v1.checkpoint import load_model_checkpoint, restore_rng_state
from latent_working_memory.v1.data import EpisodeIndex
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.sampling import PretrainExample, PretrainSampler, read_tokens
from latent_working_memory.v1.training import PretrainTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--steps', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() or args.steps < 1:
        raise ValueError('output must be new and steps must be positive')
    checkpoint = load_model_checkpoint(args.checkpoint)
    config = replace(checkpoint.config, batch_size=args.batch_size,
                     gradient_accumulation_steps=8 // args.batch_size,
                     gradient_checkpointing=args.gradient_checkpointing)
    device = torch.device('cuda:0')
    tokenizer, backbone = load_backbone(config, device, torch.bfloat16)
    writer = JointMemoryWriter(config.d_mem, config.num_layers, config.num_heads,
                               config.ffn_dim, config.k_limit).to(device)
    backbone.load_trainable_state_dict(checkpoint.model_state['backbone'])
    writer.load_state_dict(checkpoint.model_state['writer'])
    trainer = PretrainTrainer(config, backbone, writer, device)
    trainer.optimizer.load_state_dict(checkpoint.optimizer_state)
    index = EpisodeIndex(args.data)
    sampler = PretrainSampler(index, tokenizer, config)
    sampler.load_state_dict(checkpoint.progress['sampler'])
    restore_rng_state(checkpoint.rng_state)
    start = checkpoint.progress['next_step']
    # Identical sample stream across configurations; CPU sampling is excluded from timing.
    batches = [[sampler.sample(start + i) for _ in range(8)] for i in range(args.steps + 2)]
    stress = {}
    for task in ('ae', 'continuation'):
        ids = sorted((i for i, t in enumerate(index.tasks) if t == task),
                     key=lambda i: index.input_lengths[i], reverse=True)[:8]
        batch = []
        for i in ids:
            episode = index[i]
            ae, lm = read_tokens(episode, tokenizer)
            batch.append(PretrainExample(episode, ae, lm, math.ceil(len(episode.input_ids) / 2)))
        stress[task] = batch
    result = {'batch_size': args.batch_size, 'gradient_accumulation_steps': 8 // args.batch_size,
              'gradient_checkpointing': args.gradient_checkpointing, 'checkpoint': str(args.checkpoint),
              'data': str(args.data), 'steps': [], 'stress': {}, 'status': 'running'}
    try:
        for i, batch in enumerate(batches):
            if i == 2:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            row = trainer.step(batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            if not all(math.isfinite(row[k]) for k in ('loss', 'gradient_norm')):
                raise ValueError('nonfinite benchmark loss or gradient')
            if i >= 2:
                result['steps'].append({'seconds': elapsed, 'loss': row['loss'],
                                        'input_tokens': row['input_tokens'], 'target_tokens': row['target_tokens']})
        result['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        result['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        for task, batch in stress.items():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            row = trainer.step(batch)
            torch.cuda.synchronize()
            result['stress'][task] = {'seconds': time.perf_counter() - begin,
                                      'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                                      'input_lengths': [e.input_length for e in batch],
                                      'target_lengths': [len((e.ae or e.lm).target_ids) for e in batch]}
        result['status'] = 'complete'
    except torch.cuda.OutOfMemoryError:
        result['status'] = 'oom'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'steps'}), flush=True)


if __name__ == '__main__':
    main()
