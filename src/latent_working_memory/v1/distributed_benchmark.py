"""Synchronous data-parallel benchmark with one gradient reduction per update."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from latent_working_memory.v1.backbone import load_backbone
from latent_working_memory.v1.checkpoint import load_model_checkpoint, restore_rng_state
from latent_working_memory.v1.data import EpisodeIndex
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.sampling import PretrainExample, PretrainSampler, read_tokens
from latent_working_memory.v1.training import PretrainTrainer, precision_context, pretrain_forward


def synchronize_gradients(parameters):
    """Sum globally normalized local gradients; preserve globally unused parameters."""
    used = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    active = [p for p, present in zip(parameters, used.tolist(), strict=True) if present]
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in active])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for p in active:
        p.grad = flat[offset:offset + p.numel()].view_as(p)
        offset += p.numel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--global-batch', type=int, choices=(8, 16), default=8)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--steps', type=int, default=32)
    args = parser.parse_args()
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    if args.output.exists() or args.global_batch % (world * args.batch_size):
        raise ValueError('output must be new; global batch must divide across microbatches')
    checkpoint = load_model_checkpoint(args.checkpoint)
    config = replace(checkpoint.config, batch_size=args.batch_size,
                     gradient_accumulation_steps=args.global_batch // (world * args.batch_size),
                     gradient_checkpointing=args.gradient_checkpointing)
    device = torch.device('cuda', rank)
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
    # Checkpoint has one CUDA RNG stream; map it to this rank's logical device.
    rng = checkpoint.rng_state.copy()
    rng['cuda'] = tuple(checkpoint.rng_state['cuda'][0] for _ in range(torch.cuda.device_count()))
    restore_rng_state(rng)
    start = checkpoint.progress['next_step']
    batches = [[sampler.sample(start + i) for _ in range(args.global_batch)] for i in range(args.steps + 2)]
    stress = {}
    for task in ('ae', 'continuation'):
        ids = sorted((i for i, t in enumerate(index.tasks) if t == task),
                     key=lambda i: index.input_lengths[i], reverse=True)[:args.global_batch]
        batch = []
        for i in ids:
            episode = index[i]
            ae, lm = read_tokens(episode, tokenizer)
            batch.append(PretrainExample(episode, ae, lm, math.ceil(len(episode.input_ids) / 2)))
        stress[task] = batch

    def step(examples):
        # Interleave by length so every rank receives similar work.
        ordered = sorted(examples, key=lambda e: max(e.input_length, len(e.lm.target_ids) if e.lm else 0) + e.capacity)
        local = ordered[rank::world]
        counts = (sum(e.ae is not None for e in examples), sum(e.lm is not None for e in examples))
        backbone.train()
        writer.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        if world > 1:
            dist.barrier()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        loss = 0.0
        for i in range(0, len(local), config.batch_size):
            with precision_context(device):
                output = pretrain_forward(config, backbone, writer, local[i:i+config.batch_size], counts)
            output.loss.backward()
            loss += float(output.loss.detach())
            del output
        torch.cuda.synchronize()
        communication_begin = time.perf_counter()
        if world > 1:
            synchronize_gradients(trainer.parameters)
        torch.cuda.synchronize()
        communication = time.perf_counter() - communication_begin
        norm = torch.nn.utils.clip_grad_norm_(trainer.parameters, config.gradient_clip, error_if_nonfinite=True)
        trainer.optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        stats = torch.tensor([seconds, torch.cuda.max_memory_allocated()/2**30, communication], device=device)
        total_loss = torch.tensor(loss, device=device)
        if world > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            dist.all_reduce(total_loss)
        if not math.isfinite(total_loss.item()):
            raise ValueError('nonfinite loss')
        return {'seconds': stats[0].item(), 'peak_allocated_gib': stats[1].item(),
                'gradient_sync_seconds': stats[2].item(), 'loss': total_loss.item(), 'gradient_norm': float(norm),
                'samples': len(examples), 'input_tokens': sum(e.input_length for e in examples),
                'target_tokens': sum(len((e.ae or e.lm).target_ids) for e in examples)}

    result = {'world_size': world, 'microbatch': args.batch_size, 'global_batch': args.global_batch,
              'gradient_accumulation_steps': config.gradient_accumulation_steps,
              'gradient_checkpointing': config.gradient_checkpointing,
              'checkpoint': str(args.checkpoint), 'steps': [], 'stress': {}}
    for i, batch in enumerate(batches):
        if i == 2:
            torch.cuda.reset_peak_memory_stats()
        row = step(batch)
        if i >= 2:
            result['steps'].append(row)
    for task, batch in stress.items():
        torch.cuda.reset_peak_memory_stats()
        result['stress'][task] = step(batch)
    if world > 1:
        # Verify replica agreement after repeated optimizer updates.
        errors = []
        for p in trainer.parameters:
            reference = p.detach().clone()
            dist.broadcast(reference, 0)
            errors.append((reference - p).abs().max())
        error = torch.stack(errors).max()
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        result['replica_max_abs_diff'] = error.item()
        if error.item() != 0:
            raise ValueError('replicas diverged')
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print('complete', args.output, flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
