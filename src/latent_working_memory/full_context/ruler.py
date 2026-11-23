"""按 RULER 长度档位与官方生成预算组织每卡批量推理。"""

from collections import defaultdict
import time

import torch

from latent_working_memory.benchmarks.protocol import encode_prompt

from .generation import generated_ids, left_pad


def _batches(samples, batch_size, batch_size_by_length):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.metadata["length"], sample.max_new_tokens].append(sample)
    for (length, _), group in sorted(groups.items()):
        group.sort(key=lambda sample: len(sample.prompt))
        size = batch_size_by_length.get(length, batch_size)
        for start in range(0, len(group), size):
            yield group[start:start + size]


def evaluate_ruler(samples, tokenizer, model, generation, model_window, device, batch_size,
                   batch_size_by_length, commit):
    for batch in _batches(samples, batch_size, batch_size_by_length):
        prompts = [
            encode_prompt(sample, tokenizer, model_window, "ruler", use_qa_system=True)
            for sample in batch
        ]
        inputs, mask = left_pad(prompts, generation.pad_token_id, device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        outputs = model.generate(
            input_ids=inputs, attention_mask=mask, generation_config=generation,
            max_new_tokens=batch[0].max_new_tokens, use_model_defaults=False,
        )
        torch.cuda.synchronize(device)
        seconds_per_question = (time.perf_counter() - started) / len(batch)
        rows = outputs[:, inputs.shape[1]:].tolist()
        for sample, prompt, row in zip(batch, prompts, rows, strict=True):
            answer = generated_ids(row, generation.eos_token_id)
            commit(sample, len(prompt), answer, seconds_per_question, len(batch), len(batch))
