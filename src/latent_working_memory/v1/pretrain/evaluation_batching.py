"""Pack unchanged evaluation requests by adapter state, lengths and token budgets."""

from dataclasses import dataclass, fields
from itertools import groupby

from torch import Tensor

from latent_working_memory.v1.backbone import ReadTokens


@dataclass(frozen=True, slots=True)
class EvaluationBatching:
    read_batch_size: int = 8
    read_context_tokens: int = 16384
    read_target_tokens: int = 8192
    generation_batch_size: int = 8
    generation_context_tokens: int = 32768

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer")


@dataclass(slots=True)
class ReadJob:
    records: list[dict]
    memory: Tensor
    task: ReadTokens
    text_context: tuple[int, ...] = ()
    use_lora: bool = True

    @property
    def input_length(self):
        return (
            1
            + len(self.memory)
            + len(self.text_context)
            + len(self.task.prompt_ids)
            + self.target_length
        )

    @property
    def target_length(self):
        return len(self.task.target_ids)


@dataclass(slots=True)
class GenerationJob:
    records: list[dict]
    memory: Tensor
    prompt: tuple[int, ...]
    task: ReadTokens
    use_lora: bool

    @property
    def input_length(self):
        return 1 + len(self.memory) + len(self.prompt)

    @property
    def target_length(self):
        return len(self.task.target_ids)


def _pack(jobs, batch_size, context_tokens, target_tokens=None, generation=False):
    batch = []
    longest_input = longest_target = total_targets = 0
    for job in jobs:
        single_context = job.input_length + (job.target_length if generation else 0)
        if single_context > context_tokens or (
            target_tokens is not None and job.target_length > target_tokens
        ):
            raise ValueError("an evaluation request exceeds its batch token budget")
        n = len(batch) + 1
        input_length = max(longest_input, job.input_length)
        target_length = max(longest_target, job.target_length)
        padded = n * (input_length + (target_length if generation else 0))
        if batch and (
            n > batch_size
            or padded > context_tokens
            or (target_tokens is not None and total_targets + job.target_length > target_tokens)
        ):
            yield batch
            batch = []
            longest_input = longest_target = total_targets = 0
        batch.append(job)
        longest_input = max(longest_input, job.input_length)
        longest_target = max(longest_target, job.target_length)
        total_targets += job.target_length
    if batch:
        yield batch


def read_batches(jobs, limits):
    ordered = sorted(jobs, key=lambda job: (not job.use_lora, job.input_length, job.target_length))
    for _, group in groupby(ordered, key=lambda job: job.use_lora):
        yield from _pack(
            group, limits.read_batch_size, limits.read_context_tokens, limits.read_target_tokens
        )


def generation_batches(jobs, limits):
    # Bucket the generation limit first: equal P+G need not mean comparable P and G.
    def key(job):
        return not job.use_lora, (job.target_length - 1).bit_length()

    ordered = sorted(jobs, key=lambda job: (*key(job), job.input_length, job.target_length))
    for _, group in groupby(ordered, key=key):
        yield from _pack(
            group, limits.generation_batch_size, limits.generation_context_tokens, generation=True
        )
