"""Batch native harness generation only across equally long E²-TTT inputs."""

import json

from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import has_bos_prefix
from tqdm import tqdm


class EqualLengthHFLM(HFLM):
    """Reuse native generation with equal input lengths after native truncation."""

    def generate_until(self, requests, disable_tqdm=False):
        groups = {}
        for index, request in enumerate(requests):
            context, kwargs = request.args
            max_context = self.max_length - kwargs.get("max_gen_toks", self.max_gen_toks)
            if max_context <= 0:
                raise ValueError("The generation budget must be below the sequence limit")
            # Use the same tokenizer entry point as HFLM.generate_until. The
            # singleton has no padding; the native left slice is applied below.
            tokens, _ = self.tok_batch_encode([context], truncation=self.truncation)
            length = min(tokens.shape[1], max_context)
            key = (
                length,
                json.dumps(kwargs, sort_keys=True),
                has_bos_prefix(context, getattr(self.tokenizer, "bos_token", None)),
            )
            groups.setdefault(key, []).append((index, request))

        results = [None] * len(requests)
        with tqdm(
            total=len(requests),
            disable=disable_tqdm or self.rank != 0,
            desc="Running equal-length generate_until requests",
        ) as progress:
            # Start with long inputs, as the native harness does. Each call
            # keeps native generation, stop handling, cache hooks and tail size.
            for _, group in sorted(groups.items(), key=lambda item: -item[0][0]):
                generated = super().generate_until(
                    [request for _, request in group], disable_tqdm=True
                )
                for (index, _), text in zip(group, generated, strict=True):
                    results[index] = text
                progress.update(len(group))
        return results
