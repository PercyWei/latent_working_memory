"""Use native length sorting and left padding for E²-TTT generation."""

import json

from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import has_bos_prefix
from tqdm import tqdm


class LengthSortedHFLM(HFLM):
    """Keep kwargs/BOS groups, then reuse native adjacent length-sorted batches."""

    def generate_until(self, requests, disable_tqdm=False):
        groups = {}
        for index, request in enumerate(requests):
            context, kwargs = request.args
            max_context = self.max_length - kwargs.get("max_gen_toks", self.max_gen_toks)
            if max_context <= 0:
                raise ValueError("The generation budget must be below the sequence limit")
            key = (
                json.dumps(kwargs, sort_keys=True),
                has_bos_prefix(context, getattr(self.tokenizer, "bos_token", None)),
            )
            groups.setdefault(key, []).append((index, request))

        results = [None] * len(requests)
        with tqdm(
            total=len(requests),
            disable=disable_tqdm or self.rank != 0,
            desc="Running length-sorted generate_until requests",
        ) as progress:
            # Native HFLM sorts each group from long to short, takes adjacent
            # batches, left-pads to their longest input and restores its order.
            # BOS groups must remain separate: tok_batch_encode uses row 0 to
            # decide whether to add special tokens for the entire batch.
            for group in groups.values():
                generated = super().generate_until(
                    [request for _, request in group], disable_tqdm=True
                )
                for (index, _), text in zip(group, generated, strict=True):
                    results[index] = text
                progress.update(len(group))
        return results
