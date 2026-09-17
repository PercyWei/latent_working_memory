"""每次压缩后的累计历史 AE 与紧邻续文 LM，按 token、压缩次数、轨迹归一化。"""

import torch
from torch import nn


class ReconstructionTask(nn.Module):
    def __init__(self, codec, tokenizer, config):
        super().__init__()
        self.codec, self.config = codec, config
        if tokenizer.eos_token_id is None:
            raise ValueError("reconstruction requires an EOS token")
        self.eos_id = tokenizer.eos_token_id
        for name in ("ae_prompt", "lm_prompt"):
            self.register_buffer(
                name,
                torch.tensor(
                    tokenizer.encode(getattr(config, name), add_special_tokens=False),
                    dtype=torch.long,
                ),
                persistent=False,
            )

    def validate_data(self, datasets):
        limit = self.codec.max_positions
        for splits in datasets.values():
            for rows in splits.values():
                for row in rows:
                    length, k = row.write_ends[-1], row.capacity
                    q = len(row.token_ids) - length
                    if q < 1 or row.token_ids.ndim != 1:
                        raise ValueError("trajectory requires tokens and a final continuation")
                    # Also reserve the one-shot evaluation input window.
                    costs = [
                        length,
                        k + len(self.ae_prompt) + length + 1,
                        k + len(self.lm_prompt) + q + 1,
                    ]
                    for i, (start, end) in enumerate(
                        zip((0,) + row.write_ends[:-1], row.write_ends, strict=True)
                    ):
                        if end <= start or k > end - start + (k if i else 0):
                            raise ValueError("invalid write boundaries or capacity")
                        costs.append(end - start + (k if i else 0))
                    if max(costs) > limit:
                        raise ValueError(
                            f"trajectory {row.document_id} needs {max(costs)} positions; model supports {limit}"
                        )

    def forward(self, trajectory, include_lm=None, one_shot=False):
        include_lm = self.config.objective == "ae_lm" if include_lm is None else include_lm
        single = not isinstance(trajectory, (tuple, list))
        rows = [trajectory] if single else trajectory
        ids = [row.token_ids.to(self.ae_prompt.device) for row in rows]
        eos = ids[0].new_tensor([self.eos_id])
        ends = [(row.write_ends[-1],) if one_shot else row.write_ends for row in rows]
        capacity, depth = rows[0].capacity, len(ends[0])
        if any(row.capacity != capacity or len(cuts) != depth for row, cuts in zip(rows, ends)):
            raise ValueError("a microbatch requires equal capacities and compression counts")
        q = [len(tokens) - row.write_ends[-1] for tokens, row in zip(ids, rows)]
        memory, previous, losses, values = None, [0] * len(rows), [], []
        for step in range(depth):
            current = [cuts[step] for cuts in ends]
            segments = [tokens[start:end] for tokens, start, end in zip(ids, previous, current)]
            memory = (
                self.codec.write(memory, segments[0], capacity)
                if single
                else self.codec.write_batch(memory, segments, capacity)
            )
            ae_targets = [torch.cat((tokens[:end], eos)) for tokens, end in zip(ids, current)]
            ae = (
                self.codec.read_loss(memory, self.ae_prompt, ae_targets[0])[None]
                if single
                else self.codec.read_loss_batch(memory, self.ae_prompt, ae_targets)
            )
            lm = None
            if include_lm:
                lm_targets = [
                    torch.cat((tokens[end : end + length], eos))
                    for tokens, end, length in zip(ids, current, q)
                ]
                lm = (
                    self.codec.read_loss(memory, self.lm_prompt, lm_targets[0])[None]
                    if single
                    else self.codec.read_loss_batch(memory, self.lm_prompt, lm_targets)
                )
            losses.append(
                ae + self.config.lm_weight * lm
                if lm is not None and self.config.objective == "ae_lm"
                else ae
            )
            values.append(torch.stack((ae, lm if lm is not None else torch.zeros_like(ae)), dim=1))
            previous = current
        # One device-to-host transfer per microbatch; reporting never synchronizes each read.
        values = torch.stack(values).detach().cpu().tolist()
        records = [
            [
                {
                    "round": step + 1,
                    "seen_tokens": end,
                    "ae": values[step][i][0],
                    "lm": values[step][i][1] if include_lm else None,
                    "ae_tokens": end + 1,
                    "lm_tokens": q[i] + 1 if include_lm else 0,
                }
                for step, end in enumerate(cuts)
            ]
            for i, cuts in enumerate(ends)
        ]
        sample_losses = torch.stack(losses).mean(0)
        if single:
            return {"loss": sample_losses[0], "rounds": records[0]}
        return {"loss": sample_losses.mean(), "sample_losses": sample_losses, "rounds": records}
